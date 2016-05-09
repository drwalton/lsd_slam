#include "DepthMapOmniStereo.hpp"
#include "DepthMap.hpp"
#include "globalFuncs.hpp"
#include "DataStructures/Frame.hpp"

#include <opencv2/opencv.hpp>

///This file contains the depth estimation functions specific to Omnidirectional
/// camera models.

#define SHOW_DEBUG_IMAGES 0

namespace lsd_slam {

///\brief Check if an epipolar line lies within an omnidirectional image.
///\note For now, just checks if the endpoints lie in the image.
bool epipolarLineInImageOmni(vec3 lineStart, vec3 lineEnd, const OmniCameraModel &model);

///\brief Fill an array with the 5 values on the epipolar line surrounding the 
///       chosen image point. Used by doOmniStereo().
std::array<float, 5> getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel,
	float u, float v);

///\brief Increase the size of an epipolar line, if its length is found
///       to be less than minLength.
void padEpipolarLineOmni(vec3 *lineStart, vec3 *lineEnd,
	vec2* lineStartPix, vec2* lineEndPix,
	float minLength,
	const OmniCameraModel &oModel);

bool subpixelMatchOmni() {
	//TODO
	return false;
}
struct Ray {
	vec3 origin;
	vec3 dir;
};

struct RayIntersectionResult {
	enum Outcome {
		VALID, BEHIND, PARALLEL
	};
	Outcome valid;
	float distance;
	vec3 position;
	std::string to_string() const;
};

RayIntersectionResult computeRayIntersection(const Ray &r1, const Ray &r2)
{
	RayIntersectionResult ans;
	const vec3& u = r1.dir, &v = r2.dir, &p = r1.origin, &q = r2.origin;
	vec3 d = p - q;
	float uu = u.dot(u), vd = v.dot(d), vv = v.dot(v), ud = u.dot(d), uv = u.dot(v);

	//Find line parameters at intersection.
	float kd = uu*vv - uv*uv;

	//Special case: if kd == 0.f, the lines are parallel.
	if (fabsf(kd) < FLT_EPSILON) {
		ans.valid = RayIntersectionResult::Outcome::PARALLEL;
		float tc = vd / vv;
		ans.distance = (p - (q + tc*v)).norm();
		//Can't find a single intersection point.
		return ans;
	}

	float sc = (uv*vd - vv*ud) / kd;
	float tc = (uu*vd - uv*ud) / kd;

	//The intersection is valid if it happens at positive s, t.
	if (sc >= 0.f && tc >= 0.f) {
		ans.valid = RayIntersectionResult::Outcome::VALID;
	}
	else {
		ans.valid = RayIntersectionResult::Outcome::BEHIND;
	}

	//Find intersection points on r1.
	vec3 ip = p + sc*u, iq = q + tc*v;

	//Get distance between these points.
	ans.distance = (ip - iq).norm();
	ans.position = ip;

	return ans;
}

float findDepthAndVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float *resultIDepth, float *resultVar,
	float gradAlongLine, const vec2 &bestEpDir, const Frame *referenceFrame, Frame *activeKeyFrame,
	float sampleDist, bool didSubpixel, float tracedLineLen,
	OmniCameraModel *model,
	RunningStats *stats)
{
	// ================= calc depth (in KF) ====================
	float idnew_best_match;	// depth in the new image
	float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
	vec3 findDirKf = model->pixelToCam(vec2(u, v));
	vec3 bestMatchDirKf = referenceFrame->thisToOther_R * bestMatchDir;
	Ray findRay, matchRay;
	findRay.dir = findDirKf; findRay.origin = vec3::Zero();
	matchRay.dir = bestMatchDirKf; matchRay.origin = referenceFrame->thisToOther_t;
	RayIntersectionResult r = computeRayIntersection(findRay, matchRay);

	if (r.valid == RayIntersectionResult::Outcome::BEHIND) {
		if (enablePrintDebugInfo) stats->num_stereo_negative++;
		if (!allowNegativeIdepths)
			return -2;
	} else if (r.valid == RayIntersectionResult::Outcome::PARALLEL) {
		//baseline too small
		return -2;
	}
	alpha = r.position.norm();
	idnew_best_match = 1.f / alpha;
	alpha /= tracedLineLen;
	if (enablePrintDebugInfo) stats->num_stereo_successfull++;

	// ================= calc var (in NEW image) ====================

	// calculate error from photometric noise
	float photoDispError = 4.0f * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);
	float trackingErrorFac = 0.25f*(1.0f + referenceFrame->initialTrackedResidual);

	// calculate error from geometric noise (wrong camera pose / calibration)
	Eigen::Vector2f gradsInterp = getInterpolatedElement42(activeKeyFrame->gradients(0), u, v, model->w);
	float geoDispError = (gradsInterp[0] * bestEpDir[0] + gradsInterp[1] * bestEpDir[1]) + DIVISION_EPS;
	geoDispError = trackingErrorFac*trackingErrorFac*(gradsInterp[0] * gradsInterp[0] + gradsInterp[1] * gradsInterp[1]) / (geoDispError*geoDispError);

	// final error consists of a small constant part (discretization error),
	// geometric and photometric error.
	*resultVar = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist + geoDispError + photoDispError);	// square to make variance

	*resultIDepth = idnew_best_match;


	return 0.f;
}

float doOmniStereo(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	vec2 &bestEpDir, vec3 &bestMatchPos, float &gradAlongLine, float &tracedLineLen,
	cv::Mat &drawMatch)
{
	if (enablePrintDebugInfo) stats->num_stereo_calls++;

	//Find line containing possible positions for the match, in the frame of 
	// the reference image.
	//N.B. This line moves away from the epipole in the reference image.
	vec3 keyframePointDir = oModel.pixelToCam(vec2(u, v), 1.f);
//	vec3 expectedMatchPos = referenceFrame->otherToThis_R
//		* (keyframePointDir / prior_idepth) 
//		+ referenceFrame->otherToThis_t;
	vec3 lineStartPos = keyframeToReference * vec3(keyframePointDir / max_idepth);
	vec3 lineEndPos   = keyframeToReference * vec3(keyframePointDir / min_idepth);

	if (!epipolarLineInImageOmni(lineStartPos, lineEndPos, oModel))	{
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return -1;
	}

	//Find values to search for in keyframe.
	//These values are 5 samples, advancing along the epipolar line towards the
	//epipole, centred at the input point u,v.
	std::array<float, 5> valuesToFind = getValuesToFindOmni(keyframePointDir,
		epDir, keyframe, width, oModel, u, v);

	vec2 lineStartPix = oModel.camToPixel(lineStartPos);
	vec2 lineEndPix   = oModel.camToPixel(lineEndPos  );
	padEpipolarLineOmni(&lineStartPos, &lineEndPos, &lineStartPix, &lineEndPix,
		MIN_EPL_LENGTH_CROP, oModel);

	//Check padded line still in image.
	if (!epipolarLineInImageOmni(lineStartPos, lineEndPos, oModel))	{
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return -1;
	}

	//=======BEGIN LINE SEARCH CODE=======
	std::array<vec3, 5> lineDir;
	std::array<vec2, 5> linePix;
	std::array<float, 5> lineValue;
	float bestMatchErr = FLT_MAX, secondBestMatchErr = FLT_MAX;
	vec2 bestMatchPix, secondBestMatchPix;
	vec3 secondBestMatchPos;
	float centerA = 0.f;
	int loopCBest = -1, loopCSecondBest = -1;

	//Find first 4 values along line.
	lineDir[2] = lineStartPos;
	linePix[2] = lineStartPix;
	lineValue[2] = getInterpolatedElement(referenceFrameImage, lineStartPix, width);
	float a = 1.f; vec3 dir;
	a += oModel.getEpipolarParamIncrement(a, lineStartPos, lineEndPos, GRADIENT_SAMPLE_DIST);
	lineDir[1] = a*lineStartPos + (1.f - a)*lineEndPos;
	linePix[1] = oModel.camToPixel(lineDir[1]);
	if (!oModel.pixelLocValid(linePix[1])) {
		return -1;
	}
	lineValue[1] = getInterpolatedElement(referenceFrameImage, linePix[1], width);
	a += oModel.getEpipolarParamIncrement(a, lineStartPos, lineEndPos, GRADIENT_SAMPLE_DIST);
	lineDir[0] = a*lineStartPos + (1.f - a)*lineEndPos;
	linePix[0] = oModel.camToPixel(lineDir[0]);
	if (!oModel.pixelLocValid(linePix[0])) {
		return -1;
	}
	lineValue[0] = getInterpolatedElement(referenceFrameImage, linePix[0], width);
	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, lineEndPos, lineStartPos, GRADIENT_SAMPLE_DIST);
	lineDir[3] = a*lineEndPos + (1.f - a)*lineStartPos;
	linePix[3] = oModel.camToPixel(lineDir[3]);
	if (!oModel.pixelLocValid(linePix[3])) {
		return -1;
	}
	lineValue[3] = getInterpolatedElement(referenceFrameImage, linePix[3], width);

	tracedLineLen = 0.f;
	//Advance along line.
	size_t loopC = 0;
	while (centerA <= 1.f) {
		centerA = a;
		//Find fifth entry
		a += oModel.getEpipolarParamIncrement(a, lineEndPos, lineStartPos, GRADIENT_SAMPLE_DIST);
		lineDir[4] = a*lineEndPos + (1.f - a)*lineStartPos;
		linePix[4] = oModel.camToPixel(lineDir[4]);
		if (!oModel.pixelLocValid(linePix[4])) {
			//Epipolar curve has left image - terminate here.
			break;
		}

		lineValue[4] = getInterpolatedElement(referenceFrameImage, linePix[4], width);

		//Check error
		float err = findSsd(valuesToFind, lineValue);

		if (!drawMatch.empty()) {
			vec3 rgb = 255.f * hueToRgb(err / 325125.f);
			vec2 pix = oModel.camToPixel(lineDir[2]);
			drawMatch.at<cv::Vec3b>(int(pix.y()), int(pix.x())) =
				cv::Vec3b(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
		}

		if (err < bestMatchErr) {
			//Move best match to second place.
			secondBestMatchErr = bestMatchErr;
			secondBestMatchPix = bestMatchPix;
			secondBestMatchPos = bestMatchPos;
			loopCSecondBest = loopCBest;
			//Replace best match
			bestMatchErr = err;
			bestMatchPix = linePix[2];
			bestMatchPos = lineDir[2];
			loopCBest = loopC;
			bestEpDir = linePix[3] - linePix[2];
		} else if (err < secondBestMatchErr) {
			//Replace second best match.
			secondBestMatchErr = err;
			secondBestMatchPix = linePix[2];
			secondBestMatchPos = lineDir[2];
			loopCSecondBest = loopC;
		}

		//Shuffle values down
		for (size_t i = 0; i < 4; ++i) {
			lineDir[i] = lineDir[i + 1];
			linePix[i] = linePix[i + 1];
			lineValue[i] = lineValue[i + 1];
		}
		tracedLineLen += (linePix[2] - linePix[1]).norm();
		++loopC;
	}
	//Check if epipolar line left image before any error vals could be found.
	if (loopCBest < 0) {
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return -1;
	}

	// if error too big, will return -3, otherwise -2.
	if (bestMatchErr > 4.0f*(float)MAX_ERROR_STEREO)
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}

	// check if clear enough winner
	if (loopCBest >= 0 && loopCSecondBest >= 0) {
		if (abs(loopCBest - loopCSecondBest) > 1.0f && 
			MIN_DISTANCE_ERROR_STEREO * bestMatchErr > secondBestMatchErr) {
			if (enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
			return -2;
		}
	}

	//Perform subpixel matching, if necessary.
	bool didSubpixel = false;
	if (useSubpixelStereo) {
		//TODO subpixel matching
		didSubpixel = subpixelMatchOmni();
	}

	gradAlongLine = 0.f;
	float tmp = valuesToFind[4] - valuesToFind[3];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[3] - valuesToFind[2];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[2] - valuesToFind[1];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[1] - valuesToFind[0];  gradAlongLine += tmp*tmp;
	gradAlongLine /= GRADIENT_SAMPLE_DIST*GRADIENT_SAMPLE_DIST;

	// check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
	if (bestMatchErr > (float)MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20) {
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}

	bestEpDir.normalize();
	
	return bestMatchErr;
}

float DepthMap::doOmniStereo(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	OmniCameraModel &oModel = static_cast<OmniCameraModel&>(*model);
	
	vec2 bestEpDir;
	vec3 bestMatchPos;
	RigidTransform keyframeToReference;
	keyframeToReference.translation = referenceFrame->otherToThis_t;
	keyframeToReference.rotation = referenceFrame->otherToThis_R;
	float gradAlongLine, tracedLineLen;
	float bestMatchErr = lsd_slam::doOmniStereo(u, v, epDir, min_idepth, prior_idepth, max_idepth,
		activeKeyFrameImageData, referenceFrameImage, keyframeToReference,
		result_idepth, result_var, result_eplLength,
		stats, oModel, referenceFrame->width(), bestEpDir, bestMatchPos, gradAlongLine, tracedLineLen);

	bestEpDir.normalize();
	float r = findDepthAndVarOmni(u, v, bestMatchPos, &result_idepth, &result_var, 
		gradAlongLine, bestEpDir, referenceFrame, activeKeyFrame, 
		GRADIENT_SAMPLE_DIST, false, tracedLineLen, &oModel, stats);
	if (r != 0.f) {
		return r;
	}

	result_eplLength = tracedLineLen;
	
	return bestMatchErr;
}

bool epipolarLineInImageOmni(vec3 lineStart, vec3 lineEnd, const OmniCameraModel &model)
{
	if (!model.pointInImage(lineStart) || !model.pointInImage(lineEnd)) {
		return false;
	}
	return true;
}

void padEpipolarLineOmni(vec3 *lineStart, vec3 *lineEnd,
	vec2* lineStartPix, vec2* lineEndPix,
	float minLength,
	const OmniCameraModel &oModel)
{
	//N.B. this may underestimate the length of longer curves, but since
	// minLength is set to a small value (e.g. 3) this doesn't matter.
	float eplLength = (*lineEndPix - *lineStartPix).norm(); 

	if (eplLength < minLength)
	{
		// make epl long enough (pad a little bit).
		//TODO
//		float pad = (minLength - (eplLength)) / 2.0f;
//		float a = 1.f;


		*lineStartPix = oModel.camToPixel(*lineStart);
		*lineEndPix = oModel.camToPixel(*lineEnd);
	}
}

std::array<float, 5> getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel, 
	float u, float v)
{

	std::array<float, 5> valuesToFind;
	valuesToFind[2] = getInterpolatedElement(activeKeyFrameImageData, u, v, width);
	float a = 1.f; vec3 dir; vec2 pixel;
	a += oModel.getEpipolarParamIncrement(a, keyframePointDir, epDir, GRADIENT_SAMPLE_DIST);
	dir = a*keyframePointDir + (1.f - a)*epDir;
	valuesToFind[1] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);
	a += oModel.getEpipolarParamIncrement(a, keyframePointDir, epDir, GRADIENT_SAMPLE_DIST);
	dir = a*keyframePointDir + (1.f - a)*epDir;
	valuesToFind[0] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);

	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	valuesToFind[3] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	valuesToFind[4] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);

	return valuesToFind;
}

bool DepthMap::makeAndCheckEPLOmni(const int x, const int y, const Frame* const ref,
	vec3 *epDir, RunningStats* const stats)
{
	int idx = x+y*width;

	// ======= make epl ========
	*epDir = ref->thisToOther_t.normalized();
	vec2 epipole = model->camToPixel(ref->thisToOther_t);
	float epx = x - epipole.x();
	float epy = y - epipole.y();

	// ======== check epl length =========
	float eplLengthSquared = epx*epx+epy*epy;
	if(eplLengthSquared < MIN_EPL_LENGTH_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl++;
		return false;
	}


	// ===== check epl-grad magnitude ======
	float gx = activeKeyFrameImageData[idx+1    ] - activeKeyFrameImageData[idx-1    ];
	float gy = activeKeyFrameImageData[idx+width] - activeKeyFrameImageData[idx-width];
	float eplGradSquared = gx * epx + gy * epy;
	eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;	// square and norm with epl-length

	if(eplGradSquared < MIN_EPL_GRAD_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_grad++;
		return false;
	}


	// ===== check epl-grad angle ======
	if(eplGradSquared / (gx*gx+gy*gy) < MIN_EPL_ANGLE_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_angle++;
		return false;
	}

	return true;
}

/*
bool omniStereo(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	const float *reference,
	int width,
	int x, int y,
	float minDepth, float maxDepth,
	float &minSsd,
	vec3 &matchDir,
	vec2 &matchPixel,
	cv::Mat &drawMatch)
{
	vec3 pointDir;
	std::array<float, 5> searchVals = findValuesToSearchFor(keyframeToReference,
		model, keyframe, x, y, width, pointDir);

	vec3 minDPointDir = minDepth * pointDir;
	vec3 maxDPointDir = maxDepth * pointDir;

	//Find start and end of search curve
	vec3 lineStart = keyframeToReference * minDPointDir;
	vec3 lineEnd   = keyframeToReference * maxDPointDir;
	lineStart.normalize(), lineEnd.normalize();

	if (!drawMatch.empty()) {
		//vec2 pix = model.camToPixel(lineStart);
		//cv::circle(drawMatch, cv::Point(pix.x(), pix.y()), 3, cv::Scalar(0, 255, 0));
		//pix = model.camToPixel(lineEnd);
		//cv::circle(drawMatch, cv::Point(pix.x(), pix.y()), 3, cv::Scalar(0, 0, 255));
	}
	
	//Get first 5 possible matching values
	float a = 0.f;//Line parameter
	std::array<float, 5> matchVals;
	std::array<vec3, 5> matchDirs;
	matchVals[2] = getInterpolatedElement(reference, model.camToPixel(lineStart), width);
	matchDirs[2] = lineStart;
	
	//Get two points before line start.
	vec3 backDir = 2.f * lineStart - lineEnd;
	a += model.getEpipolarParamIncrement(a, backDir, lineStart);
	matchDirs[1] = a*backDir + (1.f - a)*lineStart;
	matchVals[1] = getInterpolatedElement(reference, model.camToPixel(matchDirs[1]), width);

	a += model.getEpipolarParamIncrement(a, backDir, lineStart);
	matchDirs[0] = a*backDir + (1.f - a)*lineStart;
	matchVals[0] = getInterpolatedElement(reference, model.camToPixel(matchDirs[0]), width);

	//Get two points after line start.
	a = 0.f;
	float prevA = 0.f, prevPrevA = 0.f;
	a += model.getEpipolarParamIncrement(a, lineEnd, lineStart); 
	matchDirs[3] = a*lineEnd + (1.f - a)*lineStart;
	matchVals[3] = getInterpolatedElement(reference, model.camToPixel(matchDirs[3]), width);
	prevA = a;

	a += model.getEpipolarParamIncrement(a, lineEnd, lineStart);
	matchDirs[4] = a*lineEnd + (1.f - a)*lineStart;
	matchVals[4] = getInterpolatedElement(reference, model.camToPixel(matchDirs[4]), width);
	float ssd = findSsd(searchVals, matchVals);

	matchDir = matchDirs[2];
	minSsd = ssd;

	//Advance along remainder of line.
	while (prevPrevA < 1.f) {
		prevPrevA = prevA;
		prevA = a;
		a += model.getEpipolarParamIncrement(a, lineEnd, lineStart);
		if (prevPrevA > 1.f) break;
		
		//Move values along arrays
		for (size_t i = 0; i < 4; ++i) {
			matchDirs[i] = matchDirs[i + 1];
			matchVals[i] = matchVals[i + 1];
		}

		//Find next value on line.
		matchDirs[4] = a*lineEnd + (1.f - a) * lineStart;
		matchVals[4] = getInterpolatedElement(reference, model.camToPixel(matchDirs[4]), width);

		float ssd = findSsd(searchVals, matchVals);
		if (ssd < minSsd) {
			minSsd = ssd;
			matchDir = matchDirs[2];
		}

		if (!drawMatch.empty()) {
			vec2 pix = model.camToPixel(matchDirs[2]);
			vec3 rgb = 255.f * hueToRgb(ssd / 50000.f);
			drawMatch.at<cv::Vec3b>(int(pix.y()), int(pix.x())) =
				cv::Vec3b(int(rgb.z()), int(rgb.y()), int(rgb.x()));
		}
	}

	matchPixel = model.camToPixel(matchDir);

	return true;
}
*/

std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width,
	vec3 &pointDir,
	cv::Mat &visIm)
{
	pointDir = model.pixelToCam(vec2(x, y));
	vec3 epipoleDir = -keyframeToReference.translation.normalized();

	float a = 0.f;
	//Advance two pixels from point toward epipole.
	a += model.getEpipolarParamIncrement(a, epipoleDir, pointDir);
	vec3 fwdDir1 = a*epipoleDir + (1.f - a)*pointDir;
	a += model.getEpipolarParamIncrement(a, epipoleDir, pointDir);
	vec3 fwdDir2 = a*epipoleDir + (1.f - a)*pointDir;

	//Advance two pixels from point away from epipole.
	a = 0.f;
	vec3 otherDir = 2.f*pointDir - epipoleDir;
	a += model.getEpipolarParamIncrement(a, otherDir, pointDir);
	vec3 bwdDir1 = a*otherDir + (1.f - a)*pointDir;
	a += model.getEpipolarParamIncrement(a, otherDir, pointDir);
	vec3 bwdDir2 = a*otherDir + (1.f - a)*pointDir;
	
#if SHOW_DEBUG_IMAGES
	cv::Mat lineImage(480, 640, CV_8UC1);
	lineImage.setTo(0);
	
	std::vector<vec3> dirs = {
		bwdDir2, bwdDir1, pointDir, fwdDir1, fwdDir2
	};
	
	for(vec3 &v : dirs) {
		vec2 img = model.camToPixel(v);
		static int i = 0;
		std::cout << "Point " << i++ << ": " << int(img.x()) << ", " << int(img.y()) << std::endl;
		lineImage.at<uchar>(int(img.y()), int(img.x())) = 255;
	}
	
	cv::imshow("LINE", lineImage);
	cv::waitKey();
#endif

	if (!visIm.empty()) {
		vec2 pix = model.camToPixel(bwdDir2);
		visIm.at<cv::Vec3b>(int(pix.y()), int(pix.x())) = cv::Vec3b(0, 255, 0);
		pix = model.camToPixel(bwdDir1);
		visIm.at<cv::Vec3b>(int(pix.y()), int(pix.x())) = cv::Vec3b(0, 255, 0);
		pix = model.camToPixel(pointDir);
		visIm.at<cv::Vec3b>(int(pix.y()), int(pix.x())) = cv::Vec3b(0, 255, 255);
		pix = model.camToPixel(fwdDir1);
		visIm.at<cv::Vec3b>(int(pix.y()), int(pix.x())) = cv::Vec3b(0, 255, 0);
		pix = model.camToPixel(fwdDir2);
		visIm.at<cv::Vec3b>(int(pix.y()), int(pix.x())) = cv::Vec3b(0, 255, 0);
	}
	
	//Find values of keyframe at these points.
	std::array<float, 5> vals = {
		{getInterpolatedElement(keyframe, model.camToPixel(bwdDir2), width),
		getInterpolatedElement(keyframe, model.camToPixel(bwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(pointDir), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir2), width)}
	};

	return vals;
}



}

