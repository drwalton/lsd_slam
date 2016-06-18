#include "DepthEstimation/DepthMapOmni.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "Util/settings.hpp"
#include "Util/globalFuncs.hpp"
#include "DataStructures/Frame.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>

///This file contains the depth estimation functions specific to Omnidirectional
/// camera models.

namespace lsd_slam {

const float MAX_EPL_DOT_PRODUCT = 0.9f; //Max dot product with epipole in stereo.
const float MAX_TRACED_ANGLE = float(M_PI / 16.);
const float MIN_TRACED_DOT_PROD = cos(MAX_TRACED_ANGLE);

///\brief Check if an epipolar line lies within an omnidirectional image.
///\note For now, just checks if the endpoints lie in the image.
bool epipolarLineInImageOmni(vec3 lineStart, vec3 lineEnd, const OmniCameraModel &model);

///\brief Increase the size of an epipolar line, if its length is found
///       to be less than minLength.
void padEpipolarLineOmni(vec3 *lineStart, vec3 *lineEnd,
	vec2* lineStartPix, vec2* lineEndPix,
	float minLength,
	const OmniCameraModel &oModel);

bool subpixelMatchOmni(
	vec2 *best_match, const vec2 &best_match_pre, const vec2 &best_match_post,
	float *best_match_err,
	float best_match_errPre, float best_match_DiffErrPre,
	float best_match_errPost, float best_match_DiffErrPost,
	RunningStats *stats) 
{
	bool didSubpixel = false;

	// ================== compute exact match =========================
	// compute gradients (they are actually only half the real gradient)
	float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
	float gradPre_this = +(*best_match_err - best_match_DiffErrPre);
	float gradPost_this = -(*best_match_err - best_match_DiffErrPost);
	float gradPost_post = +(best_match_errPost - best_match_DiffErrPost);

	// final decisions here.
	bool interpPost = false;
	bool interpPre = false;

	// if one is oob: return false.
	if (enablePrintDebugInfo && (best_match_errPre < 0 || best_match_errPost < 0))
	{
		stats->num_stereo_invalid_atEnd++;
	}


	// - if zero-crossing occurs exactly in between (gradient Inconsistent),
	else if ((gradPost_this < 0) ^ (gradPre_this < 0))
	{
		// return exact pos, if both central gradients are small compared to their counterpart.
		if (enablePrintDebugInfo && (gradPost_this*gradPost_this > 0.1f*0.1f*gradPost_post*gradPost_post ||
			gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre))
			stats->num_stereo_invalid_inexistantCrossing++;
	}

	// if pre has zero-crossing
	else if ((gradPre_pre < 0) ^ (gradPre_this < 0))
	{
		// if post has zero-crossing
		if ((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			if (enablePrintDebugInfo) stats->num_stereo_invalid_twoCrossing++;
		} else
			interpPre = true;
	}

	// if post has zero-crossing
	else if ((gradPost_post < 0) ^ (gradPost_this < 0))
	{
		interpPost = true;
	}

	// if none has zero-crossing
	else
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_noCrossing++;
	}


	// DO interpolation!
	// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
	// the error at that point is also computed by just integrating.
	if (interpPre)
	{
		float d = gradPre_this / (gradPre_this - gradPre_pre);
		*best_match -= d * (best_match_pre - *best_match);
		*best_match_err = *best_match_err - 2 * d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
		if (enablePrintDebugInfo) stats->num_stereo_interpPre++;
		didSubpixel = true;

	} else if (interpPost)
	{
		float d = gradPost_this / (gradPost_this - gradPost_post);
		*best_match -= d * (best_match_post - *best_match);
		*best_match_err = *best_match_err + 2 * d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
		if (enablePrintDebugInfo) stats->num_stereo_interpPost++;
		didSubpixel = true;
	} else
	{
		if (enablePrintDebugInfo) stats->num_stereo_interpNone++;
	}

	return didSubpixel;
}

std::string Ray::to_string() const
{
	std::stringstream ss;
	ss << origin << "\n" << dir;
	return ss.str();
}

std::string RayIntersectionResult::to_string() const
{
	std::stringstream ss;
	switch (valid) {
	case VALID:
		ss << "VALID\n"; break;
	case BEHIND:
		ss << "BEHIND\n"; break;
	case PARALLEL:
		ss << "PARALLEL\n"; break;
	}
	if (valid == VALID || valid == BEHIND) {
		ss << distance << "\n" << position << "\n";
	}
	return ss.str();
}
std::ostream &operator <<(std::ostream &s, const Ray &r)
{
	s << r.to_string();
	return s;
}
std::ostream &operator <<(std::ostream &s, const RayIntersectionResult &r)
{
	s << r.to_string();
	return s;
}
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

float findInvDepthOmni(const float u, const float v, const vec3 &bestMatchDir,
	OmniCameraModel *model, RigidTransform refToKeyframe, RunningStats *stats) 
{
	// ================= calc depth (in KF) ====================
	float idepth;	// depth in the new image
	float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
	vec3 findDirKf = model->pixelToCam(vec2(u, v));
	vec3 bestMatchDirKf = refToKeyframe.rotation * bestMatchDir;
	Ray findRay, matchRay;
	findRay.dir = findDirKf; findRay.origin = vec3::Zero();
	matchRay.dir = bestMatchDirKf; matchRay.origin = refToKeyframe.translation;
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
	idepth = 1.f / alpha;
	if (idepth != idepth) {
		return -2;
	}
	return idepth;
}

float findVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float gradAlongLine, const vec2 &bestEpDir, 
	const Eigen::Vector4f *activeKeyFrameGradients,
	float initialTrackedResidual,
	float sampleDist, bool didSubpixel,
	OmniCameraModel *model,
	RunningStats *stats,
	float depth)
{
	// ================= calc var (in NEW image) ====================

	// calculate error from photometric noise
	float photoDispError = 4.0f * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);
	float trackingErrorFac = 0.25f*(1.0f + initialTrackedResidual);

	// calculate error from geometric noise (wrong camera pose / calibration)
	Eigen::Vector2f gradsInterp = getInterpolatedElement42(activeKeyFrameGradients, u, v, model->w);
	float geoDispErrorDen = (gradsInterp[0] * bestEpDir[0] + 
		gradsInterp[1] * bestEpDir[1]) + DIVISION_EPS;
	if (fabsf(geoDispErrorDen) < DIVISION_EPS) {
		std::cout << "geoDispErrDenTooSmall" << std::endl;
		throw std::runtime_error("geoDispErrTooSmall");
	}
	if (geoDispErrorDen != geoDispErrorDen) {
		std::cout << "geoDispErrDen NAN" << std::endl;
		throw std::runtime_error("geoDispErrDen NAN");
	}
	float geoDispError = trackingErrorFac*trackingErrorFac*
		(gradsInterp[0] * gradsInterp[0] + gradsInterp[1] * gradsInterp[1]) / 
		(geoDispErrorDen*geoDispErrorDen);
	if (geoDispError != geoDispError) {
		std::cout << "geoDispErr NAN" << std::endl;
		throw std::runtime_error("geoDispErr NAN");
	}

	// final error consists of a small constant part (discretization error),
	// geometric and photometric error.
	float var = depth*depth*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist
		+ geoDispError + photoDispError);	// square to make variance

	if (var != var) {
		std::cout << "Var != Var" << std::endl;
		throw std::runtime_error("Var != Var!");
	}
	return var;
}

float findDepthAndVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float *resultIDepth, float *resultVar,
	float gradAlongLine, const vec2 &bestEpDir, const Frame *referenceFrame, Frame *activeKeyFrame,
	float sampleDist, bool didSubpixel, float tracedLineLen,
	OmniCameraModel *model,
	RunningStats *stats)
{
	RigidTransform referenceToKeyframe;
	referenceToKeyframe.rotation = referenceFrame->thisToOther_R;
	referenceToKeyframe.translation = referenceFrame->thisToOther_t;
	*resultIDepth = findInvDepthOmni(u, v, bestMatchDir, model,
		referenceToKeyframe, stats);

	if (*resultIDepth < 0.f) {
		//Error code returned, return this code.
		return *resultIDepth;
	}

	*resultVar = findVarOmni(u, v, bestMatchDir, gradAlongLine,
		bestEpDir, activeKeyFrame->gradients(0), referenceFrame->initialTrackedResidual,
		sampleDist, didSubpixel, model, stats, 1.f / *resultIDepth);

	return 0.f;
}

float doStereoOmniImpl(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	vec2 &bestEpImDir, vec3 &bestMatchPos, float &gradAlongLine, float &tracedLineLen,
	vec3 &bestMatchKeyframe,
	cv::Mat &drawMatch, bool drawThisMatch)
{
	if (!(max_idepth > min_idepth)) {
		std::cout << "wrong inv depths in doOmniStereo" << std::endl;
		throw std::runtime_error("wrong inv depths in doOmniStereo");
	}
	if (!(min_idepth >= 0.f)) {
		std::cout << "negative depth" << std::endl;
		throw std::runtime_error("negative inv depth in doOmniStereo");
	}

	if (enablePrintDebugInfo) stats->num_stereo_calls++;

	LineSeg3d keyframeLine, refFrameLine;
	vec3 keyframeMatchDir; //Dir towards match from keyframe's FoR.
	OmniEpLine2d refFrameLinePix;
	MakePaddedLineErrorCode padErrCode;
	padErrCode = makePaddedEpipolarLineOmni(u, v,
		prior_idepth, min_idepth, max_idepth,
		MIN_EPL_LENGTH_CROP + 5.f,
		oModel, keyframeToReference,
		&keyframeMatchDir,
		&keyframeLine, &refFrameLine,
		&refFrameLinePix);
	if(padErrCode != MakePaddedLineErrorCode::SUCCESS) {
		return -4;
	}
	
	if (drawThisMatch) {
		if (drawMatch.cols == width) {
			cv::circle(drawMatch,
			cv::Point(int(refFrameLinePix.start.x()), int(refFrameLinePix.start.y())), 3, cv::Scalar(0,255,255));
			cv::circle(drawMatch,
			cv::Point(int(refFrameLinePix.end.x()), int(refFrameLinePix.end.y())), 3, cv::Scalar(0,255,255));
		}
	}
	
	//Find values to search for in keyframe.
	//These values are 5 samples, advancing along the epipolar line towards the
	//epipole, centred at the input point u,v.
	std::array<float, 5> valuesToFind;
	bool valuesToFindFound = false;
	if (drawThisMatch) {
		valuesToFindFound = getValuesToFindOmni(keyframeMatchDir,
			epDir, keyframe, width, oModel, u, v, valuesToFind, drawMatch);
	} else {
		valuesToFindFound = getValuesToFindOmni(keyframeMatchDir,
			epDir, keyframe, width, oModel, u, v, valuesToFind);
	}
	if(!valuesToFindFound) {
		//5 values centered around point not available.
		return -1;
	}
	
	//=======BEGIN LINE SEARCH CODE=======
	std::array<vec3, 5> lineDir;
	std::array<vec2, 5> linePix;
	std::array<float, 5> lineValue;
	std::array<float, 5> e0, e1;
	bool bestWasLastLoop = false;
	float bestMatchErr = FLT_MAX, secondBestMatchErr = FLT_MAX;
	float bestMatchErrPre, bestMatchErrPost, bestMatchDiffErrPre, bestMatchDiffErrPost;
	vec2 bestMatchPix, secondBestMatchPix; //Pixel locations of best, 2nd best matches.
	float bestMatchA = 0.f;
	vec2 bestMatchPre, bestMatchPost; //Pixel locs of pixels just before, after best match.
	vec3 secondBestMatchPos;
	float centerA = 0.f;
	int loopCBest = -1, loopCSecondBest = -1;
	
	std::vector<float> searchedVals;
	std::vector<cv::Vec3b> ssdColors;

	//Find first 4 values along line.
	lineDir[2] = refFrameLine.start;
	linePix[2] = refFrameLinePix.start;
	lineValue[2] = getInterpolatedElement(referenceFrameImage,
		refFrameLinePix.start, width);
	float a = 1.f; vec3 dir;
	a += oModel.getEpipolarParamIncrement(a, refFrameLine.start, refFrameLine.end, GRADIENT_SAMPLE_DIST);
	lineDir[1] = a*refFrameLine.start + (1.f - a)*refFrameLine.end;
	linePix[1] = oModel.camToPixel(lineDir[1]);
	if (!oModel.pixelLocValid(linePix[1])) {
		return -1;
	}
	lineValue[1] = getInterpolatedElement(referenceFrameImage, linePix[1], width);
	a += oModel.getEpipolarParamIncrement(a, refFrameLine.start, refFrameLine.end, GRADIENT_SAMPLE_DIST);
	lineDir[0] = a*refFrameLine.start + (1.f - a)*refFrameLine.end;
	linePix[0] = oModel.camToPixel(lineDir[0]);
	if (!oModel.pixelLocValid(linePix[0])) {
		return -1;
	}
	lineValue[0] = getInterpolatedElement(referenceFrameImage, linePix[0], width);
	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, refFrameLine.end, refFrameLine.start, GRADIENT_SAMPLE_DIST);
	lineDir[3] = a*refFrameLine.end + (1.f - a)*refFrameLine.start;
	linePix[3] = oModel.camToPixel(lineDir[3]);
	if (!oModel.pixelLocValid(linePix[3])) {
		return -1;
	}
	lineValue[3] = getInterpolatedElement(referenceFrameImage, linePix[3], width);
	
	if (drawThisMatch && drawMatch.cols == width * 2) {
		cv::circle(drawMatch, cv::Point(int(u),int(v)), 2, CV_RGB(255, 0, 0));
	}

	tracedLineLen = 0.f;
	//Advance along line.
	size_t loopC = 0;
	float errLast = -1.f;
	while (centerA <= 1.f) {
		if (loopC == 100) {
			std::cout << "***LONG LINE***" << std::endl;
			std::cout << "Tracing from \n" <<
				refFrameLinePix.start << "\nto\n" <<
				refFrameLinePix.end << "\nRef frame pos:\n" <<
				refFrameLine.start << "\nto\n" <<
				refFrameLine.end << "\n" << std::endl;
		}
		centerA = a;
		//Find fifth entry
		a += oModel.getEpipolarParamIncrement(a, refFrameLine.end, refFrameLine.start, GRADIENT_SAMPLE_DIST);
		lineDir[4] = a*refFrameLine.end + (1.f - a)*refFrameLine.start;
		linePix[4] = oModel.camToPixel(lineDir[4]);
		if (!oModel.pixelLocValid(linePix[4])) {
			//Epipolar curve has left image - terminate here.
			break;
		}

		lineValue[4] = getInterpolatedElement(referenceFrameImage, linePix[4], width);

		//Check error
		float err = 0.f;
		if (loopC % 2 == 0) {
			for (size_t i = 0; i < 5; ++i) {
				e0[i] = lineValue[i] - valuesToFind[i];
				err += e0[i] * e0[i];
			}
		} else {
			for (size_t i = 0; i < 5; ++i) {
				e1[i] = lineValue[i] - valuesToFind[i];
				err += e1[i] * e1[i];
			}
		}

		if (drawThisMatch) {
			vec3 rgb = 255.f * hueToRgb(0.8f * err / 325125.f);
			cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
			ssdColors.push_back(rgbB);
			if (drawMatch.cols == width * 2) {
				drawMatch.at<cv::Vec3b>(int(linePix[2].y()), int(linePix[2].x() + width)) =
					rgbB;
			} else if(drawMatch.cols == width) {
				drawMatch.at<cv::Vec3b>(int(linePix[2].y()), int(linePix[2].x())) =
					rgbB;
			}
		}

		if (err < bestMatchErr) {
			//Move best match to second place.
			secondBestMatchErr = bestMatchErr;
			secondBestMatchPix = bestMatchPix;
			secondBestMatchPos = bestMatchPos;
			loopCSecondBest = loopCBest;
			//Replace best match
			bestMatchErr = err;
			bestMatchErrPre = errLast;
			bestMatchDiffErrPre =
				e0[0] * e1[0] +
				e0[1] * e1[1] +
				e0[2] * e1[2] +
				e0[3] * e1[3] +
				e0[4] * e1[4];
			bestWasLastLoop = true;
			bestMatchDiffErrPost = -1;
			bestMatchErrPost = -1;
			bestMatchPix = linePix[2];
			bestMatchPos = lineDir[2];
			bestMatchPre = linePix[1];
			bestMatchPost = linePix[3];
			loopCBest = loopC;
			bestEpImDir = linePix[3] - linePix[2];
			if (bestEpImDir == vec2::Zero()) {
				bestEpImDir = linePix[2] - linePix[1];
				if (bestEpImDir == vec2::Zero()) {
					return -4;
				}
			}
			bestMatchA = centerA;
		} else {
			if (bestWasLastLoop) {
				bestMatchErrPost = err;
				bestMatchDiffErrPre =
					e0[0] * e1[0] +
					e0[1] * e1[1] +
					e0[2] * e1[2] +
					e0[3] * e1[3] +
					e0[4] * e1[4];
				bestWasLastLoop = false;
			}

			if (err < secondBestMatchErr) {
				//Replace second best match.
				secondBestMatchErr = err;
				secondBestMatchPix = linePix[2];
				secondBestMatchPos = lineDir[2];
				loopCSecondBest = loopC;
			}
		}

		//Shuffle values down
		for (size_t i = 0; i < 4; ++i) {
			lineDir[i] = lineDir[i + 1];
			linePix[i] = linePix[i + 1];
			lineValue[i] = lineValue[i + 1];
		}
		tracedLineLen += (linePix[2] - linePix[1]).norm();
		++loopC;
		errLast = err;
	}

	if (drawThisMatch) {
		vec3 rgb(0, 255.f, 255.f);
		cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
		//ssdColors.push_back(rgbB);
		if (drawMatch.cols == width * 2) {
			drawMatch.at<cv::Vec3b>(int(bestMatchPix.y()), int(bestMatchPix.x() + width)) =
				rgbB;
		}	
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
	vec2 origpos = bestMatchPix;
	if (useSubpixelStereo) {
		didSubpixel = subpixelMatchOmni(
			&bestMatchPix, bestMatchPre, bestMatchPost,
			&bestMatchErr, 
			bestMatchErrPre, bestMatchDiffErrPre,
			bestMatchErrPost, bestMatchDiffErrPost,
			stats);
		if (didSubpixel) {
			//std::cout << "SUBPIXEL: " << origpos << "\n" << bestMatchPix << std::endl;
			bestMatchPos = oModel.pixelToCam(bestMatchPix);
		}
	}

	gradAlongLine = 0.f;
	float tmp = valuesToFind[4] - valuesToFind[3];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[3] - valuesToFind[2];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[2] - valuesToFind[1];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[1] - valuesToFind[0];  gradAlongLine += tmp*tmp;
	gradAlongLine /= GRADIENT_SAMPLE_DIST*GRADIENT_SAMPLE_DIST;

	// check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
	if (bestMatchErr > (float)MAX_ERROR_STEREO /* + sqrtf(gradAlongLine)*  20*/) {
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}


	bestEpImDir.normalize();
	if (bestEpImDir != bestEpImDir) {
		std::cout << "bestEpImDir != bestEpImDir" << std::endl;
		throw std::runtime_error("bestEpImDir != bestEpImDir");
	}
	bestMatchKeyframe = bestMatchA*keyframeLine.end +
		(1.f - bestMatchA)*keyframeLine.start;
	
	return bestMatchErr;
}

float DepthMap::doStereoOmni(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	OmniCameraModel &oModel = static_cast<OmniCameraModel&>(*camModel_);
	
	vec2 bestEpImDir;
	vec3 bestMatchPos;
	vec3 bestMatchKeyframe;
	RigidTransform keyframeToReference;
	keyframeToReference.translation = referenceFrame->otherToThis_t;
	keyframeToReference.rotation = referenceFrame->otherToThis_R;
	float gradAlongLine, tracedLineLen;
	float bestMatchErr;
	float idepth;
	if (settings.saveSearchRangeImages || settings.saveResultImages) {
		bestMatchErr = lsd_slam::doStereoOmniImpl2(u, v, epDir, min_idepth, prior_idepth, max_idepth,
			activeKeyFrameImageData, referenceFrameImage, keyframeToReference,
			stats, oModel, referenceFrame->width(), idepth, bestEpImDir, bestMatchPos, gradAlongLine, 
			tracedLineLen, bestMatchKeyframe, debugImages.searchRanges, 
			settings.saveSearchRangeImages && debugImages.drawMatchHere(u,v));
	}
	else {

		bestMatchErr = lsd_slam::doStereoOmniImpl2(u, v, epDir, min_idepth, prior_idepth, max_idepth,
			activeKeyFrameImageData, referenceFrameImage, keyframeToReference,
			stats, oModel, referenceFrame->width(), idepth, bestEpImDir, bestMatchPos, gradAlongLine, tracedLineLen, bestMatchKeyframe);
	}

	if (bestMatchErr > 0.f) {
		result_idepth = idepth;
		float var = findVarOmni(u, v, bestMatchPos.normalized(), gradAlongLine,
				bestEpImDir, activeKeyFrame->gradients(0), referenceFrame->initialTrackedResidual,
				GRADIENT_SAMPLE_DIST, false, &oModel, stats, 1.f / idepth);
		result_var = var;
		//float r = findDepthAndVarOmni(u, v, bestMatchPos, &result_idepth, &result_var,
		//	gradAlongLine, bestEpImDir, referenceFrame, activeKeyFrame,
		//	GRADIENT_SAMPLE_DIST, false, tracedLineLen, &oModel, stats);
		//if (r >= 0.f) {
		//	if (result_idepth != result_idepth) {
		//		throw std::runtime_error("idepth is nan");
		//	}
			if (settings.saveMatchImages) {
				debugImages.visualiseMatch(
					vec2(u, v), camModel_->camToPixel(bestMatchPos), camModel_.get());
			}
		//	return r;
		//}
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
//#define DEBUG_PRINT_LINE_PADDING_STUFF
	//N.B. this may underestimate the length of longer curves, but since
	// minLength is set to a small value (e.g. 3) this doesn't matter.
	float eplLength = (*lineEndPix - *lineStartPix).norm(); 

	if (eplLength < minLength)
	{
#ifdef DEBUG_PRINT_LINE_PADDING_STUFF
		std::cout << "Padding line from " << *lineStartPix << " to "
			<< *lineEndPix << " as length of " << eplLength << " is less"
			" than required length " << minLength << std::endl;
#endif
		vec3 lineEndN = lineEnd->normalized();
		vec3 lineStartN = lineStart->normalized();
		float requiredStep = 0.5f*(minLength - eplLength);
		int step = int(ceilf(requiredStep));
		float aFwd = 1.f;
		
		for (int i = 0; i < step; ++i) {
			aFwd += oModel.getEpipolarParamIncrement(aFwd, lineEndN, lineStartN, 1);
		}
		vec3 padLineEnd = aFwd*(*lineEnd) + (1.f - aFwd)*(*lineStart);
		vec2 padLineEndPix = oModel.camToPixel(padLineEnd);

		float aBwd = 1.f;
		for (int i = 0; i < step; ++i) {
			aBwd += oModel.getEpipolarParamIncrement(aBwd, lineStartN, lineEndN, 1);
		}
		vec3 padLineStart = aBwd*(*lineStart) + (1.f - aBwd)*(*lineEnd);
		vec2 padLineStartPix = oModel.camToPixel(padLineStart);

		eplLength = (padLineEnd - padLineStart).norm(); 
		if (eplLength > 100.f) {
			std::cout << "LONG LINE" << std::endl;
		}
		*lineStart = padLineStart;
		*lineEnd = padLineEnd;
		*lineStartPix = padLineStartPix;
		*lineEndPix = padLineEndPix;

#ifdef DEBUG_PRINT_LINE_PADDING_STUFF
		std::cout << "After padding, line runs from " << *lineStartPix 
			<< " to " << *lineEndPix << " and has length of "
			<< (*lineStartPix - *lineEndPix).norm() << std::endl;
#endif
	}
}

bool getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel, 
	float u, float v, std::array<float, 5> &valuesToFind, cv::Mat &visIm)
{
	valuesToFind[2] = getInterpolatedElement(activeKeyFrameImageData, u, v, width);
	float a = 0.f; vec3 dir; vec2 pixel;
	vec3 otherDir = 2.f*keyframePointDir - epDir;
	
	a += oModel.getEpipolarParamIncrement(a, otherDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*otherDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir)) return false;
	pixel = oModel.camToPixel(dir);
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[1] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	a += oModel.getEpipolarParamIncrement(a, otherDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*otherDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir)) return false;
	pixel = oModel.camToPixel(dir);
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[0] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);

	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir)) return false;
	pixel = oModel.camToPixel(dir);
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[3] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir)) return false;
	pixel = oModel.camToPixel(dir);
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[4] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	return true;
}

bool DepthMap::makeAndCheckEPLOmni(const int x, const int y, const Frame* const ref,
	vec3 *epDir, RunningStats* const stats)
{
	int idx = x+y*camModel_->w;

	// ======= make epl ========
	if (epDir == nullptr) {
		throw 1;
	}
	*epDir = ref->thisToOther_t.normalized();
	vec2 epipole = camModel_->camToPixel(ref->thisToOther_t);
	float epx = x - epipole.x();
	float epy = y - epipole.y();

	// ======== check epl length =========
	float eplLengthSquared = epx*epx + epy*epy;
	if (eplLengthSquared < MIN_EPL_LENGTH_SQUARED)
	{
		//Too close to epipole - fail.
		if (enablePrintDebugInfo) stats->num_observe_skipped_small_epl++;
		return false;
	}


	// ===== check epl-grad magnitude ======
	float gx = activeKeyFrameImageData[idx+1    ] - activeKeyFrameImageData[idx-1    ];
	float gy = activeKeyFrameImageData[idx+camModel_->w] - activeKeyFrameImageData[idx-camModel_->w];
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

MakePaddedLineErrorCode makePaddedEpipolarLineOmni(float u, float v,
	float meanIDepth, float minIDepth, float maxIDepth,
	float requiredLineLen,
	const OmniCameraModel &model,
	const RigidTransform &keyframeToReference,
	vec3 *keyframeDir,
	LineSeg3d *keyframeLine,
	LineSeg3d *refframeLine,
	OmniEpLine2d *refframeLinePix)
{
	//Create initial lines.
	vec2 p(u, v);
	*keyframeDir = model.pixelToCam(p);
	
	keyframeLine->start = *keyframeDir / maxIDepth;
	keyframeLine->end = *keyframeDir / minIDepth;
	refframeLine->start = keyframeToReference * keyframeLine->start;
	refframeLine->end = keyframeToReference * keyframeLine->end;
	
	
	vec3 epDirRefFrame = (keyframeToReference * vec3(0.f, 0.f, 0.f)).normalized();
	if (fabsf(epDirRefFrame.dot(
		(refframeLine->end - refframeLine->start).normalized())) 
		> MAX_EPL_DOT_PRODUCT) {
		//std::cout << "Fail: Too close to epipole!";
		return MakePaddedLineErrorCode::FAIL_NEAR_EPIPOLE;
	}

	refframeLinePix->start = model.camToPixel(refframeLine->start);
	refframeLinePix->end = model.camToPixel(refframeLine->end);
	if(!model.pixelLocValid(refframeLinePix->start) ||
		!model.pixelLocValid(refframeLinePix->end)) {
		return MakePaddedLineErrorCode::FAIL_OUT_OF_IMAGE;
	}
	if(int(refframeLinePix->start.x()) == int(refframeLinePix->end.x()) &&
	   int(refframeLinePix->start.y()) == int(refframeLinePix->end.y())) {
	   //The range of the line in the reference frame is too short.
		return MakePaddedLineErrorCode::FAIL_TOO_STEEP;
	}

	float initLength = (refframeLinePix->end - refframeLinePix->start).norm();

	if(initLength < requiredLineLen) { 
		//Line needs padding - length insufficient.

		float padAmt = (requiredLineLen - initLength) / 2.f;
		float lineEndA = 1.f, lineStartA = 0.f;
		lineEndA += model.getEpipolarParamIncrement(
			lineEndA, refframeLine->end, refframeLine->start, padAmt);
		lineStartA += model.getEpipolarParamIncrement(
			lineStartA, refframeLine->end, refframeLine->start, -padAmt);

		LineSeg3d padRefframeLine;
		LineSeg3d padKeyframeLine;

		padRefframeLine.start = lineStartA*refframeLine->end + 
			(1.f - lineStartA)*refframeLine->start;
		padRefframeLine.end = lineEndA*refframeLine->end + 
			(1.f - lineEndA)*refframeLine->start;

		if(!model.pointInImage(padRefframeLine.start) ||
			!model.pointInImage(padRefframeLine.end)) {
			return MakePaddedLineErrorCode::FAIL_OUT_OF_IMAGE;
		}
		refframeLinePix->start = model.camToPixel(padRefframeLine.start);
		refframeLinePix->end = model.camToPixel(padRefframeLine.end);
		if(!model.pixelLocValid(refframeLinePix->start) ||
			!model.pixelLocValid(refframeLinePix->end)) {
			return MakePaddedLineErrorCode::FAIL_OUT_OF_IMAGE;
		}
		
		float newLen = (refframeLinePix->end - refframeLinePix->start).norm();
		if (!(newLen < (1.1f * requiredLineLen))) {
			//std::cout << "Fail: Too long!";
			return MakePaddedLineErrorCode::FAIL_TOO_LONG;
		} else if (newLen < requiredLineLen - 1.f) {
			//Couldn't step far enough along line.
			return MakePaddedLineErrorCode::FAIL_TOO_SHORT;
		}

		*refframeLine = padRefframeLine;
		keyframeLine->start = lineStartA*keyframeLine->end + 
			(1.f - lineStartA)*keyframeLine->start;
		keyframeLine->end = lineEndA*keyframeLine->end + 
			(1.f - lineEndA)*keyframeLine->start;
	}
	
	return MakePaddedLineErrorCode::SUCCESS;
}

float doStereoOmniImpl2(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	float &iDepth, vec2 &bestEpImDir, vec3 &bestMatchPos, float &gradAlongLine, float &tracedLineLen,
	vec3 &bestMatchKeyframe,
	cv::Mat &drawMatch, bool drawThisMatch)
{
	if (!(max_idepth > min_idepth)) {
		std::cout << "wrong inv depths in doOmniStereo" << std::endl;
		throw std::runtime_error("wrong inv depths in doOmniStereo");
	}
	if (!(min_idepth >= 0.f)) {
		std::cout << "negative depth" << std::endl;
		throw std::runtime_error("negative inv depth in doOmniStereo");
	}

	if (enablePrintDebugInfo) stats->num_stereo_calls++;

	//Check if point is too near an epipole, terminate early if so.
	vec3 lineDirKf = oModel.pixelToCam(vec2(u, v));
	vec3 epDirKf = keyframeToReference.translation.normalized();
	if (fabsf(lineDirKf.dot(epDirKf)) > MAX_EPL_DOT_PRODUCT) {
		//KF position too close to an epipole! 
		if (enablePrintDebugInfo) ++stats->num_observe_skipped_small_epl;
		return DepthMapErrCode::START_TOO_NEAR_EPIPOLE;
	}

	//Find line endpoints in keyframe, reference frame.
	vec3 lineCloseRf = keyframeToReference * vec3(lineDirKf / max_idepth);
	float lineCloseIdepth = max_idepth;
	vec3 lineInfRf = keyframeToReference.rotation * lineDirKf;
	vec3 lineFarRf;
	float lineStartAlpha, lineEndAlpha = 1.f;
	vec3 lineCloseDirRf = lineCloseRf.normalized();
	vec3 lineFarDirRf;
	if (!std::isfinite(1.f / min_idepth)) {
		lineFarRf = lineFarDirRf = lineInfRf;
		lineStartAlpha = 0.f;
	} else {
		lineFarRf = keyframeToReference * vec3(lineDirKf / min_idepth);
		lineFarDirRf = lineFarRf.normalized();
		lineStartAlpha = (lineFarDirRf - lineInfRf).norm() / (lineCloseDirRf - lineInfRf).norm();
	}

	//Check if line is too long, and avoid processing if this is the case.
	if (lineCloseDirRf.dot(lineFarDirRf) < MIN_TRACED_DOT_PROD) {
		//Line is too long - fail.
		return DepthMapErrCode::TRACED_LINE_TOO_LONG;
	}

	//Check current length of line
	vec2 lineClosePixRf = oModel.camToPixel(lineCloseDirRf);
	vec2 lineFarPixRf = oModel.camToPixel(lineFarDirRf);
	if (!oModel.pixelLocValid(lineClosePixRf) || !oModel.pixelLocValid(lineFarPixRf)) {
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}
	float lineLen = (lineFarPixRf - lineClosePixRf).norm();

	//Extend line, if it isn't long enough
	if (lineLen < MIN_EPL_LENGTH_CROP) {
		vec3 padFarDirRf, padCloseDirRf;
		vec2 padFarPixRf, padClosePixRf;
		float amtToPad = MIN_EPL_LENGTH_CROP - lineLen;
		//First try padding backwards, towards the infinity point.
		if (lineStartAlpha > 0.f) {
			//Can pad backwards - not yet at infinity point.
			float padBackwardsAmt = amtToPad * .5f;
			float padLineStartAlpha = lineStartAlpha;
			padLineStartAlpha += oModel.getEpipolarParamIncrement(
				lineStartAlpha, lineCloseDirRf, lineInfRf, -padBackwardsAmt);
			if (padLineStartAlpha > 0.f) {
				padFarDirRf = padLineStartAlpha*lineCloseDirRf +
					(1.f - padLineStartAlpha)*lineInfRf;
				padFarPixRf = oModel.camToPixel(padFarDirRf);
				if (oModel.pixelLocValid(padFarPixRf)) {
					//Padding was successful
					lineStartAlpha = padLineStartAlpha;
					amtToPad = MIN_EPL_LENGTH_CROP - (padFarPixRf - lineClosePixRf).norm();
				}
				else {
					//Padded far point not in image: fail.
					return DepthMapErrCode::PADDED_EPL_NOT_IN_REF_FRAME;
				}
			}
		}

		//Now try padding forwards.
		float padLineEndAlpha = lineEndAlpha;
		padLineEndAlpha += oModel.getEpipolarParamIncrement(
			lineEndAlpha, lineCloseDirRf, lineInfRf, amtToPad);
		 padCloseDirRf = padLineEndAlpha*lineCloseDirRf +
			(1.f - padLineEndAlpha)*lineInfRf;
		padClosePixRf = oModel.camToPixel(padCloseDirRf);
		if (oModel.pixelLocValid(padClosePixRf)) {
			lineEndAlpha = padLineEndAlpha;
		} else {
			//Couldn't fit a long enough line in the image.
			return DepthMapErrCode::PADDED_EPL_NOT_IN_REF_FRAME;
		}
	}

	//Find values to search for in keyframe.
	//These values are 5 samples, advancing along the epipolar line towards the
	//epipole, centred at the input point u,v.
	std::array<float, 5> valuesToFind;
	bool valuesToFindFound = false;
	if (drawThisMatch) {
		valuesToFindFound = getValuesToFindOmni(lineDirKf,
			epDirKf, keyframe, width, oModel, u, v, valuesToFind, drawMatch);
	} else {
		valuesToFindFound = getValuesToFindOmni(lineDirKf,
			epDirKf, keyframe, width, oModel, u, v, valuesToFind);
	}
	if(!valuesToFindFound) {
		//5 values centered around point not available.
		return -1;
	}

	//=======BEGIN LINE SEARCH CODE=======
	std::array<vec3, 5> lineDir;
	std::array<vec2, 5> linePix;
	std::array<float, 5> lineValue;
	std::array<float, 5> e0, e1;
	bool bestWasLastLoop = false;
	float bestMatchErr = FLT_MAX, secondBestMatchErr = FLT_MAX;
	float bestMatchErrPre, bestMatchErrPost, bestMatchDiffErrPre, bestMatchDiffErrPost;
	vec2 bestMatchPix, secondBestMatchPix; //Pixel locations of best, 2nd best matches.
	float bestMatchA = 0.f;
	vec2 bestMatchPre, bestMatchPost; //Pixel locs of pixels just before, after best match.
	vec3 secondBestMatchPos;
	float centerA = 0.f;
	int loopCBest = -1, loopCSecondBest = -1;

	std::vector<float> searchedVals;
	std::vector<cv::Vec3b> ssdColors;

	float a = lineStartAlpha; vec3 dir;
	//Find first 4 values along line.
	lineDir[2] = a*lineCloseDirRf + (1.f - a)*lineInfRf;
	linePix[2] = oModel.camToPixel(lineDir[2]);
	if (!oModel.pixelLocValid(linePix[2])) {
		return -1;
	}
	lineValue[2] = getInterpolatedElement(referenceFrameImage,
		linePix[2], width);

	a += oModel.getEpipolarParamIncrement(a, lineInfRf, lineCloseDirRf, -GRADIENT_SAMPLE_DIST);
	lineDir[1] = a*lineCloseDirRf + (1.f - a)*lineInfRf;
	linePix[1] = oModel.camToPixel(lineDir[1]);
	if (!oModel.pixelLocValid(linePix[1])) {
		return -1;
	}
	lineValue[1] = getInterpolatedElement(referenceFrameImage, linePix[1], width);

	a += oModel.getEpipolarParamIncrement(a, lineInfRf, lineCloseDirRf, -GRADIENT_SAMPLE_DIST);
	lineDir[0] = a*lineCloseDirRf + (1.f - a)*lineInfRf;
	linePix[0] = oModel.camToPixel(lineDir[0]);
	if (!oModel.pixelLocValid(linePix[0])) {
		return -1;
	}
	lineValue[0] = getInterpolatedElement(referenceFrameImage, linePix[0], width);

	a = lineStartAlpha;
	a += oModel.getEpipolarParamIncrement(a, lineCloseDirRf, lineInfRf, GRADIENT_SAMPLE_DIST);
	lineDir[3] = a*lineCloseDirRf + (1.f - a)*lineInfRf;
	linePix[3] = oModel.camToPixel(lineDir[3]);
	if (!oModel.pixelLocValid(linePix[3])) {
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}
	lineValue[3] = getInterpolatedElement(referenceFrameImage, linePix[3], width);

	if (drawThisMatch && drawMatch.cols == width * 2) {
		//cv::circle(drawMatch, cv::Point(int(u), int(v)), 2, CV_RGB(255, 0, 0));
	}

	tracedLineLen = 0.f;
	//Advance along line.
	size_t loopC = 0;
	float errLast = -1.f;
	while (centerA <= lineEndAlpha) {
		if (loopC == 100) {
			return DepthMapErrCode::TRACED_LINE_TOO_LONG;
		}
		centerA = a;
		//Find fifth entry
		a += oModel.getEpipolarParamIncrement(a, lineCloseDirRf, lineInfRf, GRADIENT_SAMPLE_DIST);
		lineDir[4] = a*lineCloseDirRf + (1.f - a)*lineInfRf;
		linePix[4] = oModel.camToPixel(lineDir[4]);
		if (!oModel.pixelLocValid(linePix[4])) {
			//Epipolar curve has left image - terminate here.
			break;
		}

		lineValue[4] = getInterpolatedElement(referenceFrameImage, linePix[4], width);

		//Check error
		float err = 0.f;
		if (loopC % 2 == 0) {
			for (size_t i = 0; i < 5; ++i) {
				e0[i] = lineValue[i] - valuesToFind[i];
				err += e0[i] * e0[i];
			}
		}
		else {
			for (size_t i = 0; i < 5; ++i) {
				e1[i] = lineValue[i] - valuesToFind[i];
				err += e1[i] * e1[i];
			}
		}

		if (drawThisMatch) {
			vec3 rgb = 255.f * hueToRgb(0.8f * err / 325125.f);
			cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
			ssdColors.push_back(rgbB);
			if (drawMatch.cols == width * 2) {
				drawMatch.at<cv::Vec3b>(int(linePix[2].y()), int(linePix[2].x() + width)) =
					rgbB;
			}
			else if (drawMatch.cols == width) {
				drawMatch.at<cv::Vec3b>(int(linePix[2].y()), int(linePix[2].x())) =
					rgbB;
			}
		}

		if (err < bestMatchErr) {
			//Move best match to second place.
			secondBestMatchErr = bestMatchErr;
			secondBestMatchPix = bestMatchPix;
			secondBestMatchPos = bestMatchPos;
			loopCSecondBest = loopCBest;
			//Replace best match
			bestMatchErr = err;
			bestMatchErrPre = errLast;
			bestMatchDiffErrPre =
				e0[0] * e1[0] +
				e0[1] * e1[1] +
				e0[2] * e1[2] +
				e0[3] * e1[3] +
				e0[4] * e1[4];
			bestWasLastLoop = true;
			bestMatchDiffErrPost = -1;
			bestMatchErrPost = -1;
			bestMatchPix = linePix[2];
			bestMatchPos = lineDir[2];
			bestMatchPre = linePix[1];
			bestMatchPost = linePix[3];
			loopCBest = loopC;
			bestEpImDir = linePix[3] - linePix[2];
			if (bestEpImDir == vec2::Zero()) {
				bestEpImDir = linePix[2] - linePix[1];
				if (bestEpImDir == vec2::Zero()) {
					return -4;
				}
			}
			bestMatchA = centerA;
		}
		else {
			if (bestWasLastLoop) {
				bestMatchErrPost = err;
				bestMatchDiffErrPre =
					e0[0] * e1[0] +
					e0[1] * e1[1] +
					e0[2] * e1[2] +
					e0[3] * e1[3] +
					e0[4] * e1[4];
				bestWasLastLoop = false;
			}

			if (err < secondBestMatchErr) {
				//Replace second best match.
				secondBestMatchErr = err;
				secondBestMatchPix = linePix[2];
				secondBestMatchPos = lineDir[2];
				loopCSecondBest = loopC;
			}
		}

		//Shuffle values down
		for (size_t i = 0; i < 4; ++i) {
			lineDir[i] = lineDir[i + 1];
			linePix[i] = linePix[i + 1];
			lineValue[i] = lineValue[i + 1];
		}
		tracedLineLen += (linePix[2] - linePix[1]).norm();
		++loopC;
		errLast = err;
	}

	if (drawThisMatch) {
		vec3 rgb(0, 255.f, 255.f);
		cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
		//ssdColors.push_back(rgbB);
		if (drawMatch.cols == width * 2) {
			drawMatch.at<cv::Vec3b>(int(bestMatchPix.y()), int(bestMatchPix.x() + width)) =
				rgbB;
		}
	}

	//Check if epipolar line left image before any error vals could be found.
	if (loopCBest < 0) {
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}

	// if error too big, will return -3, otherwise -2.
	if (bestMatchErr > 4.0f*(float)MAX_ERROR_STEREO)
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return DepthMapErrCode::ERR_TOO_BIG;
	}

	// check if clear enough winner
	if (loopCBest >= 0 && loopCSecondBest >= 0) {
		if (abs(loopCBest - loopCSecondBest) > 1.0f &&
			MIN_DISTANCE_ERROR_STEREO * bestMatchErr > secondBestMatchErr) {
			if (enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
			return DepthMapErrCode::WINNER_NOT_CLEAR;
		}
	}

	//Perform subpixel matching, if necessary.
	bool didSubpixel = false;
	vec2 origpos = bestMatchPix;
	//TODO fix and re-enable subpixel.
	if (false /*useSubpixelStereo*/) {
		didSubpixel = subpixelMatchOmni(
			&bestMatchPix, bestMatchPre, bestMatchPost,
			&bestMatchErr, 
			bestMatchErrPre, bestMatchDiffErrPre,
			bestMatchErrPost, bestMatchDiffErrPost,
			stats);
		if (didSubpixel) {
			//std::cout << "SUBPIXEL: " << origpos << "\n" << bestMatchPix << std::endl;
			bestMatchPos = oModel.pixelToCam(bestMatchPix);
		}
	}

	gradAlongLine = 0.f;
	float tmp = valuesToFind[4] - valuesToFind[3];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[3] - valuesToFind[2];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[2] - valuesToFind[1];  gradAlongLine += tmp*tmp;
	tmp = valuesToFind[1] - valuesToFind[0];  gradAlongLine += tmp*tmp;
	gradAlongLine /= GRADIENT_SAMPLE_DIST*GRADIENT_SAMPLE_DIST;

	// check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
	if (bestMatchErr > (float)MAX_ERROR_STEREO /* + sqrtf(gradAlongLine)*  20*/) {
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return DepthMapErrCode::ERR_TOO_BIG;
	}


	bestEpImDir.normalize();
	if (bestEpImDir != bestEpImDir) {
		std::cout << "bestEpImDir != bestEpImDir" << std::endl;
		throw std::runtime_error("bestEpImDir != bestEpImDir");
	}

	iDepth = bestMatchA * max_idepth;
	
	return bestMatchErr;
}

}

