#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "Util/globalFuncs.hpp"
#include "DataStructures/Frame.hpp"

#include <opencv2/opencv.hpp>

///This file contains the depth estimation functions specific to Omnidirectional
/// camera models.

#define SHOW_DEBUG_IMAGES 0
#define EPIPOLE_MAX_CLOSENESS (0.95f) //Max dot product with epipole in stereo.

namespace lsd_slam {

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
	float geoDispError = (gradsInterp[0] * bestEpDir[0] + 
		gradsInterp[1] * bestEpDir[1]) + DIVISION_EPS;
	geoDispError = trackingErrorFac*trackingErrorFac*
		(gradsInterp[0] * gradsInterp[0] + gradsInterp[1] * gradsInterp[1]) / 
		(geoDispError*geoDispError);

	// final error consists of a small constant part (discretization error),
	// geometric and photometric error.
	return depth*depth*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist
		+ geoDispError + photoDispError);	// square to make variance
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

float doOmniStereo(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	vec2 &bestEpDir, vec3 &bestMatchPos, float &gradAlongLine, float &tracedLineLen,
	cv::Mat &drawMatch, bool plotSearch)
{
	if (enablePrintDebugInfo) stats->num_stereo_calls++;

	//Find line containing possible positions for the match, in the frame of 
	// the reference image.
	//N.B. This line moves away from the epipole in the reference image.
	vec3 keyframePointDir = oModel.pixelToCam(vec2(u, v), 1.f);

	float epipoleCloseness = fabsf(keyframePointDir.dot(epDir));
	if (epipoleCloseness > EPIPOLE_MAX_CLOSENESS) {
		//This point is too near an epipole - we won't get any reliable depth
		// values from it!
		if (enablePrintDebugInfo) ++stats->num_stereo_inf_oob;
		if (plotSearch) {
			std::cout << "Stereo failed due to proximity to epipole!" << std::endl;
		}
		return -1;
	}

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
	std::array<float, 5> valuesToFind;
	if(!getValuesToFindOmni(keyframePointDir,
		epDir, keyframe, width, oModel, u, v, valuesToFind)) {
		//5 values centered around point not available.
		return -1;
	}
	
	if(plotSearch) {
		cv::Mat findValMat(cv::Size(5,1), CV_8UC1);
		for(size_t i = 0; i < 5; ++i) {
			findValMat.at<uchar>(0,i) = uchar(valuesToFind[i]);
		}
		cv::Mat showMat;
		cv::resize(findValMat, showMat, cv::Size(), 10.f, 10.f, cv::INTER_NEAREST);
		cv::imshow("TO FIND", showMat);
	}

	vec2 lineStartPix = oModel.camToPixel(lineStartPos);
	vec2 lineEndPix   = oModel.camToPixel(lineEndPos  );
	padEpipolarLineOmni(&lineStartPos, &lineEndPos, &lineStartPix, &lineEndPix,
		MIN_EPL_LENGTH_CROP, oModel);
	if (!drawMatch.empty()){
		cv::circle(drawMatch, vec2Point(lineStartPix), 3, cv::Scalar(255, 0, 0));
		cv::circle(drawMatch, vec2Point(lineEndPix), 3, cv::Scalar(255, 0, 0));
		vec3 lineMidPos   = keyframeToReference * vec3(keyframePointDir / prior_idepth);
		vec2 lineMidPix = oModel.camToPixel(lineMidPos);
		cv::circle(drawMatch, vec2Point(lineMidPix), 3, cv::Scalar(0, 0, 255));
	}

	//Check padded line still in image.
	if (!epipolarLineInImageOmni(lineStartPos, lineEndPos, oModel))	{
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
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
	vec2 bestMatchPre, bestMatchPost; //Pixel locs of pixels just before, after best match.
	vec3 secondBestMatchPos;
	float centerA = 0.f;
	int loopCBest = -1, loopCSecondBest = -1;
	
	std::vector<float> searchedVals;
	std::vector<cv::Vec3b> ssdColors;

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
	
	if(plotSearch) {
		for(size_t i = 0; i < 4; ++i) {
			searchedVals.push_back(lineValue[i]);
		}
	}

	tracedLineLen = 0.f;
	//Advance along line.
	size_t loopC = 0;
	float errLast = -1.f;
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
		if(plotSearch) {
			searchedVals.push_back(lineValue[4]);
		}

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

		if (!drawMatch.empty()) {
			vec3 rgb = 255.f * hueToRgb(0.8f * err / 325125.f);
			cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
			ssdColors.push_back(rgbB);
			drawMatch.at<cv::Vec3b>(int(linePix[2].y()), int(linePix[2].x())) =
				rgbB;
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
			bestEpDir = linePix[3] - linePix[2];
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
	
	if(plotSearch) {
		cv::Mat searchValMat(cv::Size(searchedVals.size(), 1), CV_8UC1);
		for(size_t i = 0; i < searchedVals.size(); ++i) {
			searchValMat.at<uchar>(0,i) = uchar(searchedVals[i]);
		}
		cv::Mat showMat;
		cv::resize(searchValMat, showMat, cv::Size(), 10.f, 10.f, cv::INTER_NEAREST);
		cv::imshow("SEARCHED", showMat);
		cv::Mat ssdMat(cv::Size(ssdColors.size(), 1), CV_8UC3);
		for(size_t i = 0; i < ssdColors.size(); ++i) {
			ssdMat.at<cv::Vec3b>(0, i) = ssdColors[i];
		}
		cv::resize(ssdMat, showMat, cv::Size(), 10.f, 10.f, cv::INTER_NEAREST);
		cv::imshow("ERRS", showMat);
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
		if (plotSearch) {
			std::cout << "Stereo failed due to large absolute error value: "
				<< bestMatchErr << " > " << 4.0f * float(MAX_ERROR_STEREO)
				<< std::endl;
		}
		return -3;
	}

	// check if clear enough winner
	if (loopCBest >= 0 && loopCSecondBest >= 0) {
		if (abs(loopCBest - loopCSecondBest) > 1.0f && 
			MIN_DISTANCE_ERROR_STEREO * bestMatchErr > secondBestMatchErr) {
			if (enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
			if (plotSearch) {
				std::cout << "Stereo failed due to unclear winner: "
					<< "First and second best are "
					<< abs(loopCBest - loopCSecondBest)
					<< " apart, and errors are " << bestMatchErr << " * "
					<< MIN_DISTANCE_ERROR_STEREO << " > " << secondBestMatchErr
					<< std::endl;
			}
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
		if (plotSearch) {
			std::cout << "Stereo failed: absolute error too large (second check)"
				<< bestMatchErr << " > " << MAX_ERROR_STEREO << " + "
				<< sqrtf(gradAlongLine) << " * " << 20 << std::endl;
		}
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
		stats, oModel, referenceFrame->width(), bestEpDir, bestMatchPos, gradAlongLine, tracedLineLen);

	bestEpDir.normalize();
	float r = findDepthAndVarOmni(u, v, bestMatchPos, &result_idepth, &result_var, 
		gradAlongLine, bestEpDir, referenceFrame, activeKeyFrame, 
		GRADIENT_SAMPLE_DIST, false, tracedLineLen, &oModel, stats);
	if (r > 0.f) {
		if (settings.saveMatchesImages) {
			debugVisualiseMatch(vec2(u, v), model->camToPixel(bestMatchPos));
		}
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
		float requiredStep = 0.5f*(minLength - eplLength);
		float a = 1.f;
		a += oModel.getEpipolarParamIncrement(a, *lineEnd, *lineStart, requiredStep);
		*lineEnd = a*(*lineEnd) + (1.f - a)*(*lineStart);
		a = 1.f;
		a += oModel.getEpipolarParamIncrement(a, *lineStart, *lineEnd, requiredStep);
		*lineStart = a*(*lineStart) + (1.f - a)*(*lineEnd);

		*lineStartPix = oModel.camToPixel(*lineStart);
		*lineEndPix = oModel.camToPixel(*lineEnd);
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
	if(!oModel.pointInImage(dir, &pixel)) return false;
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[3] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	a += oModel.getEpipolarParamIncrement(a, otherDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*otherDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir, &pixel)) return false;
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[4] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);

	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir, &pixel)) return false;
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[1] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir, GRADIENT_SAMPLE_DIST);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	if(!oModel.pointInImage(dir, &pixel)) return false;
	if(!visIm.empty()) visIm.at<cv::Vec3b>(int(pixel.y()), int(pixel.x())) = cv::Vec3b(0,255,0);
	valuesToFind[0] = getInterpolatedElement(activeKeyFrameImageData, pixel, width);
	
	return true;
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



}

