#include "DepthMapOmniStereo.hpp"
#include "DepthMap.hpp"
#include "globalFuncs.hpp"
#include "DataStructures/Frame.hpp"

#include <opencv2/opencv.hpp>

///This file contains the depth estimation functions specific to Omnidirectional
/// camera models.

#define SHOW_DEBUG_IMAGES

namespace lsd_slam {

bool epipolarLineInImageOmni(vec3 lineStart, vec3 lineEnd, const OmniCameraModel &model);
std::array<float, 5> getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel,
	int u, int v);

void padEpipolarLineOmni(vec3 *lineStart, vec3 *lineEnd,
	vec2* lineStartPix, vec2* lineEndPix,
	float minLength,
	const OmniCameraModel &oModel);

float DepthMap::doOmniStereo(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	OmniCameraModel &oModel = static_cast<OmniCameraModel&>(*model);
	if (enablePrintDebugInfo) stats->num_stereo_calls++;

	//Find line containing possible positions for the match, in the frame of 
	// the reference image.
	//N.B. This line moves away from the epipole in the reference image.
	vec3 keyframePointDir = oModel.pixelToCam(vec2(u, v), 1.f);
	vec3 expectedMatchPos = referenceFrame->otherToThis_R
		* (keyframePointDir / prior_idepth) 
		+ referenceFrame->otherToThis_t;
	vec3 lineStartPos = referenceFrame->otherToThis_R
		* (keyframePointDir / max_idepth) 
		+ referenceFrame->otherToThis_t;
	vec3 lineEndPos = referenceFrame->otherToThis_R
		* (keyframePointDir / min_idepth) 
		+ referenceFrame->otherToThis_t;

	if (!epipolarLineInImageOmni(lineStartPos, lineEndPos, oModel))	{
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return -1;
	}

	//Find values to search for in keyframe.
	//These values are 5 samples, advancing along the epipolar line towards the
	//epipole, centred at the input point u,v.
	std::array<float, 5> valuesToFind = getValuesToFindOmni(keyframePointDir,
		epDir, activeKeyFrameImageData, width, oModel, u, v);

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
	float bestMatchErr, secondBestMatchErr;
	vec2 bestMatchPix, secondBestMatchPix;
	vec3 bestMatchPos, secondBestMatchPos;
	float centerA = 0.f;

	//Find first 4 values along line.
	lineDir[2] = lineStartPos;
	linePix[2] = lineStartPix;
	lineValue[2] = getInterpolatedElement(referenceFrameImage, lineStartPix, width);
	float a = 1.f; vec3 dir;
	a += oModel.getEpipolarParamIncrement(a, lineStartPos, lineEndPos);
	lineDir[1] = a*lineStartPos + (1.f - a)*lineEndPos;
	linePix[1] = oModel.camToPixel(lineDir[1]);
	lineValue[1] = getInterpolatedElement(referenceFrameImage, lineDir[1], width);
	a += oModel.getEpipolarParamIncrement(a, lineStartPos, lineEndPos);
	lineDir[0] = a*lineStartPos + (1.f - a)*lineEndPos;
	linePix[0] = oModel.camToPixel(lineDir[1]);
	lineValue[0] = getInterpolatedElement(referenceFrameImage, lineDir[0], width);
	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, lineEndPos, lineStartPos);
	lineDir[3] = a*lineEndPos + (1.f - a)*lineStartPos;
	linePix[3] = oModel.camToPixel(lineDir[3]);
	lineValue[3] = getInterpolatedElement(referenceFrameImage, lineDir[3], width);

	//Advance along line.
	while (centerA <= 1.f) {
		//Update centerA

		//Find fifth entry

		//Check error

		//Update if appropriate

		//Shuffle values down
		//Update 

	}

	//TODO
	return 0.f;
}

bool epipolarLineInImageOmni(vec3 lineStart, vec3 lineEnd, const OmniCameraModel &model)
{
	//TODO
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
		float pad = (minLength - (eplLength)) / 2.0f;
		float a = 1.f;

		//TODO

		*lineStartPix = oModel.camToPixel(*lineStart);
		*lineEndPix = oModel.camToPixel(*lineEnd);
	}
}

std::array<float, 5> getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel, 
	int u, int v)
{

	std::array<float, 5> valuesToFind;
	valuesToFind[2] = getInterpolatedElement(activeKeyFrameImageData, u, v, width);
	float a = 1.f; vec3 dir; vec2 pixel;
	a += oModel.getEpipolarParamIncrement(a, keyframePointDir, epDir);
	dir = a*keyframePointDir + (1.f - a)*epDir;
	valuesToFind[1] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);
	a += oModel.getEpipolarParamIncrement(a, keyframePointDir, epDir);
	dir = a*keyframePointDir + (1.f - a)*epDir;
	valuesToFind[0] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);

	a = 0.f;
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir);
	dir = a*epDir + (1.f - a)*keyframePointDir;
	valuesToFind[3] = getInterpolatedElement(activeKeyFrameImageData, oModel.camToPixel(dir), width);
	a += oModel.getEpipolarParamIncrement(a, epDir, keyframePointDir);
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
	vec2 &matchPixel)
{
	vec3 pointDir;
	std::array<float, 5> searchVals = findValuesToSearchFor(keyframeToReference,
		model, keyframe, x, y, width, pointDir);

	vec3 minDPointDir = minDepth * pointDir;
	vec3 maxDPointDir = maxDepth * pointDir;

	//Find start and end of search curve
	vec3 lineStart = keyframeToReference * minDPointDir;
	vec3 lineEnd   = keyframeToReference * maxDPointDir;
	
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
	a += model.getEpipolarParamIncrement(a, lineEnd, lineStart); 
	if(a > 1.f) return false;
	matchDirs[3] = a*lineEnd + (1.f - a)*lineStart;
	matchVals[3] = getInterpolatedElement(reference, model.camToPixel(matchDirs[3]), width);

	a += model.getEpipolarParamIncrement(a, lineEnd, lineStart);
	if(a > 1.f) return false;
	matchDirs[4] = a*lineEnd + (1.f - a)*lineStart;
	matchVals[4] = getInterpolatedElement(reference, model.camToPixel(matchDirs[4]), width);
	float ssd = findSsd(searchVals, matchVals);

	matchDir = matchDirs[2];
	minSsd = ssd;

	//Advance along remainder of line.
	while (a < 1.f) {
		a += model.getEpipolarParamIncrement(a, lineEnd, lineStart);
		if (a > 1.f) break;
		
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
	}

	matchPixel = model.camToPixel(matchDir);

	return true;
}



std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width,
	vec3 &pointDir)
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
	
#ifdef SHOW_DEBUG_IMAGES
	cv::Mat lineImage(480, 640, CV_8UC1);
	lineImage.setTo(0);
	
	std::vector<vec3> dirs = {
		bwdDir2, bwdDir1, pointDir, fwdDir1, fwdDir2
	};
	
	for(vec3 &v : dirs) {
		vec2 img = model.camToPixel(v);
		static int i = 0;
		std::cout << "Point " << i++ << ": " << int(img.x()) << ", " << int(img.y()) << std::endl;
		lineImage.at<uchar>(img.y(), img.x()) = 255;
	}
	
	cv::imshow("LINE", lineImage);
	cv::waitKey();
#endif
	
	//Find values of keyframe at these points.
	std::array<float, 5> vals = {
		getInterpolatedElement(keyframe, model.camToPixel(bwdDir2), width),
		getInterpolatedElement(keyframe, model.camToPixel(bwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(pointDir), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir2), width)
	};

	return vals;
}



}

