#include "DepthMapOmniStereo.hpp"
#include "DepthMap.hpp"
#include "globalFuncs.hpp"

#include <opencv2/opencv.hpp>

///This file contains the depth estimation functions specific to Omnidirectional
/// camera models.

#define SHOW_DEBUG_IMAGES

namespace lsd_slam {

float DepthMap::doOmniStereo(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	//TODO
	return 0.f;
}

bool DepthMap::makeAndCheckEPLOmni(const int x, const int y, const Frame* const ref,
	vec3 *epDir, RunningStats* const stats)
{
	//TODO
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

