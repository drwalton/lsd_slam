#include "DepthMapOmniStereo.hpp"
#include "globalFuncs.hpp"

#include <opencv2/opencv.hpp>

#define SHOW_DEBUG_IMAGES

namespace lsd_slam {

bool omniStereo(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
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

	vec3 lineStart = keyframeToReference * minDPointDir;
	vec3 lineEnd   = keyframeToReference * maxDPointDir;
	//TODO

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

