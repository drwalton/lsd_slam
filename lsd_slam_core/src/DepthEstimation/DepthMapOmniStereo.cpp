#include "DepthMapOmniStereo.hpp"
#include "globalFuncs.hpp"

namespace lsd_slam {

std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width)
{
	vec3 pointDir = model.pixelToCam(vec2(x, y));
	vec3 epipoleDir = -keyframeToReference.translation;

	float a = 1.f;
	//Advance two pixels from point toward epipole.
	a = model.getEpipolarParamIncrement(a, pointDir, epipoleDir);
	vec3 fwdDir1 = a*pointDir + (1.f - a)*epipoleDir;
	a = model.getEpipolarParamIncrement(a, pointDir, epipoleDir);
	vec3 fwdDir2 = a*pointDir + (1.f - a)*epipoleDir;

	//Advance two pixels from point away from epipole.
	a = 1.f;
	vec3 otherDir = 2.f*pointDir - epipoleDir;
	a = model.getEpipolarParamIncrement(a, pointDir, otherDir);
	vec3 bwdDir1 = a*pointDir + (1.f - a)*otherDir;
	a = model.getEpipolarParamIncrement(a, pointDir, otherDir);
	vec3 bwdDir2 = a*pointDir + (1.f - a)*otherDir;

	//Find values of keyframe at these points.
	std::array<float, 5> vals = {
		getInterpolatedElement(keyframe, model.camToPixel(bwdDir2), width),
		getInterpolatedElement(keyframe, model.camToPixel(bwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(pointDir), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir1), width),
		getInterpolatedElement(keyframe, model.camToPixel(fwdDir2), width),
	};

	return vals;
}

}

