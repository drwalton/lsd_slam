#include "CameraMotion.hpp"
#include "Constants.hpp"

namespace lsd_slam {

CameraMotion::CameraMotion()
{}
CameraMotion::~CameraMotion() throw()
{}


OscillatingCameraMotion::OscillatingCameraMotion(
	const WorldToCamTransform &initialTransform, vec3 maxDisplacement, int period)
	:origin_(initialTransform.translation), maxDisplacement_(maxDisplacement), rotation_(initialTransform.rotation),
	step_(2 * M_PI / static_cast<float>(period)), angle_(0.f)
{}

OscillatingCameraMotion::~OscillatingCameraMotion() throw()
{}

WorldToCamTransform OscillatingCameraMotion::getNextTransform()
{
	WorldToCamTransform t(rotation_, origin_ + cosf(angle_)*maxDisplacement_);

	angle_ += step_;
	if (angle_ >= 2 * M_PI) angle_ -= 2 * M_PI;

	return t;
}

}
