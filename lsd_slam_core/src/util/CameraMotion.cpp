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


EllipticalCameraMotion::EllipticalCameraMotion(const WorldToCamTransform &initialTransform,
	vec3 axisA, vec3 axisB, int period)
	:origin_(initialTransform.translation), rotation_(initialTransform.rotation),
	axisA_(axisA), axisB_(axisB),
	step_(2 * M_PI / static_cast<float>(period)), angle_(0.f)
{}

EllipticalCameraMotion::~EllipticalCameraMotion() throw()
{}

WorldToCamTransform EllipticalCameraMotion::getNextTransform()
{
	WorldToCamTransform t(rotation_, 
		origin_ + cosf(angle_)*axisA_ + sinf(angle_)*axisB_);

	angle_ += step_;
	if (angle_ >= 2 * M_PI) angle_ -= 2 * M_PI;

	return t;
}
}
