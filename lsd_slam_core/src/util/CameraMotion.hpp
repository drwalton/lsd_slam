#pragma once
#include "VectorTypes.hpp"

namespace lsd_slam {

class NoNewTransformException : public std::runtime_error {
	NoNewTransformException(const std::string &e) :std::runtime_error(e.c_str())
	{}
};

///\brief Abstract class encapsulating the motion of a camera over time.
class CameraMotion {
public:
	explicit CameraMotion();
	virtual ~CameraMotion() throw();

	///\brief Get the transform corresponding the the position of the camera in
	///       the next frame.
	virtual WorldToCamTransform getNextTransform() = 0;
};

///\brief CameraMotion which moves back and forth along an axis through a
///       central point.
class OscillatingCameraMotion : public CameraMotion {
public:
	explicit OscillatingCameraMotion(const WorldToCamTransform &initialTransform, 
		vec3 maxDisplacement, int period);
	virtual ~OscillatingCameraMotion() throw();

	virtual WorldToCamTransform getNextTransform();
private:
	mat3 rotation_;
	float angle_, step_;
	const vec3 origin_, maxDisplacement_;
};

///\brief CameraMotion which moves along an elliptical orbit.
class EllipticalCameraMotion : public CameraMotion {
public:
	explicit EllipticalCameraMotion(const WorldToCamTransform &initialTransform, 
		vec3 axisA, vec3 axisB, int period);
	virtual ~EllipticalCameraMotion() throw();

	virtual WorldToCamTransform getNextTransform();
private:
	mat3 rotation_;
	float angle_, step_;
	const vec3 origin_, axisA_, axisB_;
};

}
