#ifndef LSD_SLAM_CAMERAMOTION_HPP_INCLUDED
#define LSD_SLAM_CAMERAMOTION_HPP_INCLUDED

#include "VectorTypes.hpp"

namespace lsd_slam {

class NoNewTransformException : public std::runtime_error {
	NoNewTransformException(const std::string &e) :std::runtime_error(e.c_str())
	{}
	std::string what() { return std::runtime_error::what(); }
};

class CameraMotion {
public:
	explicit CameraMotion();
	virtual ~CameraMotion() throw();

	virtual WorldToCamTransform getNextTransform() = 0;
};

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

#endif

