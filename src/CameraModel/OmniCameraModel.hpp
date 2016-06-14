#pragma once

#include "CameraModel.hpp"
#include <functional>

namespace lsd_slam {

///\brief Omnidirectional camera model used in LSD-SLAM.
///\note This model does not account for radial tangential distortion - 
///      images used should have correction applied already.
class OmniCameraModel : public CameraModel
{
public:
	///\brief The default model covers a 180 degree fov in a circle of radius 200.
	static OmniCameraModel makeDefaultModel();

	///\brief Distance from the image plane in the omnidirectional model.
	float e;

	///\brief Center of the circle within the image containing valid colour values
	vec2 c;

	///\brief Radius of the circle within the image containing valid color values
	float r;

	OmniCameraModel(float fx, float fy, float cx, float cy, 
		size_t w, size_t h, float e, vec2 c, float r);
	virtual ~OmniCameraModel();

	///\brief Forward projection function, mapping from camera space to
	///       image space.
	virtual vec2 camToPixel(const vec3 &p) const;

	///\brief Inverse projection function, mapping from image space to
	///       camera space, given a depth value d.
	///\note The depth d defaults to 1, giving a unit vector in the 
	///      direction associated with this pixel.
	///\note d should be strictly greater than 0 (if not, incorrect values
	///      or NaNs will be returned).
	///\note No checking is performed to ensure the projected point
	///      corresponds to a valid pixel.
	virtual	vec3 pixelToCam(const vec2 &p, float d = 1.f) const;

	virtual std::vector<std::unique_ptr<CameraModel> >
		createPyramidCameraModels(int nLevels) const;

	///\brief Given the line segment parameterised by a*p0 + (1-a)*p1 in world space,
	///       estimates the increment in "a" necessary to move a given distance 
	///       further along the line in image space. 
	///\note Used e.g. to efficiently traverse epipolar lines.
	///\param stepSize Size of step to take in image space. Measured in pixels.
	float getEpipolarParamIncrement(float a, vec3 p0, vec3 p1,
		float stepSize = 1.f) const;

	///\brief Apply a function object to each pixel location along the projection
	///       of a line segment in image space.
	///       The supplied function is applied once with each successive pixel location
	///       along the line.
	///\note The line is parameterised by a*p0 + (1-a)*p1, for a in [0, 1].
	///\note  Intended to be used by supplying a lambda as the third argument.
	///\note Efficient, but suitable for relatively short lines only (accumulates errors
	///      and may skip pixels for longer lines).
	void traceWorldSpaceLine(vec3 p0, vec3 p1, std::function<void(vec2)> f);

	///\brief Given a point in camera space, determines whether it projects
	///       to a valid pixel location in image space.
	///\param p The point to attempt to project into image space.
	///\param[out] pixelLoc The projected pixel location (optional).
	///\return true if the location is valid, false otherwise.
	bool pointInImage(const vec3 &p) const;

	///\brief Checks if a pixel location is valid (i.e. lies within the circle
	///       defined by the parameters c, r.
	virtual bool pixelLocValid(const vec2 &p) const;

	virtual CameraModelType getType() const;

	virtual std::unique_ptr<CameraModel> clone() const;

	const vec2 fovAngles;

	virtual std::unique_ptr<CameraModel> makeProjCamModel() const;
	
	virtual vec2 getFovAngles() const;
private:
	const float minDotProduct;
};

}
