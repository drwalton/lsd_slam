#ifndef LSD_OMNI_OMNICAMERAMODEL_HPP_INCLUDED
#define LSD_OMNI_OMNICAMERAMODEL_HPP_INCLUDED

#include "VectorTypes.hpp"
#include <functional>

namespace lsd_slam {

///\brief Omnidirectional camera model used in LSD-SLAM.
///\note This model does not account for radial tangential distortion - 
///      images used should have correction applied already.
class OmniCameraModel
{
public:

	///\brief The default model covers a 180 degree fov in a circle of radius 200.
	static OmniCameraModel makeDefaultModel();

	float fx, fy, cx, cy, e;

	///\brief Forward projection function, mapping from camera space to
	///       image space.
	vec2 worldToPixel(vec3 p) const;

	///\brief Inverse projection function, mapping from image space to
	///       camera space, given a depth value d.
	///\note The depth d defaults to 1, giving a unit vector in the 
	///      direction associated with this pixel.
	///\note d should be strictly greater than 0 (if not, incorrect values
	///      or NaNs will be returned).
	vec3 pixelToWorld(vec2 p, float d = 1.f) const;

	///\brief Given the line segment parameterised by a*p0 + (1-a)*p1 in world space,
	///       estimates the increment in "a" necessary to move one pixel further along the 
	///       line in image space. 
	///\note Used e.g. to efficiently traverse epipolar lines.
	float getEpipolarParamIncrement(float a, vec3 p0, vec3 p1) const;

	///\brief Apply a function object to each pixel location along the projection
	///       of a line segment in image space.
	///       The supplied function is applied once with each successive pixel location
	///       along the line.
	///\note The line is parameterised by a*p0 + (1-a)*p1, for a in [0, 1].
	///\note  Intended to be used by supplying a lambda as the third argument.
	///\note Efficient, but suitable for relatively short lines only (accumulates errors
	///      and may skip pixels for longer lines).
	void traceWorldSpaceLine(vec3 p0, vec3 p1, std::function<void(vec2)> f);
};

}

#endif
