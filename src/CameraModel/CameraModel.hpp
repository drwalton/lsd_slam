#pragma once

#include "Util/VectorTypes.hpp"
#include <vector>
#include <memory>

namespace lsd_slam {

enum class CameraModelType { OMNI, PROJ };

///\brief Abstract class encapsulating a camera model.
class CameraModel {
public:
	CameraModel(float fx, float fy, float cx, float cy, size_t w, size_t h);
	virtual ~CameraModel();

	const float fx, fy, cx, cy;

	const size_t w, h;

	///\brief Forward projection function, mapping from camera space to
	///       image space.
	virtual vec3 pixelToCam(const vec2 &pixel, float depth = 1.f) const = 0;

	vec3 camToPixelDepth(const vec3 &cam) {
		vec2 pix = camToPixel(cam);
		return vec3(pix.x(), pix.y(), cam.z());
	}

	///\brief Inverse projection function, mapping from image space to
	///       camera space, given a depth value d.
	///\note The depth d defaults to 1, giving a unit vector in the 
	///      direction associated with this pixel.
	///\note d should be strictly greater than 0 (if not, incorrect values
	///      or NaNs will be returned).
	virtual vec2 camToPixel(const vec3 &cam) const = 0;

	///\brief Create a pyramid of camera models. The model at level 0 corresponds
	///       to the full-size image, and subsequent levels have the width
	///       and height of the image halved.
	virtual std::vector<std::unique_ptr<CameraModel> >
		createPyramidCameraModels(int nLevels) const = 0;

	virtual CameraModelType getType() const = 0;

	virtual std::unique_ptr<CameraModel> clone() const = 0;

	virtual vec2 getFovAngles() const = 0;

	static std::unique_ptr<CameraModel> loadFromFile(const std::string &filename);
	
	///\brief Check if a pixel location corresponds to a valid location in the
	/// image.
	///\note For projective camera models, this just checks that the point is
	/// within the image width and height. For omnidirectional models, it also
	/// checks that the point is inside the valid image circle (typically set
	/// to the
	/// <a href="https://en.wikipedia.org/wiki/Image_circle">image circle</a>
	/// of the lens).
	virtual bool pixelLocValid(const vec2 &loc) const = 0;

	virtual std::string getTypeName() const = 0;

	virtual std::unique_ptr<CameraModel> makeProjCamModel() const = 0;
	virtual std::unique_ptr<CameraModel> makeOmniCamModel() const = 0;
};

std::ostream &operator << (std::ostream &s, const CameraModel &model);

}
