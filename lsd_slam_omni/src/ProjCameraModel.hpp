#ifndef LSD_OMNI_PROJCAMERAMODEL_HPP_INCLUDED
#define LSD_OMNI_PROJCAMERAMODEL_HPP_INCLUDED

#include "CameraModel.hpp"

namespace lsd_slam {

class ProjCameraModel : public CameraModel {
public:
	ProjCameraModel(float fx, float fy, float cx, float cy, size_t w, size_t h);
	virtual ~ProjCameraModel();

	virtual vec2 camToPixel(const vec3& p) const;

	virtual vec3 pixelToCam(const vec2 &p, float d = 1.f) const;

	///\brief Create a pyramid of camera models. The model at level 0 corresponds
	///       to the full-size image, and subsequent levels have the width
	///       and height of the image halved.
	virtual std::vector<std::unique_ptr<CameraModel> >
		createPyramidCameraModels(int nLevels) const;

	virtual CameraModelType getType() const;

	const mat3 K, Kinv;

	virtual std::unique_ptr<CameraModel> clone() const;
};

}

#endif
