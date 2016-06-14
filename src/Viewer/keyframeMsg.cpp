#include "keyframeMsg.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"

namespace lsd_slam {

std::unique_ptr<CameraModel> lsd_slam::keyframeMsg::getCameraModel() const
{
	if (modelType == CameraModelType::PROJ) {
		return std::unique_ptr<CameraModel>(
			new ProjCameraModel(fx, fy, cx, cy, width, height));
	} else {
		return std::unique_ptr<CameraModel>(
			new OmniCameraModel(fx, fy, cx, cy, width, height, e, c, r));
	}
}

void keyframeMsg::setCameraModel(const CameraModel & model)
{
	fx = model.fx;
	fy = model.fy;
	cx = model.cx;
	cy = model.cy;
	modelType = model.getType();

	if (model.getType() == CameraModelType::OMNI) {
		const OmniCameraModel *o = static_cast<const OmniCameraModel*>(&(model));
		e = o->e;
		c = o->c;
		r = o->r;
	}
}

}
