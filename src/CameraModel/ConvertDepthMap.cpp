#include "ConvertDepthMap.hpp"

namespace lsd_slam {

void depthMapOmniToProj(float * input, float * output, CameraModel * model)
{
	for (size_t r = 0; r < model->w; ++r) {
		for (size_t c = 0; c < model->h; ++c) {
			vec3 camPos = model->pixelToCam(vec2(c, r), input[r*model->w + c]);
			float projDepth = camPos.z();
			output[r*model->w + c] = projDepth;
		}
	}
}

void depthMapProjToOmni(float * input, float * output, CameraModel * model)
{
	std::unique_ptr<CameraModel> p = model->makeProjCamModel();
	for (size_t r = 0; r < model->w; ++r) {
		for (size_t c = 0; c < model->h; ++c) {
			vec3 camPos = p->pixelToCam(vec2(c, r), input[r*model->w + c]);
			float omniDepth = camPos.norm();
			output[r*model->w + c] = omniDepth;
		}
	}
}

}
