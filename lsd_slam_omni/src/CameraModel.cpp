#include "CameraModel.hpp"

namespace lsd_slam {

CameraModel::CameraModel(float fx, float fy, float cx, float cy, size_t w, size_t h)
	:fx(fx), fy(fy), cx(cx), cy(cy), w(w), h(h)
{}

CameraModel::~CameraModel()
{}

}

