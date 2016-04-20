#include "CameraModel.hpp"
#include "ProjCameraModel.hpp"
#include "OmniCameraModel.hpp"
#include <fstream>

namespace lsd_slam {

CameraModel::CameraModel(
	float fx, float fy, float cx, float cy, size_t w, size_t h)
	:fx(fx), fy(fy), cx(cx), cy(cy), w(w), h(h)
{}

CameraModel::~CameraModel()
{}


std::unique_ptr<CameraModel> CameraModel::loadFromFile(
	const std::string &filename)
{
	std::ifstream file(filename);
	std::string type;
	std::getline(file, type);

	float fx, fy, cx, cy;
	int w, h;
	file >> fx >> fy >> cx >> cy >> w >> h;
	
	if (type == "OMNI") {
		vec2 c;
		float e, r;
		file >> e >> c.x() >> c.y() >> r;
		return std::unique_ptr<CameraModel>(
			new OmniCameraModel(fx, fy, cx, cy, w, h, e, c, r));
	}
	else if (type == "PROJ") {
		return std::unique_ptr<CameraModel>(
			new ProjCameraModel(fx, fy, cx, cy, w, h));
	}
	else {
		throw std::runtime_error("Unable to read camera model file.");
	}
}

}

