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
	std::ifstream file;
	file.open(filename, std::ios::in);
	if(!file.is_open()) {
		throw std::runtime_error("Unable to read from camera model file");
	}
	std::string type;
	std::getline(file, type);

	float fx, fy, cx, cy;
	int w, h;
	file >> fx >> fy >> cx >> cy >> w >> h;
	
	if (type == "OMNI" || type == "OMNI\r") {
		vec2 c;
		float e, r;
		file >> e >> c.x() >> c.y() >> r;
		return std::unique_ptr<CameraModel>(
			new OmniCameraModel(fx, fy, cx, cy, w, h, e, c, r));
	}
	else if (type == "PROJ" || type == "PROJ\r") {
		return std::unique_ptr<CameraModel>(
			new ProjCameraModel(fx, fy, cx, cy, w, h));
	}
	else {
		throw std::runtime_error("Invalid camera model file.");
	}
}

std::ostream &operator<<(std::ostream &s, const CameraModel & model)
{
	if (model.getType() == CameraModelType::PROJ) {
		s << "** Projective Camera Model **\n";
	} else {
		s << "** Omnidirectional Camera Model **\n";
	}
	s << "dims = (" << model.w << ", " << model.h << ")\n";
	s << "f = (" << model.fx << ", " << model.fy << ")\n";
	s << "c = (" << model.cx << ", " << model.cy << ")\n";

	if (model.getType() == CameraModelType::OMNI) {
		auto o = dynamic_cast<const OmniCameraModel*>(&model);
		s << "e = " << o->e << "\n";
	} 

	s << std::endl;

	return s;
}

}

