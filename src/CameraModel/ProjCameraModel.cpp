#include "ProjCameraModel.hpp"
#include "OmniCameraModel.hpp"
#include <Eigen/Dense>

namespace lsd_slam {

ProjCameraModel::ProjCameraModel(float fx, float fy, float cx, float cy, size_t w, size_t h)
	:CameraModel(fx, fy, cx, cy, w, h),
	K((mat3() << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f).finished()),
	Kinv(K.inverse())
{}

ProjCameraModel::~ProjCameraModel()
{}

vec2 ProjCameraModel::camToPixel(const vec3 &p) const
{
	vec3 hom = K * p;
	return vec2(hom.x() / hom.z(), hom.y() / hom.z());
}

vec3 ProjCameraModel::pixelToCam(const vec2 &p, float d) const
{
	return vec3(
		(p.x()*fxi() + cxi()) * d,
		(p.y()*fyi() + cyi()) * d,
		d);
}

std::vector<std::unique_ptr<CameraModel> >
	ProjCameraModel::createPyramidCameraModels(int nLevels) const
{
	std::vector<std::unique_ptr<CameraModel> > models;
	ProjCameraModel *newModel = new ProjCameraModel(*this);
	models.reserve(nLevels);
	models.emplace_back(newModel);
	for (size_t level = 1; level < static_cast<size_t>(nLevels); ++level) {
		if (models.back()->w % 2 != 0 || models.back()->h % 2 != 0) {
			throw std::runtime_error("Could not make a " 
				+ std::to_string(nLevels) + "-level pyramid - not divisible!");
		}
		
		int newW = models.back()->w / 2;
		int newH = models.back()->h / 2;
		float newFx = models.back()->fx * 0.5f;
		float newFy = models.back()->fy * 0.5f;
		float newCx = (this->cx + 0.5f) / static_cast<float>(1 << level) - 0.5f;
		float newCy = (this->cy + 0.5f) / static_cast<float>(1 << level) - 0.5f;

		newModel = new ProjCameraModel(newFx, newFy, newCx, newCy, newW, newH);
		models.emplace_back(newModel);
	}
	return models;
}

CameraModelType ProjCameraModel::getType() const
{
	return CameraModelType::PROJ;
}

std::string ProjCameraModel::getTypeName() const
{
	return "Proj";
}

std::unique_ptr<CameraModel> ProjCameraModel::clone() const
{
	return std::unique_ptr<CameraModel>(new ProjCameraModel(*this));
}


vec2 ProjCameraModel::getFovAngles() const
{
	return vec2(
		2.f * atan2f((float)(w / fx), 2.0f),
		2.f * atan2f((float)(h / fy), 2.0f)
	);
}

bool ProjCameraModel::pixelLocValid(const vec2 &loc) const
{
	return loc.x() > 0.f && loc.x() < w && loc.y() > 0.f && loc.y() < h;
}

std::unique_ptr<CameraModel> ProjCameraModel::makeProjCamModel() const
{
	return clone();
}

std::unique_ptr<CameraModel> ProjCameraModel::makeOmniCamModel() const
{
	return std::unique_ptr<CameraModel>(
		new OmniCameraModel(fx, fy, cx, cy, w, h, 0.f, vec2(w/2, h/2), w*h));
}

}
