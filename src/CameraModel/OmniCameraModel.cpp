#include "OmniCameraModel.hpp"
#include "Constants.hpp"
#include "ProjCameraModel.hpp"

namespace lsd_slam {


OmniCameraModel::OmniCameraModel(
	float fx, float fy, float cx, float cy, size_t w, size_t h, float e,
	vec2 c, float r)
	:CameraModel(fx, fy, cx, cy, w, h), e(e), c(c), r(r),
	fovAngles(getFovAngles()),
	minDotProduct(cosf(fminf(fovAngles.x(), fovAngles.y())))
{}

OmniCameraModel::~OmniCameraModel()
{}

OmniCameraModel OmniCameraModel::makeDefaultModel()
{
	return OmniCameraModel(200.f, 200.f, 200.f, 200.f, 400, 400, 1.f, 
		vec2(200.f, 200.f), 200.f);
}

vec2 OmniCameraModel::camToPixel(const vec3 &p) const
{
	float den = p.z() + p.norm() * e;
	vec2 i;
	i.x() = fx * (p.x() / den) + cx;
	i.y() = fy * (p.y() / den) + cy;
	return i;
}


vec3 OmniCameraModel::pixelToCam(const vec2 &p, float d) const
{
	vec2 pn((p.x() - cx) / fx, (p.y() - cy) / fy);
	float p2 = pn.x()*pn.x() + pn.y()*pn.y();
	float num = e + sqrtf(1.f + (1 - e*e)*p2);
	vec3 dir = (vec3(pn.x(), pn.y(), 1.f) * num / (1.f + p2)) - vec3(0.f, 0.f, e);
	return dir / d;
}


std::vector<std::unique_ptr<CameraModel> >
	OmniCameraModel::createPyramidCameraModels(int nLevels) const
{
	std::vector<std::unique_ptr<CameraModel> > models;
	OmniCameraModel *newModel = new OmniCameraModel(*this);
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
		float newE = static_cast<OmniCameraModel*>(models.back().get())->e;
		vec2 newC = static_cast<OmniCameraModel*>(models.back().get())->c * 0.5f;
		float newR = static_cast<OmniCameraModel*>(models.back().get())->r * 0.5f;

		newModel = new OmniCameraModel(
			newFx, newFy, newCx, newCy, newW, newH, newE, newC, newR);
		models.emplace_back(newModel);
	}
	return models;
}

float OmniCameraModel::getEpipolarParamIncrement(float a, vec3 p0p, vec3 p1p,
	float stepSize) const
{
	vec3 p0 = p0p;//.normalized();
	vec3 p1 = p1p;//.normalized();
	float na = (a*p0 + (1.f-a)*p1).norm();
	float npa = (a*(p0.dot(p0)-p0.dot(p1)) + (1-a)*(p1.dot(p0)-p1.dot(p1)))/na;
	
	vec2 J; //The Jacobian matrix of the line wrt a.
	float den = a*p0.z() + (1-a)*p1.z() + e*na;
	den *= den;
	
	J.x() = fx * (e*na*(p0.x()-p1.x()) - e*npa*(a*p0.x() + (1-a)*p1.x()) + p0.x()*p1.z() - p1.x()*p0.z()) / den;
	J.y() = fy * (e*na*(p0.y()-p1.y()) - e*npa*(a*p0.y() + (1-a)*p1.y()) + p0.y()*p1.z() - p1.y()*p0.z()) / den;
	
	return stepSize / J.norm();
}

void OmniCameraModel::traceWorldSpaceLine(vec3 p0, vec3 p1, std::function<void(vec2)> f)
{
	vec2 ia = camToPixel(p1); 
	vec3 pa = p0; 
	float a = 0.f;

	f(ia);
	while (a < 1.f) {
		a += getEpipolarParamIncrement(a, p0, p1);
		pa = a*p0 + (1.f - a)*p1;
		ia = camToPixel(pa);
		f(ia);
	}
}

CameraModelType OmniCameraModel::getType() const
{
	return CameraModelType::OMNI;
}

std::unique_ptr<CameraModel> OmniCameraModel::clone() const
{
	return std::unique_ptr<CameraModel>(new OmniCameraModel(*this));
}

std::unique_ptr<CameraModel> OmniCameraModel::makeProjCamModel() const
{
	return std::make_unique<ProjCameraModel>(fx, fy, cx, cy, w, h);
}

vec2 OmniCameraModel::getFovAngles() const
{
	vec2 fovAngles;
	vec3 maxX = pixelToCam(vec2(w, h / 2));

	float thetaX = atan2f(maxX.z(), maxX.x());
	if (maxX.z() < 0.f) {
		fovAngles.x() = 2.f*thetaX + static_cast<float>(M_PI);
	} else {
		fovAngles.x() = static_cast<float>(M_PI) - (2.f*thetaX);
	}

	vec3 maxY = pixelToCam(vec2(w / 2, h));
	float thetaY = atan2f(maxY.z(), maxY.y());
	if (maxY.z() < 0.f) {
		fovAngles.y() = 2.f*thetaY + static_cast<float>(M_PI);
	} else {
		fovAngles.y() = static_cast<float>(M_PI) - (2.f*thetaY);
	}
	return fovAngles;
}

bool OmniCameraModel::pixelLocValid(const vec2 &p) const
{
	vec2 d = p - c;
	return d.norm() < r && p.x() < w && p.x() > 0.f && p.y() < h && p.y() > 0.f;
}

bool OmniCameraModel::pointInImage(const vec3 &p) const
{
	vec3 pn = p.normalized();
	float dotProd = pn.dot(vec3(0.f, 0.f, 1.f));
	return dotProd > minDotProduct;
}

}
