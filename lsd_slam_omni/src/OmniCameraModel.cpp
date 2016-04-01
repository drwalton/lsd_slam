#include "OmniCameraModel.hpp"

namespace lsd_slam {

OmniCameraModel OmniCameraModel::makeDefaultModel()
{
	OmniCameraModel m;
	m.fx = m.fy = 200.f;
	m.cx = m.cy = 200.f;
	m.e = 1.f;
	return m;
}

vec2 OmniCameraModel::worldToPixel(vec3 p) const
{
	float den = p.z() + p.norm() * e;
	vec2 i;
	i.x() = fx * (p.x() / den) + cx;
	i.y() = fy * (p.y() / den) + cy;
	return i;
}


vec3 OmniCameraModel::pixelToWorld(vec2 p, float d) const
{
	vec2 pn((p.x() - cx) / fx, (p.y() - cy) / fy);
	float p2 = pn.x()*pn.x() + pn.y()*pn.y();
	float num = e + sqrtf(1.f + (1 - e*e)*p2);
	vec3 dir = (vec3(pn.x(), pn.y(), 1.f) * num / (1.f + p2)) - vec3(0.f, 0.f, e);
	return dir / d;
}

float OmniCameraModel::getEpipolarParamIncrement(float a, vec3 p0, vec3 p1) const
{
	float na = (a*p0 + (1.f-a)*p1).norm();
	float npa = (a*(p0.dot(p0)-p0.dot(p1)) + (1-a)*(p1.dot(p0)-p1.dot(p1)))/na;
	
	vec2 J; //The Jacobian matrix of the line wrt a.
	float den = a*p0.z() + (1-a)*p1.z() + e*na;
	den *= den;
	
	J.x() = fx * (e*na*(p0.x()-p1.x()) - e*npa*(a*p0.x() + (1-a)*p1.x()) + p0.x()*p1.z() - p1.x()*p0.z()) / den;
	J.y() = fy * (e*na*(p0.y()-p1.y()) - e*npa*(a*p0.y() + (1-a)*p1.y()) + p0.y()*p1.z() - p1.y()*p0.z()) / den;
	
	return 1.f / J.norm();
}

void OmniCameraModel::traceWorldSpaceLine(vec3 p0, vec3 p1, std::function<void(vec2)> f)
{
	vec2 ia = worldToPixel(p1); 
	vec3 pa = p0; 
	float a = 0.f;

	f(ia);
	while (a < 1.f) {
		a += getEpipolarParamIncrement(a, p0, p1);
		pa = a*p0 + (1.f - a)*p1;
		ia = worldToPixel(pa);
		f(ia);
	}
}

}
