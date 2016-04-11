#include "VectorTypes.hpp"
#include <Eigen/Dense>

namespace lsd_slam {

WorldToCamTransform::WorldToCamTransform()
	:rotation(mat3::Identity()), translation(vec3::Zero())
	{}

WorldToCamTransform::WorldToCamTransform(const mat3 &r, const vec3 &t) 
	:rotation(r), translation(t)
	{}

WorldToCamTransform::WorldToCamTransform(const mat4 &m)
	:rotation(m.block<3, 3>(0, 0)), translation(m.block<3, 1>(0, 3))
	{}

WorldToCamTransform  WorldToCamTransform::inverse() const
{
	return WorldToCamTransform(this->operator mat4().inverse());
}

vec3 WorldToCamTransform::operator *(const vec3 &rhs) const 
{
	return rotation * rhs + translation;
}

WorldToCamTransform::operator mat4() const 
{
	mat4 m = mat4::Zero();
	m.block<3, 3>(0, 0) = rotation;
	m.block<3, 1>(0, 3) = translation;
	m(3, 3) = 1.f;
	return m;
}

vec4 WorldToCamTransform::operator *(const vec4 &rhs) const 
{
	return this->operator mat4() * rhs;
}

}
