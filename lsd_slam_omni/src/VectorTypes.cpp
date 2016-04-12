#include "VectorTypes.hpp"
#include <Eigen/Dense>

namespace lsd_slam {

RigidTransform::RigidTransform()
	:rotation(mat3::Identity()), translation(vec3::Zero())
	{}

RigidTransform::RigidTransform(const mat3 &r, const vec3 &t) 
	:rotation(r), translation(t)
	{}

RigidTransform::RigidTransform(const mat4 &m)
	:rotation(m.block<3, 3>(0, 0)), translation(m.block<3, 1>(0, 3))
	{}

RigidTransform  RigidTransform::inverse() const
{
	return RigidTransform(this->operator mat4().inverse());
}

vec3 RigidTransform::operator *(const vec3 &rhs) const 
{
	return rotation * rhs + translation;
}

RigidTransform::operator mat4() const 
{
	mat4 m = mat4::Zero();
	m.block<3, 3>(0, 0) = rotation;
	m.block<3, 1>(0, 3) = translation;
	m(3, 3) = 1.f;
	return m;
}

vec4 RigidTransform::operator *(const vec4 &rhs) const 
{
	return this->operator mat4() * rhs;
}

}
