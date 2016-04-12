#ifndef LSD_OMNI_VECTORTYPES_HPP_INCLUDED
#define LSD_OMNI_VECTORTYPES_HPP_INCLUDED

#include <Eigen/Core>

namespace lsd_slam {

typedef Eigen::Vector2f vec2;
typedef Eigen::Vector3f vec3;
typedef Eigen::Vector4f vec4;

typedef Eigen::Matrix2f mat2;
typedef Eigen::Matrix3f mat3;
typedef Eigen::Matrix4f mat4;

class RigidTransform {
public:
	RigidTransform();

	RigidTransform(const mat3 &r, const vec3 &t);

	RigidTransform(const mat4 &m);

	RigidTransform inverse() const;

	mat3 rotation;
	vec3 translation;
	vec3 operator *(const vec3 &rhs) const;
	operator mat4() const;
	vec4 operator *(const vec4 &rhs) const;
};

typedef RigidTransform WorldToCamTransform;

}

#endif
