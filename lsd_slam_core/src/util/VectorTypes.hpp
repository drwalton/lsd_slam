#pragma once

#include <Eigen/Core>
#include <ostream>
#include <opencv2/core.hpp>

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
	RigidTransform operator *(const RigidTransform &rhs) const;
};

std::ostream &operator <<(std::ostream &stream, const RigidTransform &rhs);

typedef RigidTransform WorldToCamTransform;

cv::Point vec2Point(const vec2 &v);
cv::Vec3b vec3ToColor(const vec3 &v);

}
