#ifndef LSD_SLAM_RAYCAST_HPP_INCLUDED
#define LSD_SLAM_RAYCAST_HPP_INCLUDED

#include <opencv2/opencv.hpp>
#include "VectorTypes.hpp"
#include "CameraModel.hpp"

namespace lsd_slam {

struct Ray {
	vec3 origin;
	vec3 dir;
};

cv::Mat raycast(
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors,
	const WorldToCamTransform &worldToCamera,
	const CameraModel &model,
	cv::Size size);


cv::Vec3b shootRay(
	const Ray &ray,
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors);
	

bool intersectRayWithTriangle(
	const vec3 &ta,
	const vec3 &tb,
	const vec3 &tc,
	const Ray &ray,
	float &dist);

mat4 loadCamTransform(const std::string &filename);

}

#endif


