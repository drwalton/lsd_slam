#include "Raycast.hpp"
#include <Eigen/Dense>
#include <thread>
#include <fstream>

namespace lsd_slam {

const float EPS = 10e-6f;
cv::Vec3b BACKGROUND_COLOR = cv::Vec3b(0,0,0);
size_t nHits = 0;

cv::Mat raycast(
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors,
	const mat4 &worldToCamera,
	const CameraModel &model,
	cv::Size size)
{
	vec3 origin = -worldToCamera.block<3,1>(0,3);
	mat3 invRot = worldToCamera.block<3,3>(0,0).inverse();
	cv::Mat image(size, CV_8UC3);
	std::vector<std::thread> threads(image.rows);
	for(size_t r = 0; r < image.rows; ++r) {
		threads[r] = std::thread([
		r, &image, &vertices, &indices, &colors,
		&worldToCamera, &model, &origin, &invRot]() {
			cv::Vec3b *rPtr = image.ptr<cv::Vec3b>(r);
			for(size_t c = 0; c < image.cols; ++c) {
    			Ray ray;
    			ray.origin = origin;
    			ray.dir = invRot * model.pixelToCam(vec2(r,c));
				*rPtr = shootRay(ray, vertices, indices, colors);
				++rPtr;
			}
		});
	}
	
	for(std::thread &t : threads) {
		t.join();
	}
	std::cout << nHits << " hits" << std::endl;
	return image;
}

cv::Vec3b shootRay(
	const Ray &ray,
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors)
{
	float closestIntersectionDist = FLT_MAX;
	cv::Vec3b color = BACKGROUND_COLOR;
	for(size_t i = 0; i < indices.size(); i += 3) {
		float dist;
		if(intersectRayWithTriangle(
			vertices[indices[i]],
			vertices[indices[i+1]],
			vertices[indices[i+2]],
			ray, dist)) {
				++nHits;
				if(dist < closestIntersectionDist) {
					closestIntersectionDist = dist;
					color = colors[i];
				}
		}
	}
	return color;
}

bool intersectRayWithTriangle(
	const vec3 &ta,
	const vec3 &tb,
	const vec3 &tc,
	const Ray &ray,
	float &dist)
{
	// Get two edges of the triangle.
	vec3 e1 = tb - ta;
	vec3 e2 = tc - ta;
	
	vec3 norm = (ray.dir).cross(e2);
	
	if(norm.norm() < EPS) return false; //Triangle has no area.
	
	float det = e1.dot(norm);
	
	if(det < EPS && det > -EPS) return false; //Ray lies in plane of triangle.
	
	float oneOverDet = 1.0f / det;
	
	vec3 toTriangle = (ray.origin) - ta;
	// Find barycentric co-ordinates u,v.
	float u = toTriangle.dot(norm) * oneOverDet;
	if(u < 0.0f || u > 1.0f) return false; // u out of range.
	
	vec3 acrossTriangle = toTriangle.cross(e1);
	float v = ray.dir.dot(acrossTriangle) * oneOverDet;
	if(v < 0.0f || u + v > 1.0f) return false; // v out of range.
	
	float t = e2.dot(acrossTriangle) * oneOverDet;
	
	if(t < EPS) return false; // intersection lies behind ro.
	
	dist = t;
	
	return true; // u, v in range - intersection.
}

mat4 loadCamTransform(const std::string &filename)
{
	mat4 t;
	std::ifstream i(filename);
	for(int r = 0; r < 4; ++r) {
		for(int c = 0; c < 4; ++c) {
			i >> t(r,c);
		}
	}
	return t;
}

}
