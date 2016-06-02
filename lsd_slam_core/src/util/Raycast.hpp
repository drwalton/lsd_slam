#pragma once
#include <opencv2/opencv.hpp>
#include "Util/VectorTypes.hpp"
#include "CameraModel/CameraModel.hpp"
#include "Util/Constants.hpp"
#include "DepthEstimation/DepthMapOmniStereo.hpp"

namespace lsd_slam {

///\brief Generate a raycasted image given a triangle mesh and a camera model.
///       Optionally also generate depth maps.
///\param vertices Vertices of the triangle mesh.
///\param indices Indices of the triangles (in same format as GL_TRIANGLES)
///\param colors Vertex colours of the triangles (N.B. only the one with the 
///              index occuring first will be used - try to ensure that the 
///              colours at all vertices of a triangle are the same.
///\param worldToCamera Transform from world space (space of mesh) to camera 
///                     space (image centre).
///\param model The camera model used to generate the images.
///\param raycastDepths If true, the depths of the intersections are recorded
///                     in the depths parameter.
///\param[out] depths Optional 32-bit float output containing depth map. Pixels with 
///                   no intersection have depth -1.f.
cv::Mat raycast(
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors,
	const WorldToCamTransform &worldToCamera,
	const CameraModel &model,
	bool raycastDepths = false,
	cv::Mat &depths = emptyMat);


cv::Vec3b shootRay(
	const Ray &ray,
	const std::vector<vec3> &vertices,
	const std::vector<unsigned int> &indices,
	const std::vector<cv::Vec3b> &colors,
	float &depth);
	

bool intersectRayWithTriangle(
	const vec3 &ta,
	const vec3 &tb,
	const vec3 &tc,
	const Ray &ray,
	float &dist,
	float &baryAB,
	float &baryAC);

mat4 loadCamTransform(const std::string &filename);

}
