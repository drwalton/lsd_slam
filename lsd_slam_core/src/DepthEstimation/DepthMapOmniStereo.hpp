#ifndef DEPTHESTIMATION_DEPTHMAPOMNISTEREO_HPP_INCLUDED
#define DEPTHESTIMATION_DEPTHMAPOMNISTEREO_HPP_INCLUDED

#include "OmniCameraModel.hpp"
#include <array>

namespace lsd_slam {

template<typename T, size_t N>
float findSsd(const std::array<T, N> &a, const std::array<T, N> &b)
{
	float ssd = 0.f;
	for (size_t i = 0; i < N; ++i) {
		ssd += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ssd;
}

///\brief Perform omnidirectional stereo to find a match for the given point.
///\param keyframeToReference The transform mapping from the coordinate frame 
///       of the keyframe to that of the reference frame.
///\param keyframe Pointer to the keyframe image data (in floating-point format,
///       densely packed in row-major order.
///\param width Width of the keyframe image.
///\param x, y Coordinates of the point from the keyframe to find a match for.
///\param minDepth, maxDepth Range of possible depth values for the depth at the
///        given point in the keyframe.
///\param[out] minSsd The minimum SSD matching error encountered along the line.
///\param[out] matchDir The direction to the matched pixel
///\param[out] matchPixel The 2D coordinates of the matched pixel in the reference
///            frame.
///\return true if a match is found, false otherwise.
///\note If this returns false, the values of minSsd, matchDir, matchPixel are invalid!
bool omniStereo(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe, const float *matchFrame,
	int width,
	int x, int y,
	float minDepth, float maxDepth,
	float &minSsd,
	vec3 &matchDir,
	vec2 &matchPixel);

///\brief Return 5 floating-point values, centered around the given point, 
///       along the respective epipolar curve (values should be one pixel apart).
///\note Values advance along the line towards the epipole direction.
std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width,
	vec3 &pointDir);

void findReferenceFrameLineEndpoints(
	vec3 &pStart, vec3 &pEnd,
	int u, int v,
	float depth, float depthStd);

bool findLowestSSDMatch(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe
	);

}

#endif
