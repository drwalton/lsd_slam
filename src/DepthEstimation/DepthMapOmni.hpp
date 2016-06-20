#pragma once

#include "CameraModel/OmniCameraModel.hpp"
#include <array>
#include "Util/Constants.hpp"
#include "Util/settings.hpp"
#include "Util/VectorTypes.hpp"

namespace lsd_slam {

class Frame;

template<typename T, size_t N>
float findSsd(const std::array<T, N> &a, const std::array<T, N> &b)
{
	float ssd = 0.f;
	for (size_t i = 0; i < N; ++i) {
		ssd += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ssd;
}

/*
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
///\param[out] drawSearch Optional output image. The SSDs are plotted on the
///            supplied image. Make sure this image has the same size as the
///            matchFrame, and is in CV_32UC3 format.
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
vec2 &matchPixel,
cv::Mat &drawSearch = emptyMat);
*/

///\brief Function used internally by DepthMap::doOmniStereo.
///\param [out] bestEpDir The direction of the epipolar curve, in image space,
///                       at the lowest-error point found on the curve.
///\param [out] bestMatchPos The direction to the best match in camera space.
float doStereoOmniImpl(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	float &idepth,
	vec2 &bestEpDir, vec3 &bestMatchPos,
	size_t &bestMatchLoopC,
	float &gradAlongLine, float &initLineLen,
	vec3 &bestMatchKeyframe,
	cv::Mat &drawMat = emptyMat,
	bool showMatch = false);

float findVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float gradAlongLine, const vec2 &bestEpDir,
	const Eigen::Vector4f *activeKeyframeGradients,
	float initialTrackedResidual,
	float sampleDist, bool didSubpixel,
	OmniCameraModel *model,
	RunningStats *stats,
	float depth);

bool getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyframeImageData, int width, const OmniCameraModel &oModel,
	float u, float v, std::array<float, 5> &valuesToFind,
	vec2 &epImDir,
	cv::Mat &visIm = emptyMat);

}
