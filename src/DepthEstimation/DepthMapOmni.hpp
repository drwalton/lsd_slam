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
	vec2 &bestEpDir, vec3 &bestMatchPos, float &gradAlongLine, float &tracedLineLen,
	vec3 &bestMatchKeyframe,
	cv::Mat &drawMat = emptyMat,
	bool showMatch = false);


float doStereoOmniImpl2(
	const float u, const float v, const vec3 &epDir,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframe, const float* referenceFrameImage,
	const RigidTransform &keyframeToReference,
	RunningStats* stats, const OmniCameraModel &oModel, size_t width,
	float &idepth,
	vec2 &bestEpDir, vec3 &bestMatchPos,
	size_t &bestMatchLoopC,
	float &gradAlongLine, float &tracedLineLen,
	vec3 &bestMatchKeyframe,
	cv::Mat &drawMat = emptyMat,
	bool showMatch = false);

float findInvDepthOmni(const float u, const float v, const vec3 &bestMatchDir,
	OmniCameraModel *model, RigidTransform refToKeyframe, RunningStats *stats);

float findVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float gradAlongLine, const vec2 &bestEpDir,
	const Eigen::Vector4f *activeKeyFrameGradients,
	float initialTrackedResidual,
	float sampleDist, bool didSubpixel,
	OmniCameraModel *model,
	RunningStats *stats,
	float depth);

///\brief Return 5 floating-point values, centered around the given point, 
///       along the respective epipolar curve (values should be one pixel apart).
///\note Values advance along the line towards the epipole direction.
std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width,
	vec3 &pointDir,
	cv::Mat &visIm = emptyMat);

bool getValuesToFindOmni(const vec3 &keyframePointDir, const vec3 &epDir,
	const float *activeKeyFrameImageData, int width, const OmniCameraModel &oModel,
	float u, float v, std::array<float, 5> &valuesToFind,
	cv::Mat &visIm = emptyMat);

float findDepthAndVarOmni(const float u, const float v, const vec3 &bestMatchDir,
	float *resultIDepth, float *resultVar,
	float gradAlongLine, const vec2 &bestEpDir, const Frame *referenceFrame, Frame *activeKeyFrame,
	float sampleDist, bool didSubpixel, float tracedLineLen,
	OmniCameraModel *model,
	RunningStats *stats);

struct Ray {
	vec3 origin;
	vec3 dir;
	std::string to_string() const;
};
std::ostream &operator <<(std::ostream &s, const Ray &r);

struct RayIntersectionResult {
	enum Outcome {
		VALID, BEHIND, PARALLEL
	};
	Outcome valid;
	float distance;
	vec3 position;
	std::string to_string() const;
};
std::ostream &operator <<(std::ostream &s, const RayIntersectionResult &r);

RayIntersectionResult computeRayIntersection(const Ray &r1, const Ray &r2);


enum class MakePaddedLineErrorCode {
	SUCCESS, FAIL_NEAR_EPIPOLE, FAIL_TOO_LONG, FAIL_OUT_OF_IMAGE, FAIL_TOO_SHORT,
	FAIL_TOO_STEEP
};
///\brief Generate a padded epipolar line, of a specified minimum length.
///       Return the line in the keyframe and reference frames' FoRs.
///\return Error code: 
///\param u Pixel location in the keyframe.
///\param v Pixel location in the keyframe.
///\param meanIDepth Expected inverse depth
///\param minIDepth Min inverse depth (i.e. max depth). Defines endpoint of line.
///\param maxIDepth Max inverse depth (i.e. min depth). Defines endpoint of line.
///\param requiredLineLen Required length of the line in the reference frame image,
///       where length is given in pixels.
///\param model Camera model
///\param keyframeToReference Transform from the keyframes FoR to that of the reference.
///\param[out] keyframeLine The line in the FoR of the keyframe.
///\param[out] refframeLine The line in the FoR of the reference frame.
///\note Can fail if chosen pixel location is too close to one of the epipoles.
MakePaddedLineErrorCode makePaddedEpipolarLineOmni(float u, float v,
	float meanIDepth, float minIDepth, float maxIDepth,
	float requiredLineLen,
	const OmniCameraModel &model,
	const RigidTransform &keyframeToReference,
	vec3 *keyframeDir,
	LineSeg3d *keyframeLine,
	LineSeg3d *refframeLine,
	OmniEpLine2d *refframeLinePix);

}
