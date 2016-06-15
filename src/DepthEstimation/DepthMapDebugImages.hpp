#pragma once

#include <opencv2/opencv.hpp>
#include "Util/VectorTypes.hpp"

namespace lsd_slam
{

class CameraModel;

namespace DepthMapErrCode
{
const float EPL_NOT_IN_REF_FRAME = -1;
const float WINNER_NOT_CLEAR = -2;
const float ERR_TOO_BIG = -3;
const float NAN_MAKING_EPL = -4;
const float PADDED_EPL_NOT_IN_REF_FRAME = -5;
}

struct DepthMapDebugImages
{
	explicit DepthMapDebugImages();

	//Shows the keyframe and ref frame side by side, and lines joining matched pixels.
	cv::Mat matches; 

	//Shows the keyframe and ref frame side by side. The keyframe has circles on matched pixels.
	// The reference frame has search lines, and matches (if found) as turquoise.
	cv::Mat searchRanges;

	//Shows the outcome of stereo on each pixel of the keyframe.
	// Each pixel is assigned a colour indicating success, or alternatively the reason
	// for failure.
	cv::Mat results;

	bool drawMatchHere(size_t x, size_t y) const;
	size_t drawIntervalX, drawIntervalY;

	void clearMatchesIm(const float *keyframe, const float *refFrame, const CameraModel *camModel);
	void visualiseMatch(vec2 keyframePos, vec2 referenceFramePos, const CameraModel *model);

	void clearSearchRangesIm(const float *keyframe, const float *refFrame, const CameraModel *camModel);
	void clearResultIm(const float *keyframe, const float *refFrame, const CameraModel *model);

	static cv::Vec3b getStereoResultVisColor(float err);
};

}
