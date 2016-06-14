#pragma once

#include <opencv2/opencv.hpp>
#include "Util/VectorTypes.hpp"

namespace lsd_slam
{

class CameraModel;

struct DepthMapDebugImages
{
	explicit DepthMapDebugImages();

	cv::Mat matches;
	cv::Mat searchRanges;

	bool drawMatchHere(size_t x, size_t y) const;
	size_t drawIntervalX, drawIntervalY;

	void clearMatchesIm(const float *keyframe, const float *refFrame, const CameraModel *camModel);
	void visualiseMatch(vec2 keyframePos, vec2 referenceFramePos, const CameraModel *model);

	void clearSearchRangesIm(const float *keyframe, const float *refFrame, const CameraModel *camModel);
};

}
