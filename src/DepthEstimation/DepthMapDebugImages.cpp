#include "DepthMapDebugImages.hpp"
#include "CameraModel/CameraModel.hpp"
#include "util/globalFuncs.hpp"

namespace lsd_slam
{

void makeDebugCompareIm(cv::Mat &i, const float * keyframe, const float * refFrame, const CameraModel *camModel)
{
	if (i.cols != camModel->w * 2 || i.rows != camModel->h) {
		i = cv::Mat(camModel->h, camModel->w * 2, CV_8UC3);
	}
	for (size_t r = 0; r < camModel->h; ++r) {
		cv::Vec3b *rptr = i.ptr<cv::Vec3b>(r);
		for (size_t c = 0; c < camModel->w; ++c) {
			uchar kfVal = static_cast<uchar>(keyframe[r*camModel->w + c]);
			uchar refVal = static_cast<uchar>(refFrame[r*camModel->w + c]);
			rptr[c] = cv::Vec3b(kfVal, kfVal, kfVal);
			rptr[c + camModel->w] = cv::Vec3b(refVal, refVal, refVal);
		}
	}
}

DepthMapDebugImages::DepthMapDebugImages()
	:drawIntervalX(10), drawIntervalY(10)
{}

bool DepthMapDebugImages::drawMatchHere(size_t x, size_t y) const
{
	return ((x % drawIntervalX == 0) && (y % drawIntervalY == 0));
}

void DepthMapDebugImages::clearMatchesIm(const float * keyframe, const float * refFrame, const CameraModel *camModel)
{
	makeDebugCompareIm(matches, keyframe, refFrame, camModel);
}

void DepthMapDebugImages::visualiseMatch(vec2 keyframePos, vec2 referenceFramePos, const CameraModel *model)
{
	const int matchDisplayInvChance = 100;
	if (rand() % matchDisplayInvChance == 0) {
		//Show match.
		float hue = float(rand() % 256) / 256.f;
		vec3 rgb = hueToRgb(hue);
		cv::line(matches,
			cv::Point(int(keyframePos.x()), int(keyframePos.y())),
			cv::Point(int(referenceFramePos.x()) + model->w, int(referenceFramePos.y())),
			cv::Vec3b(uchar(255.f*rgb[0]), uchar(255.f*rgb[1]), uchar(255.f*rgb[2])));
	}
}

void DepthMapDebugImages::clearSearchRangesIm(const float * keyframe, const float * refFrame, const CameraModel *camModel)
{
	makeDebugCompareIm(searchRanges, keyframe, refFrame, camModel);
}

}
