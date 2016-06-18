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

void DepthMapDebugImages::clearResultIm(const float * keyframe, const float * refFrame, const CameraModel * model)
{
	makeDebugCompareIm(results, keyframe, refFrame, model);
}

void DepthMapDebugImages::clearDepthIm(const float * keyframe, const float *refFrame, const CameraModel * model)
{

	makeDebugCompareIm(depths, keyframe, refFrame, model);
}

void DepthMapDebugImages::visualisePixelDisparity(size_t x, size_t y, size_t disparity)
{
	vec3 rgb = 255*hueToRgb(float(disparity % 4)  / 5.f);
	pixelDisparity.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::clearPixelDisparityIm(const float * keyframe, const float * refFrame, const CameraModel * model)
{
	makeDebugCompareIm(pixelDisparity, keyframe, refFrame, model);
}

cv::Vec3b DepthMapDebugImages::getStereoResultVisColor(float err)
{
	if (err >= 0.f) {
		//Green: successful match
		return cv::Vec3b(0, 255, 0);
	}
	else if (err == DepthMapErrCode::ERR_TOO_BIG) {
		//Red: error too large
		return cv::Vec3b(0, 0, 255);
	}
	else if (err == DepthMapErrCode::WINNER_NOT_CLEAR) {
		//Blue: winner not clear
		return cv::Vec3b(255, 0, 0);
	}
	else if (err == DepthMapErrCode::EPL_NOT_IN_REF_FRAME) {
		//Purple: epipolar line segment did not lie entirely
		//        inside the reference frame.
		return cv::Vec3b(255, 0, 255);
	}
	else if (err == DepthMapErrCode::NAN_MAKING_EPL) {
		//Turquoise: NaN encountered in generating epl.
		return cv::Vec3b(255, 255, 0);
	}
	else if (err == DepthMapErrCode::PADDED_EPL_NOT_IN_REF_FRAME) {
		//Dark Purple: Padded line wouldn't fit in ref. frame.
		return cv::Vec3b(128, 0, 128);
	}
	else if (err == DepthMapErrCode::START_TOO_NEAR_EPIPOLE) {
		//Dark Green: Started too near to epipole
		return cv::Vec3b(0, 128, 0);
	}
	else if (err == DepthMapErrCode::TRACED_LINE_TOO_LONG) {
		//Dark Blue: Line ended up too long when tracing.
		return cv::Vec3b(128, 0, 0);
	}

	//TODO other colours
	else if (err == DepthMapErrCode::SKIP_BAD_TRACKING) {
		//Dark Red: Bad Tracking
		return cv::Vec3b(0, 0, 128);
	}

	//Default colour
	return cv::Vec3b(255, 255, 255);
}

cv::Vec3b DepthMapDebugImages::getIDepthVisColor(float idepth)
{
	vec3 rgb = 255*hueToRgb(idepth  * MIN_DEPTH);
	return cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

}
