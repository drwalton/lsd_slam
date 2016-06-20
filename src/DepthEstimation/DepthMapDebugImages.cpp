#include "DepthMapDebugImages.hpp"
#include "CameraModel/CameraModel.hpp"
#include "util/globalFuncs.hpp"
#include "DepthMapDebugDefines.hpp"

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
			float p = refVal*.75f + kfVal*.25f;
			rptr[c + camModel->w] = cv::Vec3b(p,p,p);
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

void DepthMapDebugImages::clearStereoImages(const float * keyframe, const float * refFrame, const CameraModel *camModel)
{
#if DEBUG_SAVE_SEARCH_RANGE_IMS
	makeDebugCompareIm(searchRanges, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_RESULT_IMS
	makeDebugCompareIm(results, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_IDEPTH_IMS
	makeDebugCompareIm(depths, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_VAR_IMS
	makeDebugCompareIm(vars, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_FRAME_POINT_CLOUDS
	clearFramePtCloud();
#endif
#if DEBUG_SAVE_PIXEL_DISPARITY_IMS
	makeDebugCompareIm(pixelDisparity, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_GRAD_ALONG_LINE_IMS
	makeDebugCompareIm(gradAlongLines, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_GEO_DISP_ERROR_IMS
	makeDebugCompareIm(geoDispErrs, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_PHOTO_DISP_ERROR_IMS
	makeDebugCompareIm(photoDispErrs, keyframe, refFrame, camModel);
#endif
#if DEBUG_SAVE_DISCRETIZATION_ERROR_IMS
	makeDebugCompareIm(discretizationErrs, keyframe, refFrame, camModel);
#endif
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

void DepthMapDebugImages::visualisePixelDisparity(size_t x, size_t y, size_t disparity)
{
	vec3 rgb = 255*hueToRgb(float(disparity % 4)  / 5.f);
	pixelDisparity.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::addVar(size_t x, size_t y, float var)
{
	vec3 rgb = 255*hueToRgb(var  / MAX_VAR);
	vars.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::clearFramePtCloud()
{
	framePtCloud.vertices() = std::vector<vec3>();
	framePtCloud.vertColors() = std::vector<vec3>();
}

void DepthMapDebugImages::addFramePt(const vec3 & point, const float color)
{
	framePtMutex.lock();
	framePtCloud.vertices().push_back(point);
	framePtCloud.vertColors().push_back(vec3(color, color, color));
	framePtMutex.unlock();
}

void DepthMapDebugImages::clearPropagatePtClouds()
{
	prePropagatePointCloud.vertices() = std::vector<vec3>();
	prePropagatePointCloud.vertColors() = std::vector<vec3>();
	postPropagatePointCloud.vertices() = std::vector<vec3>();
	postPropagatePointCloud.vertColors() = std::vector<vec3>();
}

void DepthMapDebugImages::addPrePropagatePt(const vec3 & point, const float color)
{
	propagatePtMutex.lock();
	prePropagatePointCloud.vertices().push_back(point);
	prePropagatePointCloud.vertColors().push_back(vec3(color, color, color));
	propagatePtMutex.unlock();
}

void DepthMapDebugImages::addPostPropagatePt(const vec3 & point, const float color)
{
	propagatePtMutex.lock();
	postPropagatePointCloud.vertices().push_back(point);
	postPropagatePointCloud.vertColors().push_back(vec3(color, color, color));
	propagatePtMutex.unlock();
}

void DepthMapDebugImages::saveStereoIms(int kfID, int refID)
{
#if DEBUG_SAVE_SEARCH_RANGE_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "RangeIms/RangeKF" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), searchRanges);
	}
#endif
#if DEBUG_SAVE_RESULT_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "ResultIms/ResultKF" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), results);
	}
#endif
#if DEBUG_SAVE_IDEPTH_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "DepthIms/DepthKF" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), depths);
	}
#endif
#if DEBUG_SAVE_VAR_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "VarIms/VarKF" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), vars);
	}
#endif
#if DEBUG_SAVE_FRAME_POINT_CLOUDS
	{
		std::stringstream ss;
		ss << resourcesDir() << "FramePtClouds/CloudKF" << kfID <<
			"_f" << refID << ".ply";
		framePtCloud.saveFile(ss.str());
	}
#endif
#if DEBUG_SAVE_PIXEL_DISPARITY_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "PixelDispIms/PixelDispKF" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), pixelDisparity);
	}
#endif

#if DEBUG_SAVE_GRAD_ALONG_LINE_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "GradAlongLineIms/GradAlongLine" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), gradAlongLines);
	}
#endif
#if DEBUG_SAVE_GEO_DISP_ERROR_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "GeoDispErrIms/GeoDispErr" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), geoDispErrs);
	}
#endif
#if DEBUG_SAVE_PHOTO_DISP_ERROR_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "PhotoDispErrIms/PhotoDispErr" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), photoDispErrs);
	}
#endif
#if DEBUG_SAVE_DISCRETIZATION_ERROR_IMS
	{
		std::stringstream ss;
		ss << resourcesDir() << "DiscretizeErrIms/DiscretizeErr" << kfID <<
			"_f" << refID << ".png";
		cv::imwrite(ss.str(), discretizationErrs);
	}
#endif
}

void DepthMapDebugImages::addGradAlongLine(size_t x, size_t y, float gradAlongLine)
{
	vec3 rgb = 255*hueToRgb(gradAlongLine  / 10'000.f);
	gradAlongLines.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::addGeoDispError(size_t x, size_t y, float geoDispError)
{
	vec3 rgb = 255*hueToRgb(geoDispError  / 1.f);
	geoDispErrs.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::addPhotoDispError(size_t x, size_t y, float photoDispError)
{
	vec3 rgb = 255*hueToRgb(photoDispError  / 1.f);
	photoDispErrs.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

void DepthMapDebugImages::addDiscretizationError(size_t x, size_t y, float discretizationErr)
{
	vec3 rgb = 255*hueToRgb(discretizationErr  / 1.f);
	discretizationErrs.at<cv::Vec3b>(y,x) = cv::Vec3b(rgb[2], rgb[1], rgb[0]);
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
	vec3 rgb = 255*hueToRgb(5.f / idepth);
	return cv::Vec3b(rgb[2], rgb[1], rgb[0]);
}

}
