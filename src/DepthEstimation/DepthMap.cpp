/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "DepthEstimation/DepthMap.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "util/settings.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.hpp"
#include "DataStructures/Frame.hpp"
#include "util/globalFuncs.hpp"
#include "IOWrapper/ImageDisplay.hpp"
#include "GlobalMapping/KeyframeGraph.hpp"
#include "CameraModel/ProjCameraModel.hpp"

#include "DepthMapDebugDefines.hpp"

namespace lsd_slam
{

DepthMapDebugSettings::DepthMapDebugSettings()
	:drawMatchInvChance(1)
{}


DepthMap::DepthMap(const CameraModel &model, DepthMapInitMode mode)
//TODO TMP
	:camModel_(model.clone()), 
	//:camModel_(model.makeProjCamModel()), 
	initMode_(mode),
	debugImages(model.getTypeName()),
	activeKeyframeImageData(nullptr)
{
	int w = camModel_->w, h = camModel_->h;
	activeKeyframe = 0;
	activeKeyframeIsReactivated = false;
	otherDepthMap = new DepthMapPixelHypothesis[w*h];
	currentDepthMap = new DepthMapPixelHypothesis[w*h];

	validityIntegralBuffer = (int*)Eigen::internal::aligned_malloc(w*h*sizeof(int));

	debugImageHypothesisHandling = cv::Mat(h,w, CV_8UC3);
	debugImageHypothesisPropagation = cv::Mat(h,w, CV_8UC3);
	debugImageStereoLines = cv::Mat(h,w, CV_8UC3);
	debugImageDepth = cv::Mat(h,w, CV_8UC3);

	reset();

	msUpdate =  msCreate =  msFinalize = 0;
	msObserve =  msRegularize =  msPropagate =  msFillHoles =  msSetDepth = 0;
	gettimeofday(&lastHzUpdate, NULL);
	nUpdate = nCreate = nFinalize = 0;
	nObserve = nRegularize = nPropagate = nFillHoles = nSetDepth = 0;
	nAvgUpdate = nAvgCreate = nAvgFinalize = 0;
	nAvgObserve = nAvgRegularize = nAvgPropagate = nAvgFillHoles = nAvgSetDepth = 0;
}

DepthMap::~DepthMap()
{
	if(activeKeyframe != 0)
		activeKeyframelock.unlock();

	debugImageHypothesisHandling.release();
	debugImageHypothesisPropagation.release();
	debugImageStereoLines.release();
	debugImageDepth.release();

	delete[] otherDepthMap;
	delete[] currentDepthMap;

	Eigen::internal::aligned_free((void*)validityIntegralBuffer);
}


void DepthMap::reset()
{
	for(DepthMapPixelHypothesis* pt = otherDepthMap+camModel_->w*camModel_->h-1; pt >= otherDepthMap; pt--)
		pt->isValid = false;
	for(DepthMapPixelHypothesis* pt = currentDepthMap+camModel_->w*camModel_->h-1; pt >= currentDepthMap; pt--)
		pt->isValid = false;
}


void DepthMap::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
	const float* keyframeMaxGradBuf = activeKeyframe->maxGradients(0);

	int successes = 0;

	for(int y=yMin;y<yMax; y++)
		for(size_t x=3;x<camModel_->w-3;x++)
		{
			if (!camModel_->pixelLocValid(vec2(x, y))) {
				continue;
			}
			size_t idx = x+y*camModel_->w;
			DepthMapPixelHypothesis* target = currentDepthMap+idx;
			bool hasHypothesis = target->isValid;

			// ======== 1. check absolute grad =========
			if(hasHypothesis && keyframeMaxGradBuf[idx] < MIN_ABS_GRAD_DECREASE)
			{
				//Avoid using: has a hypothesis, and grad is too low (will increase error, 
				// rather than refining depth).
				target->isValid = false;
				continue;
			}

			if (keyframeMaxGradBuf[idx] < MIN_ABS_GRAD_CREATE) {
				//Avoid using - gradient too low to create a new hypothesis here.
				continue;
			}
			if (target->blacklisted < MIN_BLACKLIST) {
				//Avoid using - this pixel has been blacklisted after failing stereo
				// too often.
#if DEBUG_SAVE_RESULT_IMS
				debugImages.results.at<cv::Vec3b>(y, int(x)) = cv::Vec3b(0,0,0);
#endif
				continue;
			}

			//Set all pixels used as input to a stereo procedure to white initially.
#if DEBUG_SAVE_RESULT_IMS
			debugImages.results.at<cv::Vec3b>(y, int(x)) = cv::Vec3b(255,255,255);
#endif

			bool success;
			if(!hasHypothesis)
				success = observeDepthCreate(int(x), y, idx, stats);
			else
				success = observeDepthUpdate(int(x), y, idx, keyframeMaxGradBuf, stats);

			if(success)
				successes++;
		}


}
void DepthMap::observeDepth()
{
	{
		if (activeKeyframeImageData != nullptr) {
			Frame* refFrame = activeKeyframeIsReactivated ?
				newest_referenceFrame : oldest_referenceFrame;
			debugImages.clearStereoImages(activeKeyframeImageData, refFrame->image(), camModel_.get());
		}
	}

	threadReducer.reduce(boost::bind(&DepthMap::observeDepthRow, this, _1, _2, _3), 3, int(camModel_->h)-3, 10);

	if(enablePrintDebugInfo && printObserveStatistics)
	{
		printf("OBSERVE (%d): %d / %d created; %d / %d updated; %d skipped; %d init-blacklisted\n",
				activeKeyframe->id(),
				runningStats.num_observe_created,
				runningStats.num_observe_create_attempted,
				runningStats.num_observe_updated,
				runningStats.num_observe_update_attempted,
				runningStats.num_observe_skip_alreadyGood,
				runningStats.num_observe_blacklisted
		);
	}


	if(enablePrintDebugInfo && printObservePurgeStatistics)
	{
		printf("OBS-PRG (%d): Good: %d; inconsistent: %d; notfound: %d; oob: %d; failed: %d; addSkip: %d;\n",
				activeKeyframe->id(),
				runningStats.num_observe_good,
				runningStats.num_observe_inconsistent,
				runningStats.num_observe_notfound,
				runningStats.num_observe_skip_oob,
				runningStats.num_observe_skip_fail,
				runningStats.num_observe_addSkip
		);
	}
}





bool DepthMap::makeAndCheckEPLProj(const int x, const int y, const Frame* const ref, float* pepx, float* pepy, RunningStats* const stats)
{
	size_t idx = x+y*camModel_->w;

	// ======= make epl ========
	// calculate the plane spanned by the two camera centers and the point (x,y,1)
	// intersect it with the keyframe's image plane (at depth=1)
	float epx = - camModel_->fx * ref->thisToOther_t[0] + ref->thisToOther_t[2]*(x - camModel_->cx);
	float epy = - camModel_->fy * ref->thisToOther_t[1] + ref->thisToOther_t[2]*(y - camModel_->cy);

	if (isnan(epx + epy)) {
		return false;
	}


	// ======== check epl length =========
	float eplLengthSquared = epx*epx+epy*epy;
	if(eplLengthSquared < MIN_EPL_LENGTH_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl++;
		return false;
	}


	// ===== check epl-grad magnitude ======
	float gx = activeKeyframeImageData[idx+1] - activeKeyframeImageData[idx-1];
	float gy = activeKeyframeImageData[idx+camModel_->w] - activeKeyframeImageData[idx-camModel_->w];
	float eplGradSquared = gx * epx + gy * epy;
	eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;	// square and norm with epl-length

	if(eplGradSquared < MIN_EPL_GRAD_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_grad++;
		return false;
	}


	// ===== check epl-grad angle ======
	if(eplGradSquared / (gx*gx+gy*gy) < MIN_EPL_ANGLE_SQUARED)
	{
		if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_angle++;
		return false;
	}


	// ===== DONE - return "normalized" epl =====
	float fac = GRADIENT_SAMPLE_DIST / sqrt(eplLengthSquared);
	*pepx = epx * fac;
	*pepy = epy * fac;

	return true;
}


bool DepthMap::observeDepthCreate(const int &x, const int &y, const int &idx, RunningStats* const &stats)
{
	DepthMapPixelHypothesis* target = currentDepthMap+idx;

	Frame* refFrame = activeKeyframeIsReactivated ? newest_referenceFrame : oldest_referenceFrame;

	if(refFrame->getTrackingParent() == activeKeyframe)
	{
		bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
		if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (camModel_->w >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,0,0); // BLUE for SKIPPED NOT GOOD TRACKED

#if DEBUG_SAVE_RESULT_IMS
			cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(
				DepthMapErrCode::SKIP_BAD_TRACKING);
			debugImages.results.at<cv::Vec3b>(y, x) = color;
#endif
			return false;
		}
	}

	float epx, epy;
	vec3 epDir;
	bool isGood;
	if (camModel_->getType() == CameraModelType::PROJ) {
		isGood = makeAndCheckEPLProj(x, y, refFrame, &epx, &epy, stats);
	} else /* OMNI */ {
		//isGood = makeAndCheckEPLOmni(x, y, refFrame, &epDir, stats);
		isGood = true;
	}
	if (!isGood) {
#if DEBUG_SAVE_RESULT_IMS
		{
			cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(
				DepthMapErrCode::EPL_NOT_IN_REF_FRAME);
			debugImages.results.at<cv::Vec3b>(y, x) = color;
		}
#endif
		return false;
	}

	if(enablePrintDebugInfo) stats->num_observe_create_attempted++;

	float new_u = float(x);
	float new_v = float(y);
	float result_idepth, result_var, result_eplLength;
	float error;
	if (camModel_->getType() == CameraModelType::PROJ) {
		error = doStereoProj(
			new_u, new_v, epx, epy,
			0.0f, 1.0f, 1.0f / MIN_DEPTH,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	} else /* OMNI */ {
		error = doStereoOmni(new_u, new_v, epDir,
			0.0f, 1.0f, 1.0f / MIN_DEPTH,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	}

#if DEBUG_SAVE_RESULT_IMS
	{
		cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(error);
		debugImages.results.at<cv::Vec3b>(y, x) = color;
	}
#endif

	if(error == DepthMapErrCode::ERR_TOO_BIG ||
		error == DepthMapErrCode::WINNER_NOT_CLEAR)
	{
		target->blacklisted--;
		if(enablePrintDebugInfo) stats->num_observe_blacklisted++;
	}

	if (error == DepthMapErrCode::START_TOO_NEAR_EPIPOLE) {
		//Invalidate this point - it started too near an epipole.
		*target = DepthMapPixelHypothesis();
	}

	if (error < 0) {
#if DEBUG_SAVE_RESULT_IMS
		{
			cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(error);
			debugImages.results.at<cv::Vec3b>(y, x) = color;
		}
#endif
		return false;
	}
	if(result_var > MAX_VAR) {
#if DEBUG_SAVE_RESULT_IMS
		{
			cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(
				DepthMapErrCode::ERR_TOO_BIG);
			debugImages.results.at<cv::Vec3b>(y, x) = color;
		}
#endif
		return false;
	}
	
	result_idepth = static_cast<float>(UNZERO(result_idepth));

	// add hypothesis
	*target = DepthMapPixelHypothesis(
			result_idepth,
			result_var,
			VALIDITY_COUNTER_INITIAL_OBSERVE);

	if(plotStereoImages)
		debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,255,255); // white for GOT CREATED

	if(enablePrintDebugInfo) stats->num_observe_created++;
	
	return true;
}

bool DepthMap::observeDepthUpdate(const int &x, const int &y, const int &idx, const float* keyframeMaxGradBuf, RunningStats* const &stats)
{
	DepthMapPixelHypothesis* target = currentDepthMap+idx;
	Frame* refFrame;


	if(!activeKeyframeIsReactivated)
	{
		if((int)target->nextStereoFrameMinID - referenceFrameByID_offset >= (int)referenceFrameByID.size())
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255,0);	// GREEN FOR skip
#if DEBUG_SAVE_RESULT_IMS
	{
		cv::Vec3b color = cv::Vec3b(0,255,255);
		debugImages.results.at<cv::Vec3b>(y, x) = color;
	}
#endif

			if(enablePrintDebugInfo) stats->num_observe_skip_alreadyGood++;

			return false;
		}

		if((int)target->nextStereoFrameMinID - referenceFrameByID_offset < 0)
			refFrame = oldest_referenceFrame;
		else
			refFrame = referenceFrameByID[(int)target->nextStereoFrameMinID - referenceFrameByID_offset];
	}
	else
		refFrame = newest_referenceFrame;


	if(refFrame->getTrackingParent() == activeKeyframe)
	{
		bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
		if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (camModel_->w >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,0,0); // BLUE for SKIPPED NOT GOOD TRACKED
			return false;
		}
	}

	float epx, epy;
	vec3 epDir;
	bool isGood;
	if (camModel_->getType() == CameraModelType::PROJ) {
		isGood = makeAndCheckEPLProj(x, y, refFrame, &epx, &epy, stats);
	} else /* OMNI */ {
		//isGood = makeAndCheckEPLOmni(x, y, refFrame, &epDir, stats);
		isGood = true;
	}
	if(!isGood) return false;

	// which exact point to track, and where from.
	float sv = sqrt(target->idepth_var_smoothed);
	float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC;
	float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC;
	if(min_idepth < 0) min_idepth = 0;
	if(max_idepth > 1/MIN_DEPTH) max_idepth = 1/MIN_DEPTH;

	float result_idepth, result_var, result_eplLength;
	float error, diff;

	if (max_idepth > min_idepth) {

		stats->num_observe_update_attempted++;

		if (camModel_->getType() == CameraModelType::PROJ) {
			error = doStereoProj(
				float(x), float(y), epx, epy,
				min_idepth, target->idepth_smoothed, max_idepth,
				refFrame, refFrame->image(0),
				result_idepth, result_var, result_eplLength, stats);
		}
		else /* OMNI */ {
			error = doStereoOmni(
				float(x), float(y), epDir,
				min_idepth, target->idepth_smoothed, max_idepth,
				refFrame, refFrame->image(0),
				result_idepth, result_var, result_eplLength, stats);
			if (error == DepthMapErrCode::START_TOO_NEAR_EPIPOLE) {
				//If this is the first new frame, invalidate.
				if (refFrame->id() == 2) {
					*target = DepthMapPixelHypothesis();
				}
			}
		}


		diff = result_idepth - target->idepth_smoothed;
#if DEBUG_SAVE_EPL_LENGTH_IMS
		{
			vec3 rgb = 255*hueToRgb(result_eplLength  / 3.f);
			cv::Vec3b color(rgb[2], rgb[1], rgb[0]);
			debugImages.eplLengths.at<cv::Vec3b>(y, x) = color;
		}
#endif
	}
	else {
		error = -2;
		diff = 0.f;
	}

#if DEBUG_SAVE_RESULT_IMS
	{
		cv::Vec3b color = DepthMapDebugImages::getStereoResultVisColor(error);
		debugImages.results.at<cv::Vec3b>(y, x) = color;
	}
#endif


	// if oob: (really out of bounds)
	if(error == -1)
	{
		// do nothing, pixel got oob, but is still in bounds in original. I will want to try again.
		if(enablePrintDebugInfo) stats->num_observe_skip_oob++;

		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,255);	// RED FOR OOB
		return false;
	}

	// if just not good for stereo (e.g. some inf / nan occured; has inconsistent minimum; ..)
	else if(error == -2)
	{
		if(enablePrintDebugInfo) stats->num_observe_skip_fail++;

		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,0,255);	// PURPLE FOR NON-GOOD


		target->validity_counter -= VALIDITY_COUNTER_DEC;
		if(target->validity_counter < 0) target->validity_counter = 0;


		target->nextStereoFrameMinID = 0;

		target->idepth_var *= FAIL_VAR_INC_FAC;
		if(target->idepth_var > MAX_VAR)
		{
			target->isValid = false;
			target->blacklisted--;
		}
		return false;
	}

	// if not found (error too high)
	else if(error == -3)
	{
		if(enablePrintDebugInfo) stats->num_observe_notfound++;
		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,0);	// BLACK FOR big not-found


		return false;
	}

	else if(error == -4)
	{
		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,0);	// BLACK FOR big arithmetic error

		return false;
	}

	// if inconsistent
	else if(DIFF_FAC_OBSERVE*diff*diff > result_var + target->idepth_var_smoothed)
	{
		if(enablePrintDebugInfo) stats->num_observe_inconsistent++;
		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,255,0);	// Turkoise FOR big inconsistent

		target->idepth_var *= FAIL_VAR_INC_FAC;
		if(target->idepth_var > MAX_VAR) target->isValid = false;

		return false;
	}


	else
	{
#if DEBUG_SAVE_IDEPTH_IMS
		debugImages.depths.at<cv::Vec3b>(cv::Point(x, y)) =
			DepthMapDebugImages::getIDepthVisColor(result_idepth);
#endif
#if DEBUG_SAVE_VAR_IMS
		debugImages.addVar(x, y, result_var);
#endif
#if DEBUG_SAVE_FRAME_STEREO_POINT_CLOUDS
		{
			float depth = 1.f / result_idepth;
			if (std::isfinite(depth)) {
				vec3 pt = camModel_->pixelToCam(vec2(x, y), depth);
				debugImages.addFramePt(pt, activeKeyframeImageData[y*camModel_->w + x]);
				//debugImages.addFramePt(pt, debugImages.pixelDisparity.at<cv::Vec3b>(y,x)[0]);
			}
		}
#endif

		// one more successful observation!
		if(enablePrintDebugInfo) stats->num_observe_good++;

		if(enablePrintDebugInfo) stats->num_observe_updated++;


		// do textbook ekf update:
		// increase var by a little (prediction-uncertainty)
		float id_var = target->idepth_var*SUCC_VAR_INC_FAC;

		// update var with observation
		float w = result_var / (result_var + id_var);
		float new_idepth = (1-w)*result_idepth + w*target->idepth;
		target->idepth = static_cast<float>(UNZERO(new_idepth));

		// variance can only decrease from observation; never increase.
		id_var = id_var * w;
		if(id_var < target->idepth_var)
			target->idepth_var = id_var;

		// increase validity!
		target->validity_counter += VALIDITY_COUNTER_INC;
		float absGrad = keyframeMaxGradBuf[idx];
		if(target->validity_counter > VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f)
			target->validity_counter = static_cast<int>(
				VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f);

		// increase Skip!
		//TODO look into adapting this for omni.
		//if(result_eplLength < MIN_EPL_LENGTH_CROP && camModel_->getType() != CameraModelType::OMNI)
		if(false)
		{
			float inc = activeKeyframe->numFramesTrackedOnThis / 
				(float)(activeKeyframe->numMappedOnThis+5);
			if(inc < 3) inc = 3;

			inc +=  ((int)(result_eplLength*10000)%2);

			if(enablePrintDebugInfo) stats->num_observe_addSkip++;

			if(result_eplLength < 0.5*MIN_EPL_LENGTH_CROP)
				inc *= 3;

			if (camModel_->getType() == CameraModelType::OMNI) {
				//inc /= 2;
			}

			target->nextStereoFrameMinID = refFrame->id() + inc;
		}

		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255,255); // yellow for GOT UPDATED

		return true;
	}
}

void DepthMap::propagateDepth(Frame* new_keyframe)
{
	runningStats.num_prop_removed_out_of_bounds = 0;
	runningStats.num_prop_removed_colorDiff = 0;
	runningStats.num_prop_removed_validity = 0;
	runningStats.num_prop_grad_decreased = 0;
	runningStats.num_prop_color_decreased = 0;
	runningStats.num_prop_attempts = 0;
	runningStats.num_prop_occluded = 0;
	runningStats.num_prop_created = 0;
	runningStats.num_prop_merged = 0;


	if(new_keyframe->getTrackingParent() != activeKeyframe)
	{
		printf("WARNING: propagating depth from frame %d to %d, "
			"which was tracked on a different frame (%d).\n"
			"While this should work, it is not recommended.",
				activeKeyframe->id(), new_keyframe->id(),
				new_keyframe->getTrackingParent()->id());
	}

	// wipe depthmap
	for(DepthMapPixelHypothesis* pt = otherDepthMap+camModel_->w*camModel_->h-1; 
		pt >= otherDepthMap; pt--)
	{
		pt->isValid = false;
		pt->blacklisted = 0;
	}

	// re-usable values.
	SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();
	Eigen::Vector3f trafoInv_t = oldToNew_SE3.translation().cast<float>();
	Eigen::Matrix3f trafoInv_R = oldToNew_SE3.rotationMatrix().matrix().cast<float>();


	const bool* trackingWasGood = 
		new_keyframe->getTrackingParent() == activeKeyframe ? 
		new_keyframe->refPixelWasGoodNoCreate() : 0;


	const float* activeKFImageData = activeKeyframe->image(0);
	const float* newKFMaxGrad = new_keyframe->maxGradients(0);
	const float* newKFImageData = new_keyframe->image(0);





	// go through all pixels of OLD image, propagating forwards.
	for(size_t y=0;y<camModel_->h;y++)
		for(size_t x=0;x<camModel_->w;x++)
		{
			DepthMapPixelHypothesis* source = currentDepthMap + x + y*camModel_->w;

			if(!source->isValid)
				continue;

			if(enablePrintDebugInfo) runningStats.num_prop_attempts++;

			vec3 p = camModel_->pixelToCam(vec2(x, y)) / source->idepth_smoothed;
			vec3 pn = (trafoInv_R * p) + trafoInv_t;
#if DEBUG_SAVE_KEYFRAME_PROPAGATION_CLOUDS
			if (p == p) {
				debugImages.addPrePropagatePt(camModel_->pixelToCam(vec2(x, y)) / source->idepth_smoothed,
					activeKeyframeImageData[x + y*camModel_->w]);
			}
#endif
			//Eigen::Vector3f pn = (trafoInv_R * Eigen::Vector3f(x*fxi + cxi,y*fyi + cyi,1.0f)) / source->idepth_smoothed + trafoInv_t;

			float new_idepth;
			if (camModel_->getType() == CameraModelType::PROJ) {
				new_idepth = 1.0f / pn[2];
			} else /* OMNI */ {
				new_idepth = 1.0f / pn.norm();
			}

			vec2 uv = camModel_->camToPixel(pn);
			float u_new = uv[0];
			float v_new = uv[1];
			//float u_new = pn[0]*new_idepth*fx + cx;
			//float v_new = pn[1]*new_idepth*fy + cy;

			// check if still within image, if not: DROP.
			if (camModel_->getType() == CameraModelType::OMNI && (!camModel_->pixelLocValid(uv))) {
				if (enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
				continue;
			}
			if(!(u_new > 2.1f && v_new > 2.1f && u_new < camModel_->w-3.1f && v_new < camModel_->h-3.1f))
			{
				if(enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
				continue;
			}

			size_t newIDX = (int)(u_new+0.5f) + ((int)(v_new+0.5f))*camModel_->w;
			float destAbsGrad = newKFMaxGrad[newIDX];

			if(trackingWasGood != 0)
			{
				if(!trackingWasGood[(x >> SE3TRACKING_MIN_LEVEL) + (camModel_->w >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)]
				                    || destAbsGrad < MIN_ABS_GRAD_DECREASE)
				{
					if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
					continue;
				}
			}
			else
			{
				float sourceColor = activeKFImageData[x + y*camModel_->w];
				float destColor = getInterpolatedElement(newKFImageData, u_new, v_new, camModel_->w);

				float residual = destColor - sourceColor;


				if(residual*residual / (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*destAbsGrad*destAbsGrad) > 1.0f || destAbsGrad < MIN_ABS_GRAD_DECREASE)
				{
					if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
					continue;
				}
			}

			DepthMapPixelHypothesis* targetBest = otherDepthMap +  newIDX;

			// large idepth = point is near = large increase in variance.
			// small idepth = point is far = small increase in variance.
			float idepth_ratio_4 = new_idepth / source->idepth_smoothed;
			idepth_ratio_4 *= idepth_ratio_4;
			idepth_ratio_4 *= idepth_ratio_4;

			float new_var =idepth_ratio_4*source->idepth_var;


			// check for occlusion
			if(targetBest->isValid)
			{
				// if they occlude one another, one gets removed.
				float diff = targetBest->idepth - new_idepth;
				if(DIFF_FAC_PROP_MERGE*diff*diff >
					new_var +
					targetBest->idepth_var)
				{
					if(new_idepth < targetBest->idepth)
					{
						if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
						continue;
					}
					else
					{
						if(enablePrintDebugInfo) runningStats.num_prop_occluded++;
						targetBest->isValid = false;
					}
				}
			}


			if(!targetBest->isValid)
			{
				if(enablePrintDebugInfo) runningStats.num_prop_created++;

				*targetBest = DepthMapPixelHypothesis(
						new_idepth,
						new_var,
						source->validity_counter);

			}
			else
			{
				if(enablePrintDebugInfo) runningStats.num_prop_merged++;

				// merge idepth ekf-style
				float w = new_var / (targetBest->idepth_var + new_var);
				float merged_new_idepth = w*targetBest->idepth + (1.0f-w)*new_idepth;

				// merge validity
				int merged_validity = source->validity_counter + targetBest->validity_counter;
				if(merged_validity > VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE))
					merged_validity = static_cast<int>(
						VALIDITY_COUNTER_MAX+(VALIDITY_COUNTER_MAX_VARIABLE));

				*targetBest = DepthMapPixelHypothesis(
						merged_new_idepth,
						1.0f/(1.0f/targetBest->idepth_var + 1.0f/new_var),
						merged_validity);

#if DEBUG_SAVE_KEYFRAME_PROPAGATION_CLOUDS
				debugImages.addPostPropagatePt(pn,
					activeKeyframeImageData[x + y*camModel_->w]);
#endif
			}
		}

	// swap!
	std::swap(currentDepthMap, otherDepthMap);


	if(enablePrintDebugInfo && printPropagationStatistics)
	{
		printf("PROPAGATE: %d: %d drop (%d oob, %d color); %d created; %d merged; %d occluded. %d col-dec, %d grad-dec.\n",
				runningStats.num_prop_attempts,
				runningStats.num_prop_removed_validity + runningStats.num_prop_removed_out_of_bounds + runningStats.num_prop_removed_colorDiff,
				runningStats.num_prop_removed_out_of_bounds,
				runningStats.num_prop_removed_colorDiff,
				runningStats.num_prop_created,
				runningStats.num_prop_merged,
				runningStats.num_prop_occluded,
				runningStats.num_prop_color_decreased,
				runningStats.num_prop_grad_decreased);
	}

#if DEBUG_SAVE_KEYFRAME_PROPAGATION_CLOUDS
	std::stringstream ss;
	ss << resourcesDir() << "KeyframePropagationPtClouds/KF" << activeKeyframe->id() 
		<< "PRE.ply";
	debugImages.prePropagatePointCloud.saveFile(ss.str());
	ss.str(std::string());
	ss << resourcesDir() << "KeyframePropagationPtClouds/KF" << activeKeyframe->id() 
		<< "POST.ply";
	debugImages.postPropagatePointCloud.saveFile(ss.str());
	debugImages.clearPropagatePtClouds();
#endif
}


void DepthMap::regularizeDepthMapFillHolesRow(int yMin, int yMax, RunningStats* stats)
{
	// =========== regularize fill holes
	const float* keyframeMaxGradBuf = activeKeyframe->maxGradients(0);

	for(int y=yMin; y<yMax; y++)
	{
		for(size_t x=3;x<camModel_->w-2;x++)
		{
			size_t idx = x+y*camModel_->w;
			DepthMapPixelHypothesis* dest = otherDepthMap + idx;
			if(dest->isValid) continue;
			if(keyframeMaxGradBuf[idx]<MIN_ABS_GRAD_DECREASE) continue;

			int* io = validityIntegralBuffer + idx;
			int val = io[2+2*camModel_->w] - io[2-3*camModel_->w] - io[-3+2*camModel_->w] + io[-3-3*camModel_->w];


			if((dest->blacklisted >= MIN_BLACKLIST && val > VAL_SUM_MIN_FOR_CREATE) || val > VAL_SUM_MIN_FOR_UNBLACKLIST)
			{
				float sumIdepthObs = 0, sumIVarObs = 0;
				int num = 0;

				DepthMapPixelHypothesis* s1max = otherDepthMap + (x-2) + (y+3)*camModel_->w;
				for (DepthMapPixelHypothesis* s1 = otherDepthMap + (x-2) + (y-2)*camModel_->w; s1 < s1max; s1+=camModel_->w)
					for(DepthMapPixelHypothesis* source = s1; source < s1+5; source++)
					{
						if(!source->isValid) continue;

						sumIdepthObs += source->idepth /source->idepth_var;
						sumIVarObs += 1.0f/source->idepth_var;
						num++;
					}

				float idepthObs = sumIdepthObs / sumIVarObs;
				idepthObs = static_cast<float>(UNZERO(idepthObs));

				currentDepthMap[idx] =
					DepthMapPixelHypothesis(
						idepthObs,
						VAR_RANDOM_INIT_INITIAL,
						0);

				if(enablePrintDebugInfo) stats->num_reg_created++;
			}
		}
	}
}


void DepthMap::regularizeDepthMapFillHoles()
{

	buildRegIntegralBuffer();

	runningStats.num_reg_created=0;

	memcpy(otherDepthMap,currentDepthMap,camModel_->w*camModel_->h*sizeof(DepthMapPixelHypothesis));
	threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapFillHolesRow, this, _1, _2, _3), 3, int(camModel_->h)-2, 10);
	if(enablePrintDebugInfo && printFillHolesStatistics)
		printf("FillHoles (discreteDepth): %d created\n",
				runningStats.num_reg_created);
}



void DepthMap::buildRegIntegralBufferRow1(int yMin, int yMax, RunningStats* stats)
{
	// ============ build inegral buffers
	int* validityIntegralBufferPT = validityIntegralBuffer+yMin*camModel_->w;
	DepthMapPixelHypothesis* ptSrc = currentDepthMap+yMin*camModel_->w;
	for(int y=yMin;y<yMax;y++)
	{
		int validityIntegralBufferSUM = 0;

		for(size_t x=0;x<camModel_->w;x++)
		{
			if(ptSrc->isValid)
				validityIntegralBufferSUM += ptSrc->validity_counter;

			*(validityIntegralBufferPT++) = validityIntegralBufferSUM;
			ptSrc++;
		}
	}
}


void DepthMap::buildRegIntegralBuffer()
{
	threadReducer.reduce(boost::bind(&DepthMap::buildRegIntegralBufferRow1, this, _1, _2,_3), 0, camModel_->h);

	int* validityIntegralBufferPT = validityIntegralBuffer;
	int* validityIntegralBufferPT_T = validityIntegralBuffer+camModel_->w;

	size_t wh = camModel_->h*camModel_->w;
	for(int idx=camModel_->w;idx<wh;idx++)
		*(validityIntegralBufferPT_T++) += *(validityIntegralBufferPT++);

}



template<bool removeOcclusions> void DepthMap::regularizeDepthMapRow(int validityTH, int yMin, int yMax, RunningStats* stats)
{
	const int regularize_radius = 2;

	const float regDistVar = REG_DIST_VAR;

	for(int y=yMin;y<yMax;y++)
	{
		for(size_t x=regularize_radius;x<camModel_->w-regularize_radius;x++)
		{
			DepthMapPixelHypothesis* dest = currentDepthMap + x + y*camModel_->w;
			DepthMapPixelHypothesis* destRead = otherDepthMap + x + y*camModel_->w;

			// if isValid need to do better examination and then update.

			if(enablePrintDebugInfo && destRead->blacklisted < MIN_BLACKLIST)
				stats->num_reg_blacklisted++;

			if(!destRead->isValid)
				continue;
			
			float sum=0, val_sum=0, sumIvar=0;//, min_varObs = 1e20;
			int numOccluding = 0, numNotOccluding = 0;

			for(int dx=-regularize_radius; dx<=regularize_radius;dx++)
				for(int dy=-regularize_radius; dy<=regularize_radius;dy++)
				{
					DepthMapPixelHypothesis* source = destRead + dx + dy*camModel_->w;

					if(!source->isValid) continue;
//					stats->num_reg_total++;

					float diff =source->idepth - destRead->idepth;
					if(DIFF_FAC_SMOOTHING*diff*diff > source->idepth_var + destRead->idepth_var)
					{
						if(removeOcclusions)
						{
							if(source->idepth > destRead->idepth)
								numOccluding++;
						}
						continue;
					}

					val_sum += source->validity_counter;

					if(removeOcclusions)
						numNotOccluding++;

					float distFac = (float)(dx*dx+dy*dy)*regDistVar;
					float ivar = 1.0f/(source->idepth_var + distFac);

					sum += source->idepth * ivar;
					sumIvar += ivar;


				}

			if(val_sum < validityTH)
			{
				dest->isValid = false;
				if(enablePrintDebugInfo) stats->num_reg_deleted_secondary++;
				dest->blacklisted--;

				if(enablePrintDebugInfo) stats->num_reg_setBlacklisted++;
				continue;
			}


			if(removeOcclusions)
			{
				if(numOccluding > numNotOccluding)
				{
					dest->isValid = false;
					if(enablePrintDebugInfo) stats->num_reg_deleted_occluded++;

					continue;
				}
			}

			sum = sum / sumIvar;
			sum = static_cast<float>(UNZERO(sum));
			

			// update!
			dest->idepth_smoothed = sum;
			dest->idepth_var_smoothed = 1.0f/sumIvar;

			if(enablePrintDebugInfo) stats->num_reg_smeared++;
		}
	}
}
template void DepthMap::regularizeDepthMapRow<true>(int validityTH, int yMin, int yMax, RunningStats* stats);
template void DepthMap::regularizeDepthMapRow<false>(int validityTH, int yMin, int yMax, RunningStats* stats);


void DepthMap::regularizeDepthMap(bool removeOcclusions, int validityTH)
{
	runningStats.num_reg_smeared=0;
	runningStats.num_reg_total=0;
	runningStats.num_reg_deleted_secondary=0;
	runningStats.num_reg_deleted_occluded=0;
	runningStats.num_reg_blacklisted=0;
	runningStats.num_reg_setBlacklisted=0;

	memcpy(otherDepthMap,currentDepthMap,camModel_->w*camModel_->h*sizeof(DepthMapPixelHypothesis));


	if(removeOcclusions)
		threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<true>, this, validityTH, _1, _2, _3), 2, camModel_->h-2, 10);
	else
		threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<false>, this, validityTH, _1, _2, _3), 2, camModel_->h-2, 10);


	if(enablePrintDebugInfo && printRegularizeStatistics)
		printf("REGULARIZE (%d): %d smeared; %d blacklisted /%d new); %d deleted; %d occluded; %d filled\n",
				activeKeyframe->id(),
				runningStats.num_reg_smeared,
				runningStats.num_reg_blacklisted,
				runningStats.num_reg_setBlacklisted,
				runningStats.num_reg_deleted_secondary,
				runningStats.num_reg_deleted_occluded,
				runningStats.num_reg_created);
}


void DepthMap::initializeRandomly(Frame* new_frame)
{
	CameraModelType camType = camModel_->getType();
	std::unique_ptr<CameraModel> projVersion;
	if (camType == CameraModelType::OMNI) {
		projVersion = camModel_->makeProjCamModel();
	}
	if (initMode_ == DepthMapInitMode::CONSISTENT) {
		srand(1);
	}

	activeKeyframelock = new_frame->getActiveLock();
	activeKeyframe = new_frame;
	activeKeyframeImageData = activeKeyframe->image(0);
	activeKeyframeIsReactivated = false;

	const float* maxGradients = new_frame->maxGradients();

	for(size_t y=1;y<camModel_->h-1;y++)
	{
		for(size_t x=1;x<camModel_->w-1;x++)
		{
			if(maxGradients[x+y*camModel_->w] > MIN_ABS_GRAD_CREATE &&
				camModel_->pixelLocValid(vec2(x,y)))
			{
				float idepth;
				if (initMode_ == DepthMapInitMode::CONSTANT) {
					idepth = 1.f;
				} else {
					idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
				}

				currentDepthMap[x+y*camModel_->w] = DepthMapPixelHypothesis(
						idepth,
						idepth,
						VAR_RANDOM_INIT_INITIAL*0.25,
						VAR_RANDOM_INIT_INITIAL*0.25,
						20);
			}
			else
			{
				currentDepthMap[x+y*camModel_->w].isValid = false;
				currentDepthMap[x+y*camModel_->w].blacklisted = 0;
			}
		}
	}


	activeKeyframe->setDepth(currentDepthMap);
}



void DepthMap::setFromExistingKF(Frame* kf)
{
	assert(kf->hasIDepthBeenSet());

	activeKeyframelock = kf->getActiveLock();
	activeKeyframe = kf;

	const float* idepth = activeKeyframe->idepth_reAct();
	const float* idepthVar = activeKeyframe->idepthVar_reAct();
	const unsigned char* validity = activeKeyframe->validity_reAct();

	DepthMapPixelHypothesis* pt = currentDepthMap;
	activeKeyframe->numMappedOnThis = 0;
	activeKeyframe->numFramesTrackedOnThis = 0;
	activeKeyframeImageData = activeKeyframe->image(0);
	activeKeyframeIsReactivated = true;

	for(size_t y=0;y<camModel_->h;y++)
	{
		for(size_t x=0;x<camModel_->w;x++)
		{
			if(*idepthVar > 0)
			{
				*pt = DepthMapPixelHypothesis(
						*idepth,
						*idepthVar,
						*validity);
			}
			else
			{
				currentDepthMap[x+y*camModel_->w].isValid = false;
				currentDepthMap[x+y*camModel_->w].blacklisted = (*idepthVar == -2) ? MIN_BLACKLIST-1 : 0;
			}

			idepth++;
			idepthVar++;
			validity++;
			pt++;
		}
	}

	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
}


void DepthMap::initializeFromGTDepth(Frame* new_frame)
{
	assert(new_frame->hasIDepthBeenSet());

	activeKeyframelock = new_frame->getActiveLock();
	activeKeyframe = new_frame;
	activeKeyframeImageData = activeKeyframe->image(0);
	activeKeyframeIsReactivated = false;

	const float* idepth = new_frame->idepth();


	float averageGTIDepthSum = 0;
	int averageGTIDepthNum = 0;
	for(size_t y=0;y<camModel_->h;y++)
	{
		for(size_t x=0;x<camModel_->w;x++)
		{
			float idepthValue = idepth[x+y*camModel_->w];
			if(!isnan(idepthValue) && idepthValue > 0)
			{
				averageGTIDepthSum += idepthValue;
				averageGTIDepthNum ++;
			}
		}
	}
	

	for(size_t y=0;y<camModel_->h;y++)
	{
		for(size_t x=0;x<camModel_->w;x++)
		{
			float idepthValue = idepth[x+y*camModel_->w];
			
			if(!isnan(idepthValue) && idepthValue > 0)
			{
				currentDepthMap[x+y*camModel_->w] = DepthMapPixelHypothesis(
						idepthValue,
						idepthValue,
						VAR_GT_INIT_INITIAL,
						VAR_GT_INIT_INITIAL,
						20);
			}
			else
			{
				currentDepthMap[x+y*camModel_->w].isValid = false;
				currentDepthMap[x+y*camModel_->w].blacklisted = 0;
			}
		}
	}


	activeKeyframe->setDepth(currentDepthMap);
}

void DepthMap::resetCounters()
{
	runningStats.num_stereo_comparisons=0;
	runningStats.num_pixelInterpolations=0;
	runningStats.num_stereo_calls = 0;

	runningStats.num_stereo_rescale_oob = 0;
	runningStats.num_stereo_inf_oob = 0;
	runningStats.num_stereo_near_oob = 0;
	runningStats.num_stereo_invalid_unclear_winner = 0;
	runningStats.num_stereo_invalid_atEnd = 0;
	runningStats.num_stereo_invalid_inexistantCrossing = 0;
	runningStats.num_stereo_invalid_twoCrossing = 0;
	runningStats.num_stereo_invalid_noCrossing = 0;
	runningStats.num_stereo_invalid_bigErr = 0;
	runningStats.num_stereo_interpPre = 0;
	runningStats.num_stereo_interpPost = 0;
	runningStats.num_stereo_interpNone = 0;
	runningStats.num_stereo_negative = 0;
	runningStats.num_stereo_successfull = 0;

	runningStats.num_observe_created=0;
	runningStats.num_observe_create_attempted=0;
	runningStats.num_observe_updated=0;
	runningStats.num_observe_update_attempted=0;
	runningStats.num_observe_skipped_small_epl=0;
	runningStats.num_observe_skipped_small_epl_grad=0;
	runningStats.num_observe_skipped_small_epl_angle=0;
	runningStats.num_observe_transit_finalizing=0;
	runningStats.num_observe_transit_idle_oob=0;
	runningStats.num_observe_transit_idle_scale_angle=0;
	runningStats.num_observe_trans_idle_exhausted=0;
	runningStats.num_observe_inconsistent_finalizing=0;
	runningStats.num_observe_inconsistent=0;
	runningStats.num_observe_notfound_finalizing2=0;
	runningStats.num_observe_notfound_finalizing=0;
	runningStats.num_observe_notfound=0;
	runningStats.num_observe_skip_fail=0;
	runningStats.num_observe_skip_oob=0;
	runningStats.num_observe_good=0;
	runningStats.num_observe_good_finalizing=0;
	runningStats.num_observe_state_finalizing=0;
	runningStats.num_observe_state_initializing=0;
	runningStats.num_observe_skip_alreadyGood=0;
	runningStats.num_observe_addSkip=0;

	runningStats.num_observe_blacklisted=0;
}



void DepthMap::updateKeyframe(std::deque< std::shared_ptr<Frame> > referenceFrames)
{
	assert(isValid());

	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);

	oldest_referenceFrame = referenceFrames.front().get();
	newest_referenceFrame = referenceFrames.back().get();
	referenceFrameByID.clear();
	referenceFrameByID_offset = oldest_referenceFrame->id();

	for(std::shared_ptr<Frame> frame : referenceFrames)
	{
		assert(frame->hasTrackingParent());

		if(frame->getTrackingParent() != activeKeyframe)
		{
			printf("WARNING: updating frame %d with %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
					activeKeyframe->id(), frame->id(),
					frame->getTrackingParent()->id());
		}

		Sim3 refToKf;
		if(frame->pose->trackingParent->frameID == activeKeyframe->id())
			refToKf = frame->pose->thisToParent_raw;
		else
			refToKf = activeKeyframe->getScaledCamToWorld().inverse() *  frame->getScaledCamToWorld();

		frame->prepareForStereoWith(activeKeyframe, refToKf, *camModel_, 0);

		while((int)referenceFrameByID.size() + referenceFrameByID_offset <= frame->id())
			referenceFrameByID.push_back(frame.get());
	}

	resetCounters();

	
	if(plotStereoImages)
	{
		cv::Mat keyframeImage(activeKeyframe->height(), activeKeyframe->width(), CV_32F, const_cast<float*>(activeKeyframeImageData));
		keyframeImage.convertTo(debugImageHypothesisHandling, CV_8UC1);
		cv::cvtColor(debugImageHypothesisHandling, debugImageHypothesisHandling, CV_GRAY2RGB);

		cv::Mat oldest_refImage(oldest_referenceFrame->height(), oldest_referenceFrame->width(), CV_32F, const_cast<float*>(oldest_referenceFrame->image(0)));
		cv::Mat newest_refImage(newest_referenceFrame->height(), newest_referenceFrame->width(), CV_32F, const_cast<float*>(newest_referenceFrame->image(0)));
		cv::Mat rfimg = 0.5f*oldest_refImage + 0.5f*newest_refImage;
		rfimg.convertTo(debugImageStereoLines, CV_8UC1);
		cv::cvtColor(debugImageStereoLines, debugImageStereoLines, CV_GRAY2RGB);
	}

	struct timeval tv_start, tv_end;


	gettimeofday(&tv_start, NULL);
	observeDepth();
	gettimeofday(&tv_end, NULL);
	msObserve = 0.9f*msObserve + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nObserve++;

	//if(rand()%10==0)
	{
		gettimeofday(&tv_start, NULL);
		regularizeDepthMapFillHoles();
		gettimeofday(&tv_end, NULL);
		msFillHoles = 0.9f*msFillHoles + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFillHoles++;
	}


	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9f*msRegularize + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;

	
	// Update depth in keyframe
	if(!activeKeyframe->depthHasBeenUpdatedFlag)
	{
		gettimeofday(&tv_start, NULL);
		activeKeyframe->setDepth(currentDepthMap);
		gettimeofday(&tv_end, NULL);
		msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nSetDepth++;
	}


	gettimeofday(&tv_end_all, NULL);
	msUpdate = 0.9f*msUpdate + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nUpdate++;


	activeKeyframe->numMappedOnThis++;
	activeKeyframe->numMappedOnThisTotal++;


	if(plotStereoImages)
	{
		Util::displayImage( "Stereo Key Frame", debugImageHypothesisHandling, false );
		Util::displayImage( "Stereo Reference Frame", debugImageStereoLines, false );
	}



	if(enablePrintDebugInfo && printLineStereoStatistics)
	{
		printf("ST: calls %6d, comp %6d, int %7d; good %6d (%.0f%%), neg %6d (%.0f%%); interp %6d / %6d / %6d\n",
				runningStats.num_stereo_calls,
				runningStats.num_stereo_comparisons,
				runningStats.num_pixelInterpolations,
				runningStats.num_stereo_successfull,
				100*runningStats.num_stereo_successfull / (float) runningStats.num_stereo_calls,
				runningStats.num_stereo_negative,
				100*runningStats.num_stereo_negative / (float) runningStats.num_stereo_successfull,
				runningStats.num_stereo_interpPre,
				runningStats.num_stereo_interpNone,
				runningStats.num_stereo_interpPost);
	}
	if(enablePrintDebugInfo && printLineStereoFails)
	{
		printf("ST-ERR: oob %d (scale %d, inf %d, near %d); err %d (%d uncl; %d end; zro: %d btw, %d no, %d two; %d big)\n",
				runningStats.num_stereo_rescale_oob+
					runningStats.num_stereo_inf_oob+
					runningStats.num_stereo_near_oob,
				runningStats.num_stereo_rescale_oob,
				runningStats.num_stereo_inf_oob,
				runningStats.num_stereo_near_oob,
				runningStats.num_stereo_invalid_unclear_winner+
					runningStats.num_stereo_invalid_atEnd+
					runningStats.num_stereo_invalid_inexistantCrossing+
					runningStats.num_stereo_invalid_noCrossing+
					runningStats.num_stereo_invalid_twoCrossing+
					runningStats.num_stereo_invalid_bigErr,
				runningStats.num_stereo_invalid_unclear_winner,
				runningStats.num_stereo_invalid_atEnd,
				runningStats.num_stereo_invalid_inexistantCrossing,
				runningStats.num_stereo_invalid_noCrossing,
				runningStats.num_stereo_invalid_twoCrossing,
				runningStats.num_stereo_invalid_bigErr);
	}

	debugImages.saveStereoIms(activeKeyframe->id(), referenceFrames[0]->id());
#if DEBUG_SAVE_KEYFRAME_POINT_CLOUDS_EACH_FRAME
	debugImages.saveKeyframeDepthMap(activeKeyframe->idepth(0), activeKeyframe->image(0),
		camModel_.get(),
		activeKeyframe->id(), referenceFrames[0]->id());
#endif 
}

void DepthMap::invalidate()
{
	if(activeKeyframe==0) return;
	activeKeyframe=0;
	activeKeyframelock.unlock();
}

void DepthMap::createKeyframe(Frame* new_keyframe)
{
	assert(isValid());
	assert(new_keyframe != nullptr);
	assert(new_keyframe->hasTrackingParent());

	//boost::shared_lock<boost::shared_mutex> lock = activeKeyframe->getActiveLock();
	boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);


	resetCounters();

	if(plotStereoImages)
	{
		cv::Mat keyframeImage(new_keyframe->height(), new_keyframe->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
		keyframeImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
		cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation, CV_GRAY2RGB);
	}

	SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();

	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);

	//TODO Temporary test: not propagating depth, but instead randomly initialising every time.
	propagateDepth(new_keyframe);
	//initializeRandomly(new_keyframe);

	gettimeofday(&tv_end, NULL);
	msPropagate = 0.9f*msPropagate + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nPropagate++;

	activeKeyframe = new_keyframe;
	activeKeyframelock = activeKeyframe->getActiveLock();
	activeKeyframeImageData = new_keyframe->image(0);
	activeKeyframeIsReactivated = false;



	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9f*msRegularize + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;


	gettimeofday(&tv_start, NULL);
	regularizeDepthMapFillHoles();
	gettimeofday(&tv_end, NULL);
	msFillHoles = 0.9f*msFillHoles + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nFillHoles++;


	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9f*msRegularize + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;




	// make mean inverse depth be one.
	float sumIdepth=0, numIdepth=0;
	for(DepthMapPixelHypothesis* source = currentDepthMap; 
		source < currentDepthMap+camModel_->w*camModel_->h; source++)
	{
		if(!source->isValid)
			continue;
		sumIdepth += source->idepth_smoothed;
		numIdepth++;
	}
	float rescaleFactor = numIdepth / sumIdepth;
	float rescaleFactor2 = rescaleFactor*rescaleFactor;
	for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+camModel_->w*camModel_->h; source++)
	{
		if(!source->isValid)
			continue;
		source->idepth *= rescaleFactor;
		source->idepth_smoothed *= rescaleFactor;
		source->idepth_var *= rescaleFactor2;
		source->idepth_var_smoothed *= rescaleFactor2;
	}
	//TODO this goes haywire in omni mode, always decreasing. 
	//In proj mode, stays close to 1, sometimes higher, sometimes lower.
	//This is symptom, rather than cause: this means average depth is increasing.
	std::cout << "RESCALE: " << rescaleFactor << std::endl;
	activeKeyframe->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
	activeKeyframe->pose->invalidateCache();

	// Update depth in keyframe

	gettimeofday(&tv_start, NULL);
	activeKeyframe->setDepth(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msCreate = 0.9f*msCreate + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nCreate++;



	if(plotStereoImages)
	{
		//Util::displayImage( "KeyframePropagation", debugImageHypothesisPropagation );
	}

}

void DepthMap::addTimingSample()
{
	struct timeval now;
	gettimeofday(&now, NULL);
	float sPassed = ((now.tv_sec-lastHzUpdate.tv_sec) + (now.tv_usec-lastHzUpdate.tv_usec)/1000000.0f);
	if(sPassed > 1.0f)
	{
		nAvgUpdate = 0.8f*nAvgUpdate + 0.2f*(nUpdate / sPassed); nUpdate = 0;
		nAvgCreate = 0.8f*nAvgCreate + 0.2f*(nCreate / sPassed); nCreate = 0;
		nAvgFinalize = 0.8f*nAvgFinalize + 0.2f*(nFinalize / sPassed); nFinalize = 0;
		nAvgObserve = 0.8f*nAvgObserve + 0.2f*(nObserve / sPassed); nObserve = 0;
		nAvgRegularize = 0.8f*nAvgRegularize + 0.2f*(nRegularize / sPassed); nRegularize = 0;
		nAvgPropagate = 0.8f*nAvgPropagate + 0.2f*(nPropagate / sPassed); nPropagate = 0;
		nAvgFillHoles = 0.8f*nAvgFillHoles + 0.2f*(nFillHoles / sPassed); nFillHoles = 0;
		nAvgSetDepth = 0.8f*nAvgSetDepth + 0.2f*(nSetDepth / sPassed); nSetDepth = 0;
		lastHzUpdate = now;

		if(enablePrintDebugInfo && printMappingTiming)
		{
			printf("Upd %3.1fms (%.1fHz); Create %3.1fms (%.1fHz); Final %3.1fms (%.1fHz) // Obs %3.1fms (%.1fHz); Reg %3.1fms (%.1fHz); Prop %3.1fms (%.1fHz); Fill %3.1fms (%.1fHz); Set %3.1fms (%.1fHz)\n",
					msUpdate, nAvgUpdate,
					msCreate, nAvgCreate,
					msFinalize, nAvgFinalize,
					msObserve, nAvgObserve,
					msRegularize, nAvgRegularize,
					msPropagate, nAvgPropagate,
					msFillHoles, nAvgFillHoles,
					msSetDepth, nAvgSetDepth);
		}
	}
}

void DepthMap::finalizeKeyframe()
{
	assert(isValid());


	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);
	struct timeval tv_start, tv_end;

	gettimeofday(&tv_start, NULL);
	regularizeDepthMapFillHoles();
	gettimeofday(&tv_end, NULL);
	msFillHoles = 0.9f*msFillHoles + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nFillHoles++;

	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9f*msRegularize + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;

	gettimeofday(&tv_start, NULL);
	activeKeyframe->setDepth(currentDepthMap);
	activeKeyframe->calculateMeanInformation();
	activeKeyframe->takeReActivationData(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msFinalize = 0.9f*msFinalize + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nFinalize++;
}




int DepthMap::debugPlotDepthMap()
{
	if(activeKeyframe == 0) return 1;

	cv::Mat keyframeImage(activeKeyframe->height(), activeKeyframe->width(), CV_32F, const_cast<float*>(activeKeyframeImageData));
	keyframeImage.convertTo(debugImageDepth, CV_8UC1);
	cv::cvtColor(debugImageDepth, debugImageDepth, CV_GRAY2RGB);
	assert(debugImageDepth.channels() == 3);

	// debug plot & publish sparse version?
	int refID = referenceFrameByID_offset;


	for(size_t y=0;y<camModel_->h;y++)
		for(size_t x=0;x<camModel_->w;x++)
		{
			size_t idx = x + y*camModel_->w;

			if(currentDepthMap[idx].blacklisted < MIN_BLACKLIST && debugDisplay == 2)
				debugImageDepth.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);

			if(!currentDepthMap[idx].isValid) continue;

			cv::Vec3b color = currentDepthMap[idx].getVisualizationColor(refID);
			debugImageDepth.at<cv::Vec3b>(y,x) = color;
		}


	return 1;
}




// find pixel in image (do stereo along epipolar line).
// mat: NEW image
// KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
// trafo: x_old = trafo * x_new; (from new to old image)
// realVal: descriptor in OLD image.
// returns: result_idepth : point depth in new camera's coordinate system
// returns: result_u/v : point's coordinates in new camera's coordinate system
// returns: idepth_var: (approximated) measurement variance of inverse depth of result_point_NEW
// returns error if sucessful; -1 if out of bounds, -2 if not found.
inline float DepthMap::doStereoProj(
	const float u, const float v, const float epxn, const float epyn,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	if(enablePrintDebugInfo) stats->num_stereo_calls++;
#if DEBUG_SAVE_SEARCH_RANGE_IMS
	bool drawThisPixel = debugImages.drawMatchHere(size_t(u), size_t(v));
#endif

	// calculate epipolar line start and end point in old image
	const ProjCameraModel *pm = static_cast<const ProjCameraModel*>(camModel_.get());
	Eigen::Vector3f KinvP = Eigen::Vector3f(pm->fxi()*u+pm->cxi(),pm->fyi()*v+pm->cyi(),1.0f);
	mat3 K_otherToThis_R = pm->K * referenceFrame->otherToThis_R;
	vec3 K_otherToThis_t = pm->K * referenceFrame->otherToThis_t;
	Eigen::Vector3f pInf = K_otherToThis_R * KinvP;
	Eigen::Vector3f pReal = pInf / prior_idepth + K_otherToThis_t;

	float rescaleFactor = pReal[2] * prior_idepth;

	float firstX = u - 2*epxn*rescaleFactor;
	float firstY = v - 2*epyn*rescaleFactor;
	float lastX = u + 2*epxn*rescaleFactor;
	float lastY = v + 2*epyn*rescaleFactor;
	// width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
	if (firstX <= 0 || firstX >= camModel_->w - 2
		|| firstY <= 0 || firstY >= camModel_->h - 2
		|| lastX <= 0 || lastX >= camModel_->w - 2
		|| lastY <= 0 || lastY >= camModel_->h - 2) {
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}

	if(!(rescaleFactor > 0.7f && rescaleFactor < 1.4f))
	{
		if(enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}

	// calculate values to search for
	float realVal_p1 = getInterpolatedElement(activeKeyframeImageData,u + epxn*rescaleFactor, v + epyn*rescaleFactor, camModel_->w);
	float realVal_m1 = getInterpolatedElement(activeKeyframeImageData,u - epxn*rescaleFactor, v - epyn*rescaleFactor, camModel_->w);
	float realVal = getInterpolatedElement(activeKeyframeImageData,u, v, camModel_->w);
	float realVal_m2 = getInterpolatedElement(activeKeyframeImageData,u - 2*epxn*rescaleFactor, v - 2*epyn*rescaleFactor, camModel_->w);
	float realVal_p2 = getInterpolatedElement(activeKeyframeImageData,u + 2*epxn*rescaleFactor, v + 2*epyn*rescaleFactor, camModel_->w);

#if DEBUG_SAVE_SEARCH_RANGE_IMS
	if (drawThisPixel) {
		debugImages.searchRanges.at<cv::Vec3b>(
			int(v + epyn*rescaleFactor), int(u + epxn*rescaleFactor)) = cv::Vec3b(0, 255, 0);
		debugImages.searchRanges.at<cv::Vec3b>(
			int(v - epyn*rescaleFactor), int(u - epxn*rescaleFactor)) = cv::Vec3b(0, 255, 0);
		debugImages.searchRanges.at<cv::Vec3b>(
			int(v), int(u)) = cv::Vec3b(0, 255, 0);
		debugImages.searchRanges.at<cv::Vec3b>(
			int(v - 2*epyn*rescaleFactor), int(u - 2*epxn*rescaleFactor)) = cv::Vec3b(0, 255, 0);
		debugImages.searchRanges.at<cv::Vec3b>(
			int(v + 2*epyn*rescaleFactor), int(u + 2*epxn*rescaleFactor)) = cv::Vec3b(0, 255, 0);
	}
#endif


//	if(referenceFrame->K_otherToThis_t[2] * max_idepth + pInf[2] < 0.01)


	Eigen::Vector3f pClose = pInf + K_otherToThis_t*max_idepth;
	// if the assumed close-point lies behind the
	// image, have to change that.
	if(pClose[2] < 0.001f)
	{
		max_idepth = (0.001f-pInf[2]) / K_otherToThis_t[2];
		pClose = pInf + K_otherToThis_t*max_idepth;
	}
	pClose = pClose / pClose[2]; // pos in new image of point (xy), assuming max_idepth

	Eigen::Vector3f pFar = pInf + K_otherToThis_t*min_idepth;
	// if the assumed far-point lies behind the image or closter than the near-point,
	// we moved past the Point it and should stop.
	if(pFar[2] < 0.001f || max_idepth < min_idepth)
	{
		if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
		return DepthMapErrCode::EPL_NOT_IN_REF_FRAME;
	}
	pFar = pFar / pFar[2]; // pos in new image of point (xy), assuming min_idepth


	// check for nan due to eg division by zero.
	if(isnan((float)(pFar[0]+pClose[0])))
		return DepthMapErrCode::NAN_MAKING_EPL;

	// calculate increments in which we will step through the epipolar line.
	// they are sampleDist (or half sample dist) long
	float incx = pClose[0] - pFar[0];
	float incy = pClose[1] - pFar[1];
	float eplLength = sqrt(incx*incx+incy*incy);
	if (!(eplLength > 0) || std::isinf(eplLength)) {
		return DepthMapErrCode::NAN_MAKING_EPL;
	}

	if(eplLength > MAX_EPL_LENGTH_CROP)
	{
		pClose[0] = pFar[0] + incx*MAX_EPL_LENGTH_CROP/eplLength;
		pClose[1] = pFar[1] + incy*MAX_EPL_LENGTH_CROP/eplLength;
	}

	incx *= GRADIENT_SAMPLE_DIST/eplLength;
	incy *= GRADIENT_SAMPLE_DIST/eplLength;


	// extend one sample_dist to left & right.
	pFar[0] -= incx;
	pFar[1] -= incy;
	pClose[0] += incx;
	pClose[1] += incy;


	// make epl long enough (pad a little bit).
	if(eplLength < MIN_EPL_LENGTH_CROP)
	{
		float pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2.0f;
		pFar[0] -= incx*pad;
		pFar[1] -= incy*pad;

		pClose[0] += incx*pad;
		pClose[1] += incy*pad;
	}

	// if inf point is outside of image: skip pixel.
	if(
			pFar[0] <= SAMPLE_POINT_TO_BORDER ||
			pFar[0] >= camModel_->w-SAMPLE_POINT_TO_BORDER ||
			pFar[1] <= SAMPLE_POINT_TO_BORDER ||
			pFar[1] >= camModel_->h-SAMPLE_POINT_TO_BORDER)
	{
		if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
		return DepthMapErrCode::PADDED_EPL_NOT_IN_REF_FRAME;
	}



	// if near point is outside: move inside, and test length again.
	if(
			pClose[0] <= SAMPLE_POINT_TO_BORDER ||
			pClose[0] >= camModel_->w-SAMPLE_POINT_TO_BORDER ||
			pClose[1] <= SAMPLE_POINT_TO_BORDER ||
			pClose[1] >= camModel_->h-SAMPLE_POINT_TO_BORDER)
	{
		if(pClose[0] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}
		else if(pClose[0] >= camModel_->w-SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (camModel_->w-SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		if(pClose[1] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}
		else if(pClose[1] >= camModel_->h-SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (camModel_->h-SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		// get new epl length
		float fincx = pClose[0] - pFar[0];
		float fincy = pClose[1] - pFar[1];
		float newEplLength = sqrt(fincx*fincx+fincy*fincy);

		// test again
		if(
				pClose[0] <= SAMPLE_POINT_TO_BORDER ||
				pClose[0] >= camModel_->w-SAMPLE_POINT_TO_BORDER ||
				pClose[1] <= SAMPLE_POINT_TO_BORDER ||
				pClose[1] >= camModel_->h-SAMPLE_POINT_TO_BORDER ||
				newEplLength < 8.0f
				)
		{
			if(enablePrintDebugInfo) stats->num_stereo_near_oob++;
			return DepthMapErrCode::PADDED_EPL_NOT_IN_REF_FRAME;
		}


	}


	// from here on:
	// - pInf: search start-point
	// - p0: search end-point
	// - incx, incy: search steps in pixel
	// - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.


	float cpx = pFar[0];
	float cpy =  pFar[1];

	float val_cp_m2 = getInterpolatedElement(referenceFrameImage,cpx-2.0f*incx, cpy-2.0f*incy, int(camModel_->w));
	float val_cp_m1 = getInterpolatedElement(referenceFrameImage,cpx-incx, cpy-incy, int(camModel_->w));
	float val_cp = getInterpolatedElement(referenceFrameImage,cpx, cpy, int(camModel_->w));
	float val_cp_p1 = getInterpolatedElement(referenceFrameImage,cpx+incx, cpy+incy, int(camModel_->w));
	float val_cp_p2;



	/*
	 * Subsequent exact minimum is found the following way:
	 * - assuming lin. interpolation, the gradient of Error at p1 (towards p2) is given by
	 *   dE1 = -2sum(e1*e1 - e1*e2)
	 *   where e1 and e2 are summed over, and are the residuals (not squared).
	 *
	 * - the gradient at p2 (coming from p1) is given by
	 * 	 dE2 = +2sum(e2*e2 - e1*e2)
	 *
	 * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
	 *   p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
	 *
	 *
	 *
	 * => I for later exact min calculation, I need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
	 *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
	 *    where i is the respective winning index.
	 */


	// walk in equally sized steps, starting at depth=infinity.
	int loopCounter = 0;
	float best_match_x = -1;
	float best_match_y = -1;
	float best_match_err = FLT_MAX;
	float second_best_match_err = FLT_MAX;

	// best pre and post errors.
	float best_match_errPre=NAN, best_match_errPost=NAN, best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
	bool bestWasLastLoop = false;

	float eeLast = -1; // final error of last comp.

	// alternating intermediate vars
	float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

	int loopCBest=-1, loopCSecond =-1;
	while(((incx < 0) == (cpx > pClose[0]) && (incy < 0) == (cpy > pClose[1])) || loopCounter == 0)
	{
		// interpolate one new point
		val_cp_p2 = getInterpolatedElement(referenceFrameImage,cpx+2*incx, cpy+2*incy, camModel_->w);


		// hacky but fast way to get error and differential error: switch buffer variables for last loop.
		float ee = 0;
		if(loopCounter%2==0)
		{
			// calc error and accumulate sums.
			e1A = val_cp_p2 - realVal_p2;ee += e1A*e1A;
			e2A = val_cp_p1 - realVal_p1;ee += e2A*e2A;
			e3A = val_cp - realVal;      ee += e3A*e3A;
			e4A = val_cp_m1 - realVal_m1;ee += e4A*e4A;
			e5A = val_cp_m2 - realVal_m2;ee += e5A*e5A;
		}
		else
		{
			// calc error and accumulate sums.
			e1B = val_cp_p2 - realVal_p2;ee += e1B*e1B;
			e2B = val_cp_p1 - realVal_p1;ee += e2B*e2B;
			e3B = val_cp - realVal;      ee += e3B*e3B;
			e4B = val_cp_m1 - realVal_m1;ee += e4B*e4B;
			e5B = val_cp_m2 - realVal_m2;ee += e5B*e5B;
		}


#if DEBUG_SAVE_SEARCH_RANGE_IMS
		if (drawThisPixel) {
			vec3 rgb = 255.f * hueToRgb(0.8f * ee / 325125.f);
			cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
			debugImages.searchRanges.at<cv::Vec3b>(int(cpy + 2 * incy),
				int(cpx + 2 * incx + camModel_->w)) = rgbB;
		}
#endif


		// do I have a new winner??
		// if so: set.
		if(ee < best_match_err)
		{
			// put to second-best
			second_best_match_err=best_match_err;
			loopCSecond = loopCBest;

			// set best.
			best_match_err = ee;
			loopCBest = loopCounter;

			best_match_errPre = eeLast;
			best_match_DiffErrPre = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
			best_match_errPost = -1;
			best_match_DiffErrPost = -1;

			best_match_x = cpx;
			best_match_y = cpy;
			bestWasLastLoop = true;
		}
		// otherwise: the last might be the current winner, in which case i have to save these values.
		else
		{
			if(bestWasLastLoop)
			{
				best_match_errPost = ee;
				best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
				bestWasLastLoop = false;
			}

			// collect second-best:
			// just take the best of all that are NOT equal to current best.
			if(ee < second_best_match_err)
			{
				second_best_match_err=ee;
				loopCSecond = loopCounter;
			}
		}


		// shift everything one further.
		eeLast = ee;
		val_cp_m2 = val_cp_m1; val_cp_m1 = val_cp; val_cp = val_cp_p1; val_cp_p1 = val_cp_p2;

		if(enablePrintDebugInfo) stats->num_stereo_comparisons++;

		cpx += incx;
		cpy += incy;

		loopCounter++;
	}

	// if error too big, will return -3, otherwise -2.
	if(best_match_err > 4.0f*(float)MAX_ERROR_STEREO)
	{
		if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return DepthMapErrCode::ERR_TOO_BIG;
	}


	// check if clear enough winner
	if(abs(loopCBest - loopCSecond) > 1.0f && MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err)
	{
		if(enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
		return DepthMapErrCode::WINNER_NOT_CLEAR;
	}

#if DEBUG_SAVE_SEARCH_RANGE_IMS
	if (drawThisPixel) {
		vec3 rgb(0, 255.f, 255.f);
		cv::Vec3b rgbB(uchar(rgb.z()), uchar(rgb.y()), uchar(rgb.x()));
		debugImages.searchRanges.at<cv::Vec3b>(
			int(best_match_y), int(best_match_x + camModel_->w)) = rgbB;
	}
#endif

	bool didSubpixel = false;
	if(false /*useSubpixelStereo*/)
	{
		// ================== compute exact match =========================
		// compute gradients (they are actually only half the real gradient)
		float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
		float gradPre_this = +(best_match_err - best_match_DiffErrPre);
		float gradPost_this = -(best_match_err - best_match_DiffErrPost);
		float gradPost_post = +(best_match_errPost - best_match_DiffErrPost);

		// final decisions here.
		bool interpPost = false;
		bool interpPre = false;

		// if one is oob: return false.
		if(enablePrintDebugInfo && (best_match_errPre < 0 || best_match_errPost < 0))
		{
			stats->num_stereo_invalid_atEnd++;
		}


		// - if zero-crossing occurs exactly in between (gradient Inconsistent),
		else if((gradPost_this < 0) ^ (gradPre_this < 0))
		{
			// return exact pos, if both central gradients are small compared to their counterpart.
			if(enablePrintDebugInfo && (gradPost_this*gradPost_this > 0.1f*0.1f*gradPost_post*gradPost_post ||
			   gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre))
				stats->num_stereo_invalid_inexistantCrossing++;
		}

		// if pre has zero-crossing
		else if((gradPre_pre < 0) ^ (gradPre_this < 0))
		{
			// if post has zero-crossing
			if((gradPost_post < 0) ^ (gradPost_this < 0))
			{
				if(enablePrintDebugInfo) stats->num_stereo_invalid_twoCrossing++;
			}
			else
				interpPre = true;
		}

		// if post has zero-crossing
		else if((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			interpPost = true;
		}

		// if none has zero-crossing
		else
		{
			if(enablePrintDebugInfo) stats->num_stereo_invalid_noCrossing++;
		}


		// DO interpolation!
		// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
		// the error at that point is also computed by just integrating.
		if(interpPre)
		{
			float d = gradPre_this / (gradPre_this - gradPre_pre);
			best_match_x -= d*incx;
			best_match_y -= d*incy;
			best_match_err = best_match_err - 2*d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
			if(enablePrintDebugInfo) stats->num_stereo_interpPre++;
			didSubpixel = true;

		}
		else if(interpPost)
		{
			float d = gradPost_this / (gradPost_this - gradPost_post);
			best_match_x += d*incx;
			best_match_y += d*incy;
			best_match_err = best_match_err + 2*d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
			if(enablePrintDebugInfo) stats->num_stereo_interpPost++;
			didSubpixel = true;
		}
		else
		{
			if(enablePrintDebugInfo) stats->num_stereo_interpNone++;
		}
	}


	// sampleDist is the distance in pixel at which the realVal's were sampled
	float sampleDist = GRADIENT_SAMPLE_DIST*rescaleFactor;

	float gradAlongLine = 0;
	float tmp = realVal_p2 - realVal_p1;  gradAlongLine+=tmp*tmp;
	tmp = realVal_p1 - realVal;  gradAlongLine+=tmp*tmp;
	tmp = realVal - realVal_m1;  gradAlongLine+=tmp*tmp;
	tmp = realVal_m1 - realVal_m2;  gradAlongLine+=tmp*tmp;

	gradAlongLine /= sampleDist*sampleDist;

	// check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
	if(best_match_err > (float)MAX_ERROR_STEREO + sqrtf( gradAlongLine)*20)
	{
		if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}


	// ================= calc depth (in KF) ====================
	// * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the KF.
	// * best_match_x = x-coordinate of found correspondence in the reference frame.

	float idnew_best_match;	// depth in the new image
	float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
	if(incx*incx>incy*incy)
	{
		float oldX = pm->fxi()*best_match_x+pm->cxi();
		float nominator = (oldX*referenceFrame->otherToThis_t[2] - referenceFrame->otherToThis_t[0]);
		float dot0 = KinvP.dot(referenceFrame->otherToThis_R_row0);
		float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

		idnew_best_match = (dot0 - oldX*dot2) / nominator;
		alpha = incx*pm->fxi()*(dot0*referenceFrame->otherToThis_t[2] - dot2*referenceFrame->otherToThis_t[0]) / (nominator*nominator);

	}
	else
	{
		float oldY = pm->fyi()*best_match_y+pm->cyi();

		float nominator = (oldY*referenceFrame->otherToThis_t[2] - referenceFrame->otherToThis_t[1]);
		float dot1 = KinvP.dot(referenceFrame->otherToThis_R_row1);
		float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

		idnew_best_match = (dot1 - oldY*dot2) / nominator;
		alpha = incy*pm->fyi()*(dot1*referenceFrame->otherToThis_t[2] - 
			dot2*referenceFrame->otherToThis_t[1]) / (nominator*nominator);
	}

	if(idnew_best_match < 0)
	{
		if(enablePrintDebugInfo) stats->num_stereo_negative++;
		if(!allowNegativeIdepths)
			return -2;
	}

	if(enablePrintDebugInfo) stats->num_stereo_successfull++;

	// ================= calc var (in NEW image) ====================

	// calculate error from photometric noise
	float photoDispError = 4.0f * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);

	float trackingErrorFac = 0.25f*(1.0f+referenceFrame->initialTrackedResidual);

	// calculate error from geometric noise (wrong camera pose / calibration)
	Eigen::Vector2f gradsInterp = getInterpolatedElement42(
		activeKeyframe->gradients(0), u, v, camModel_->w);
	if (gradsInterp[0] == 0 && gradsInterp[1] == 0) {
		throw 1;
	}
	float geoDispError = (gradsInterp[0]*epxn + gradsInterp[1]*epyn) + DIVISION_EPS;
	geoDispError = trackingErrorFac*trackingErrorFac*
		(gradsInterp[0]*gradsInterp[0] + gradsInterp[1]*gradsInterp[1]) / 
		(geoDispError*geoDispError);


	//geoDispError *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5 *result_idepth);

	// final error consists of a small constant part (discretization error),
	// geometric and photometric error.
	float discretizationError = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist);
	result_var = discretizationError +  geoDispError + photoDispError;	// square to make variance
#if DEBUG_SAVE_GRAD_ALONG_LINE_IMS
	debugImages.addGradAlongLine(size_t(u), size_t(v), gradAlongLine);
#endif
#if DEBUG_SAVE_PHOTO_DISP_ERROR_IMS	
	debugImages.addPhotoDispError(size_t(u), size_t(v), photoDispError);
#endif
#if DEBUG_SAVE_GEO_DISP_ERROR_IMS	
	debugImages.addGeoDispError(size_t(u), size_t(v), geoDispError);
#endif
#if DEBUG_SAVE_DISCRETIZATION_ERROR_IMS
	debugImages.addDiscretizationError(size_t(u), size_t(v), discretizationError);
#endif

	if(plotStereoImages)
	{
		if(rand()%5==0)
		{
			//if(rand()%500 == 0)
			//	printf("geo: %f, photo: %f, alpha: %f\n", sqrt(geoDispError), sqrt(photoDispError), alpha, sqrt(result_var));


			//int idDiff = (keyframe->pyramidID - referenceFrame->id);
			//cv::Scalar color = cv::Scalar(0,0, 2*idDiff);// bw

			//cv::Scalar color = cv::Scalar(sqrt(result_var)*2000, 255-sqrt(result_var)*2000, 0);// bw

//			float eplLengthF = std::min((float)MIN_EPL_LENGTH_CROP,(float)eplLength);
//			eplLengthF = std::max((float)MAX_EPL_LENGTH_CROP,(float)eplLengthF);
//
//			float pixelDistFound = sqrtf((float)((pReal[0]/pReal[2] - best_match_x)*(pReal[0]/pReal[2] - best_match_x)
//					+ (pReal[1]/pReal[2] - best_match_y)*(pReal[1]/pReal[2] - best_match_y)));
//
			float fac = best_match_err / ((float)MAX_ERROR_STEREO + sqrtf( gradAlongLine)*20);

			cv::Scalar color = cv::Scalar(255*fac, 255-255*fac, 0);// bw


			/*
			if(rescaleFactor > 1)
				color = cv::Scalar(500*(rescaleFactor-1),0,0);
			else
				color = cv::Scalar(0,500*(1-rescaleFactor),500*(1-rescaleFactor));
			*/

			cv::line(debugImageStereoLines,cv::Point2f(pClose[0], pClose[1]),cv::Point2f(pFar[0], pFar[1]),color,1,8,0);
		}
	}

	result_idepth = idnew_best_match;

	result_eplLength = eplLength;

#if DEBUG_SAVE_RESULT_IMS
	debugImages.results.at<cv::Vec3b>(int(v), int(u)) =
		cv::Vec3b(0, 255, 0);
#endif

	return best_match_err;
}

}
