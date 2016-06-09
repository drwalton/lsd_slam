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

#include "Util/settings.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.hpp"
#include "DepthEstimation/DepthMapInitSettings.hpp"
#include "DataStructures/Frame.hpp"
#include "Util/globalFuncs.hpp"
#include "IOWrapper/ImageDisplay.hpp"
#include "GlobalMapping/KeyFrameGraph.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"
#include "Util/ModelLoader.hpp"
#include "Util/KahanVal.hpp"


namespace lsd_slam
{

DepthMapDebugSettings::DepthMapDebugSettings()
	:debugShowEstimatedDepths(false),
	printPropagationStatistics(false),
    printFillHolesStatistics(false),
    printObserveStatistics(false),
    printObservePurgeStatistics(false),
    printRegularizeStatistics(false),
    printLineStereoStatistics(false),
    printLineStereoFails(false),
	saveAllFramesAsPointClouds(false),
	saveAllFramesAsVectorClouds(false),
	saveMatchesImages(false),
	saveSearchRangesImages(false)
{}

DepthMapDebugSettings::~DepthMapDebugSettings() throw()
{}

DepthMap::DepthMap(const CameraModel &model)
	:model(model.clone()), width(model.w), height(model.h)
{
	modelType = this->model->getType();
	size_t width = model.w, height = model.h;
	activeKeyFrame = 0;
	activeKeyFrameIsReactivated = false;
	otherDepthMap = new DepthMapPixelHypothesis[width*height];
	currentDepthMap = new DepthMapPixelHypothesis[width*height];

	validityIntegralBuffer = (int*)Eigen::internal::aligned_malloc(width*height*sizeof(int));

	debugImageHypothesisHandling = cv::Mat(height,width, CV_8UC3);
	debugImageHypothesisPropagation = cv::Mat(height,width, CV_8UC3);
	debugImageStereoLines = cv::Mat(height,width, CV_8UC3);
	debugImageDepth = cv::Mat(height,width, CV_8UC3);
	debugImageDepthGray = cv::Mat(height, width, CV_8UC1);

	reset();

	msUpdate =  msCreate =  msFinalize = 0;
	msObserve =  msRegularize =  msPropagate =  msFillHoles =  msSetDepth = 0;
	gettimeofday(&lastHzUpdate, NULL);
	nUpdate = nCreate = nFinalize = 0;
	nObserve = nRegularize = nPropagate = nFillHoles = nSetDepth = 0;
	nAvgUpdate = nAvgCreate = nAvgFinalize = 0;
	nAvgObserve = nAvgRegularize = nAvgPropagate = nAvgFillHoles = nAvgSetDepth = 0;
}

DepthMap::~DepthMap() throw()
{
	if(activeKeyFrame != 0)
		activeKeyFramelock.unlock();

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
	size_t width = model->w, height = model->h;
	for(DepthMapPixelHypothesis* pt = otherDepthMap+width*height-1; pt >= otherDepthMap; pt--)
		pt->isValid = false;
	for(DepthMapPixelHypothesis* pt = currentDepthMap+width*height-1; pt >= currentDepthMap; pt--)
		pt->isValid = false;
}


void DepthMap::observeDepthRow(size_t yMin, size_t yMax, RunningStats* stats)
{
	const float* keyFrameMaxGradBuf = activeKeyFrame->maxGradients(0);

	int successes = 0;

	for(size_t y=yMin;y<yMax; y++)
		for(size_t x=3;x<width-3;x++)
		{
			int idx = x+y*width;
			DepthMapPixelHypothesis* target = currentDepthMap+idx;
			bool hasHypothesis = target->isValid;

			// ======== 1. check absolute grad =========
			if(hasHypothesis && keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_DECREASE)
			{
				target->isValid = false;
				continue;
			}

			if(keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_CREATE || target->blacklisted < MIN_BLACKLIST)
				continue;


			bool success;
			if(!hasHypothesis)
				success = observeDepthCreate(x, y, idx, stats);
			else
				success = observeDepthUpdate(x, y, idx, keyFrameMaxGradBuf, stats);

			if(success)
				successes++;
		}


}
void DepthMap::observeDepth()
{
	if (settings.saveMatchesImages) {
		Frame* refFrame = activeKeyFrameIsReactivated ? 
			newest_referenceFrame : oldest_referenceFrame;
		debugUpdateVisualiseMatchIms(activeKeyFrameImageData, refFrame->image());
	}
	if (settings.saveSearchRangesImages) {
		Frame* refFrame = activeKeyFrameIsReactivated ? 
			newest_referenceFrame : oldest_referenceFrame;
		debugUpdateVisualiseSearchRangesIms(activeKeyFrameImageData, refFrame->image());
	}

	threadReducer.reduce(boost::bind(&DepthMap::observeDepthRow, this, _1, _2, _3), 3, height-3, 10);

	if(enablePrintDebugInfo && settings.printObserveStatistics)
	{
		printf("OBSERVE (%d): %d / %d created; %d / %d updated; %d skipped; %d init-blacklisted\n",
				activeKeyFrame->id(),
				runningStats.num_observe_created,
				runningStats.num_observe_create_attempted,
				runningStats.num_observe_updated,
				runningStats.num_observe_update_attempted,
				runningStats.num_observe_skip_alreadyGood,
				runningStats.num_observe_blacklisted
		);
	}


	if(enablePrintDebugInfo && settings.printObservePurgeStatistics)
	{
		printf("OBS-PRG (%d): Good: %d; inconsistent: %d; notfound: %d; oob: %d; failed: %d; addSkip: %d;\n",
				activeKeyFrame->id(),
				runningStats.num_observe_good,
				runningStats.num_observe_inconsistent,
				runningStats.num_observe_notfound,
				runningStats.num_observe_skip_oob,
				runningStats.num_observe_skip_fail,
				runningStats.num_observe_addSkip
		);
	}

	if (settings.debugShowEstimatedDepths) {
		debugPlotDepthMap();
		cv::imshow("DebugImageDepth", debugImageDepth);
		cv::waitKey(1);
	}
}



bool DepthMap::observeDepthCreate(const int &x, const int &y, const int &idx, RunningStats* const &stats)
{
	DepthMapPixelHypothesis* target = currentDepthMap+idx;

	Frame* refFrame = activeKeyFrameIsReactivated ? newest_referenceFrame : oldest_referenceFrame;

	if(refFrame->getTrackingParent() == activeKeyFrame)
	{
		bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
		if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,0,0); // BLUE for SKIPPED NOT GOOD TRACKED
			return false;
		}
	}

	float new_u = float(x);
	float new_v = float(y);
	float result_idepth, result_var, result_eplLength;
	float error;
	if (modelType == CameraModelType::PROJ) {
		float epx, epy;
		bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);
		if(!isGood) return false;
		error = DepthMap::doLineStereo(
			new_u, new_v, epx, epy,
			0.0f, 1.0f, 1.0f / MIN_DEPTH,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	} else /*modelType == CameraModelType::OMNI*/ {
		vec3 epDir;
		bool isGood = makeAndCheckEPLOmni(x, y, refFrame, &epDir, stats);
		if(!isGood) return false;
		error = doOmniStereo(
			new_u, new_v, epDir,
			0.0f, 1.0f, 1.0f / MIN_DEPTH,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	}

	if(enablePrintDebugInfo) stats->num_observe_create_attempted++;

	if(error == -3 || error == -2)
	{
		target->blacklisted--;
		if(enablePrintDebugInfo) stats->num_observe_blacklisted++;
	}

	if(error < 0 || result_var > MAX_VAR)
		return false;
	
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

bool DepthMap::observeDepthUpdate(const int &x, const int &y, const int &idx, const float* keyFrameMaxGradBuf, RunningStats* const &stats)
{
	DepthMapPixelHypothesis* target = currentDepthMap+idx;
	Frame* refFrame;


	if(!activeKeyFrameIsReactivated)
	{
		if((int)target->nextStereoFrameMinID - referenceFrameByID_offset >= (int)referenceFrameByID.size())
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255,0);	// GREEN FOR skip

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


	if(refFrame->getTrackingParent() == activeKeyFrame)
	{
		bool* wasGoodDuringTracking = refFrame->refPixelWasGoodNoCreate();
		if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
		{
			if(plotStereoImages)
				debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(255,0,0); // BLUE for SKIPPED NOT GOOD TRACKED
			return false;
		}
	}

	float error = 0.f;
	float result_idepth, result_var, result_eplLength;
	if (modelType == CameraModelType::PROJ) {
		float epx, epy;
		bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);
		if (!isGood) return false;

		// which exact point to track, and where from.
		float sv = sqrt(target->idepth_var_smoothed);
		float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC;
		float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC;
		if (min_idepth < 0) min_idepth = 0;
		if (max_idepth > 1 / MIN_DEPTH) max_idepth = 1 / MIN_DEPTH;

		error = doLineStereo(
			float(x), float(y), epx, epy,
			min_idepth, target->idepth_smoothed, max_idepth,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	} else /*CameraModelType::OMNI*/ {
		vec3 epDir;
		bool isGood = makeAndCheckEPLOmni(x, y, refFrame, &epDir, stats);
		if (!isGood) return false;

		// which exact point to track, and where from.
		float sv = sqrt(target->idepth_var_smoothed);
		float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC_OMNI;
		float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC_OMNI;
		if (min_idepth < 0) min_idepth = 0;
		if (max_idepth > 1 / MIN_DEPTH) max_idepth = 1 / MIN_DEPTH;

		error = doOmniStereo(
			float(x), float(y), epDir,
			min_idepth, target->idepth_smoothed, max_idepth,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);
	}
	stats->num_observe_update_attempted++;

	float diff = result_idepth - target->idepth_smoothed;


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
		// one more successful observation!
		if(enablePrintDebugInfo) stats->num_observe_good++;

		if(enablePrintDebugInfo) stats->num_observe_updated++;


		// do textbook ekf update:
		// increase var by a little (prediction-uncertainty)
		float id_var = target->idepth_var*SUCC_VAR_INC_FAC;

		// update var with observation
		float w = result_var / (result_var + id_var);
		float new_idepth = (1-w)*result_idepth + w*target->idepth;
		assert(new_idepth == new_idepth);
		target->idepth = static_cast<float>(UNZERO(new_idepth));

		// variance can only decrease from observation; never increase.
		id_var = id_var * w;
		if(id_var < target->idepth_var)
			target->idepth_var = id_var;

		// increase validity!
		target->validity_counter += VALIDITY_COUNTER_INC;
		float absGrad = keyFrameMaxGradBuf[idx];
		if(target->validity_counter > VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f)
			target->validity_counter = static_cast<int>(
				VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f);

		// increase Skip!
		if(result_eplLength < MIN_EPL_LENGTH_CROP)
		{
			float inc = activeKeyFrame->numFramesTrackedOnThis / (float)(activeKeyFrame->numMappedOnThis+5);
			if(inc < 3) inc = 3;

			inc +=  ((int)(result_eplLength*10000)%2);

			if(enablePrintDebugInfo) stats->num_observe_addSkip++;

			if(result_eplLength < 0.5*MIN_EPL_LENGTH_CROP)
				inc *= 3;


			target->nextStereoFrameMinID = refFrame->id() + inc;
		}

		if(plotStereoImages)
			debugImageHypothesisHandling.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255,255); // yellow for GOT UPDATED

		return true;
	}
}

void DepthMap::propagateDepth(Frame* new_keyframe)
{
	//TODO write test for depth map propagation
	runningStats.num_prop_removed_out_of_bounds = 0;
	runningStats.num_prop_removed_colorDiff = 0;
	runningStats.num_prop_removed_validity = 0;
	runningStats.num_prop_grad_decreased = 0;
	runningStats.num_prop_color_decreased = 0;
	runningStats.num_prop_attempts = 0;
	runningStats.num_prop_occluded = 0;
	runningStats.num_prop_created = 0;
	runningStats.num_prop_merged = 0;


	if(new_keyframe->getTrackingParent() != activeKeyFrame)
	{
		printf("WARNING: propagating depth from frame %d to %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
				activeKeyFrame->id(), new_keyframe->id(),
				new_keyframe->getTrackingParent()->id());
	}

	// wipe depthmap
	for(DepthMapPixelHypothesis* pt = otherDepthMap+width*height-1; pt >= otherDepthMap; pt--)
	{
		pt->isValid = false;
		pt->blacklisted = 0;
	}

	// re-usable values.
	SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();
	Eigen::Vector3f trafoInv_t = oldToNew_SE3.translation().cast<float>();
	Eigen::Matrix3f trafoInv_R = oldToNew_SE3.rotationMatrix().matrix().cast<float>();


	const bool* trackingWasGood = new_keyframe->getTrackingParent() == activeKeyFrame ? new_keyframe->refPixelWasGoodNoCreate() : 0;


	const float* activeKFImageData = activeKeyFrame->image(0);
	const float* newKFMaxGrad = new_keyframe->maxGradients(0);
	const float* newKFImageData = new_keyframe->image(0);
	
	// go through all pixels of OLD image, propagating forwards.
	for(size_t y=0;y<height;y++)
		for(size_t x=0;x<width;x++)
		{
			DepthMapPixelHypothesis* source = currentDepthMap + x + y*width;

			if(!source->isValid)
				continue;
			
			if(enablePrintDebugInfo) runningStats.num_prop_attempts++;

			Eigen::Vector3f pn = (trafoInv_R *
				model->pixelToCam(vec2(x, y))) / source->idepth_smoothed + trafoInv_t;
			//TODO: See if we need to use below formula for Omni mode.
			//Eigen::Vector3f pn = (trafoInv_R * 
			//	model->pixelToCam(vec2(x, y), 1.f/source->idepth_smoothed)) + trafoInv_t;

			float new_idepth;
			if(model->getType() == CameraModelType::PROJ) {
				new_idepth = 1.0f / pn[2];
			} else {
				new_idepth = pn.norm();
			}

			vec2 uv = model->camToPixel(pn);
			float u_new = uv[0];
			float v_new = uv[1];

			// check if still within image, if not: DROP.
			if (model->getType() == CameraModelType::OMNI && (!model->pixelLocValid(uv))) {
				if(enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
				continue;
			}
			if(!(u_new > 2.1f && v_new > 2.1f && u_new < width-3.1f && v_new < height-3.1f)) {
				if(enablePrintDebugInfo) runningStats.num_prop_removed_out_of_bounds++;
				continue;
			}

			int newIDX = (int)(u_new+0.5f) + ((int)(v_new+0.5f))*width;
			float destAbsGrad = newKFMaxGrad[newIDX];

			if(trackingWasGood != 0)
			{
				if(!trackingWasGood[(x >> SE3TRACKING_MIN_LEVEL) + 
					(width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)]
				                    || destAbsGrad < MIN_ABS_GRAD_DECREASE)
				{
					if(enablePrintDebugInfo) runningStats.num_prop_removed_colorDiff++;
					continue;
				}
			}
			else
			{
				float sourceColor = activeKFImageData[x + y*width];
				float destColor = getInterpolatedElement(newKFImageData, u_new, v_new, width);

				float residual = destColor - sourceColor;


				if(residual*residual / (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*
					destAbsGrad*destAbsGrad) > 1.0f || destAbsGrad < MIN_ABS_GRAD_DECREASE)
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
			}
		}

	// swap!
	std::swap(currentDepthMap, otherDepthMap);


	if(enablePrintDebugInfo && settings.printPropagationStatistics)
	{
		printf("PROPAGATE: %d: %d drop (%d oob, %d color); %d created; "
			"%d merged; %d occluded. %d col-dec, %d grad-dec.\n",
				runningStats.num_prop_attempts,
				runningStats.num_prop_removed_validity + 
					runningStats.num_prop_removed_out_of_bounds + 
					runningStats.num_prop_removed_colorDiff,
				runningStats.num_prop_removed_out_of_bounds,
				runningStats.num_prop_removed_colorDiff,
				runningStats.num_prop_created,
				runningStats.num_prop_merged,
				runningStats.num_prop_occluded,
				runningStats.num_prop_color_decreased,
				runningStats.num_prop_grad_decreased);
	}
}

void DepthMap::regularizeDepthMapFillHolesRow(size_t yMin, size_t yMax, RunningStats* stats)
{
	// =========== regularize fill holes
	const float* keyFrameMaxGradBuf = activeKeyFrame->maxGradients(0);

	for(size_t y=yMin; y<yMax; y++)
	{
		for(size_t x=3;x<width-2;x++)
		{
			int idx = x+y*width;
			DepthMapPixelHypothesis* dest = otherDepthMap + idx;
			if(dest->isValid) continue;
			if(keyFrameMaxGradBuf[idx]<MIN_ABS_GRAD_DECREASE) continue;

			int* io = validityIntegralBuffer + idx;
			int val = io[2+2*width] - io[2-3*width] - io[-3+2*width] + io[-3-3*width];


			if((dest->blacklisted >= MIN_BLACKLIST && val > VAL_SUM_MIN_FOR_CREATE) 
				|| val > VAL_SUM_MIN_FOR_UNBLACKLIST)
			{
				float sumIdepthObs = 0, sumIVarObs = 0;
				int num = 0;

				DepthMapPixelHypothesis* s1max = otherDepthMap + (x-2) + (y+3)*width;
				for (DepthMapPixelHypothesis* s1 = otherDepthMap + (x-2) + 
					(y-2)*width; s1 < s1max; s1+=width)
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

	memcpy(otherDepthMap,currentDepthMap,width*height*sizeof(DepthMapPixelHypothesis));
	threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapFillHolesRow,
		this, _1, _2, _3), 3, height-2, 10);
	if(enablePrintDebugInfo && settings.printFillHolesStatistics)
		printf("FillHoles (discreteDepth): %d created\n",
				runningStats.num_reg_created);
}

void DepthMap::buildRegIntegralBufferRow1(size_t yMin, size_t yMax, RunningStats* stats)
{
	// ============ build inegral buffers
	int* validityIntegralBufferPT = validityIntegralBuffer+yMin*width;
	DepthMapPixelHypothesis* ptSrc = currentDepthMap+yMin*width;
	for(size_t y=yMin;y<yMax;y++)
	{
		int validityIntegralBufferSUM = 0;

		for(size_t x=0;x<width;x++)
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
	threadReducer.reduce(boost::bind(&DepthMap::buildRegIntegralBufferRow1, this, _1, _2,_3), 0, height);

	int* validityIntegralBufferPT = validityIntegralBuffer;
	int* validityIntegralBufferPT_T = validityIntegralBuffer+width;

	int wh = height*width;
	for(int idx=width;idx<wh;idx++)
		*(validityIntegralBufferPT_T++) += *(validityIntegralBufferPT++);

}

template<bool removeOcclusions> void DepthMap::regularizeDepthMapRow(int validityTH, int yMin, int yMax, RunningStats* stats)
{
	const int regularize_radius = 2;

	const float regDistVar = REG_DIST_VAR;

	for(int y=yMin;y<yMax;y++)
	{
		for(int x=regularize_radius; x < static_cast<int>(width) - regularize_radius; x++)
		{
			DepthMapPixelHypothesis* dest = currentDepthMap + x + y*width;
			DepthMapPixelHypothesis* destRead = otherDepthMap + x + y*width;

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
					DepthMapPixelHypothesis* source = destRead + dx + dy*width;

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

	memcpy(otherDepthMap,currentDepthMap,width*height*sizeof(DepthMapPixelHypothesis));


	if(removeOcclusions)
		threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<true>, this, validityTH, _1, _2, _3), 2, height-2, 10);
	else
		threadReducer.reduce(boost::bind(&DepthMap::regularizeDepthMapRow<false>, this, validityTH, _1, _2, _3), 2, height-2, 10);


	if(enablePrintDebugInfo &&
		settings.printRegularizeStatistics)
		printf("REGULARIZE (%d): %d smeared; %d blacklisted /%d new); %d deleted; %d occluded; %d filled\n",
				activeKeyFrame->id(),
				runningStats.num_reg_smeared,
				runningStats.num_reg_blacklisted,
				runningStats.num_reg_setBlacklisted,
				runningStats.num_reg_deleted_secondary,
				runningStats.num_reg_deleted_occluded,
				runningStats.num_reg_created);
}

void DepthMap::initializeRandomly(Frame* new_frame)
{
	activeKeyFramelock = new_frame->getActiveLock();
	activeKeyFrame = new_frame;
	activeKeyFrameImageData = activeKeyFrame->image(0);
	activeKeyFrameIsReactivated = false;

	const float* maxGradients = new_frame->maxGradients();

	for(size_t y=1;y<height-1;y++)
	{
		for(size_t x=1;x<width-1;x++)
		{
			if(maxGradients[x+y*width] > MIN_ABS_GRAD_CREATE)
			{
				//float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
				float idepth = 1.f + 2.f * ((rand() % 100001) / 100000.0f);
				currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
						idepth,
						idepth,
						VAR_RANDOM_INIT_INITIAL,
						VAR_RANDOM_INIT_INITIAL,
						20);
			}
			else
			{
				currentDepthMap[x+y*width].isValid = false;
				currentDepthMap[x+y*width].blacklisted = 0;
			}
		}
	}


	activeKeyFrame->setDepth(currentDepthMap);
}

void DepthMap::setFromExistingKF(Frame* kf)
{
	assert(kf->hasIDepthBeenSet());

	activeKeyFramelock = kf->getActiveLock();
	activeKeyFrame = kf;

	const float* idepth = activeKeyFrame->idepth_reAct();
	const float* idepthVar = activeKeyFrame->idepthVar_reAct();
	const unsigned char* validity = activeKeyFrame->validity_reAct();

	DepthMapPixelHypothesis* pt = currentDepthMap;
	activeKeyFrame->numMappedOnThis = 0;
	activeKeyFrame->numFramesTrackedOnThis = 0;
	activeKeyFrameImageData = activeKeyFrame->image(0);
	activeKeyFrameIsReactivated = true;

	for(size_t y=0;y<height;y++)
	{
		for(size_t x=0;x<width;x++)
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
				currentDepthMap[x+y*width].isValid = false;
				currentDepthMap[x+y*width].blacklisted = (*idepthVar == -2) ? MIN_BLACKLIST-1 : 0;
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

	activeKeyFramelock = new_frame->getActiveLock();
	activeKeyFrame = new_frame;
	activeKeyFrameImageData = activeKeyFrame->image(0);
	activeKeyFrameIsReactivated = false;

	const float* idepth = new_frame->idepth();


	float averageGTIDepthSum = 0;
	int averageGTIDepthNum = 0;
	for(size_t y=0;y<height;y++)
	{
		for(size_t x=0;x<width;x++)
		{
			float idepthValue = idepth[x+y*width];
			if(!isnan(idepthValue) && idepthValue > 0)
			{
				averageGTIDepthSum += idepthValue;
				averageGTIDepthNum ++;
			}
		}
	}
	

	for(size_t y=0;y<height;y++)
	{
		for(size_t x=0;x<width;x++)
		{
			float idepthValue = idepth[x+y*width];
			
			if(!isnan(idepthValue) && idepthValue > 0)
			{
				currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
						idepthValue,
						idepthValue,
						VAR_GT_INIT_INITIAL,
						VAR_GT_INIT_INITIAL,
						20);
			}
			else
			{
				currentDepthMap[x+y*width].isValid = false;
				currentDepthMap[x+y*width].blacklisted = 0;
			}
		}
	}


	activeKeyFrame->setDepth(currentDepthMap);
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

		if(frame->getTrackingParent() != activeKeyFrame)
		{
			printf("WARNING: updating frame %d with %d, which was tracked on a different frame (%d).\nWhile this should work, it is not recommended.",
					activeKeyFrame->id(), frame->id(),
					frame->getTrackingParent()->id());
		}

		Sim3 refToKf;
		if(frame->pose->trackingParent->frameID == activeKeyFrame->id())
			refToKf = frame->pose->thisToParent_raw;
		else
			refToKf = activeKeyFrame->getScaledCamToWorld().inverse() *  frame->getScaledCamToWorld();

		frame->prepareForStereoWith(activeKeyFrame, refToKf, *model, 0);

		while((int)referenceFrameByID.size() + referenceFrameByID_offset <= frame->id())
			referenceFrameByID.push_back(frame.get());
	}

	resetCounters();

	
	if(plotStereoImages)
	{
		cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));
		keyFrameImage.convertTo(debugImageHypothesisHandling, CV_8UC1);
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
	msObserve = 0.9f*msObserve + 
		0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + 
		(tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nObserve++;

	//if(rand()%10==0)
	{
		gettimeofday(&tv_start, NULL);
		regularizeDepthMapFillHoles();
		gettimeofday(&tv_end, NULL);
		msFillHoles = 0.9f*msFillHoles + 
			0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + 
			(tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFillHoles++;
	}


	gettimeofday(&tv_start, NULL);
	regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
	gettimeofday(&tv_end, NULL);
	msRegularize = 0.9f*msRegularize + 
		0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + 
		(tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nRegularize++;

	
	// Update depth in keyframe
	if(!activeKeyFrame->depthHasBeenUpdatedFlag)
	{
		gettimeofday(&tv_start, NULL);
		activeKeyFrame->setDepth(currentDepthMap);
		gettimeofday(&tv_end, NULL);
		msSetDepth = 0.9f*msSetDepth + 
			0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + 
			(tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nSetDepth++;
	}


	gettimeofday(&tv_end_all, NULL);
	msUpdate = 0.9f*msUpdate + 
		0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + 
		(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nUpdate++;


	activeKeyFrame->numMappedOnThis++;
	activeKeyFrame->numMappedOnThisTotal++;


	if(plotStereoImages)
	{
		Util::displayImage( "Stereo Key Frame", debugImageHypothesisHandling, false );
		Util::displayImage( "Stereo Reference Frame", debugImageStereoLines, false );
	}



	if(enablePrintDebugInfo && settings.printLineStereoStatistics)
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
	if(enablePrintDebugInfo && settings.printLineStereoFails)
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
	
	if(settings.saveAllFramesAsPointClouds) {
		std::stringstream ss;
		ss << resourcesDir() << "KFClouds/KF" << activeKeyFrame->id() <<
			"_f" << referenceFrames[0]->id() << ".ply";
		saveCurrMapAsPointCloud(ss.str());
	}
	if(settings.saveAllFramesAsVectorClouds) {
		std::stringstream ss;
		ss << resourcesDir() << "KFClouds/VKF" << activeKeyFrame->id() <<
			"_f" << referenceFrames[0]->id() << ".ply";
		saveCurrMapAsVectorCloud(ss.str());
	}
	if (settings.saveMatchesImages) {
		std::stringstream ss;
		ss << resourcesDir() << "MatchIms/MatchKF" << activeKeyFrame->id() <<
			"_f" << referenceFrames[0]->id() << ".png";
		cv::imwrite(ss.str(), debugVisualiseMatchesIm);
	}
	if (settings.saveSearchRangesImages) {
		std::stringstream ss;
		ss << resourcesDir() << "RangeIms/RangeKF" << activeKeyFrame->id() <<
			"_f" << referenceFrames[0]->id() << ".png";
		cv::imwrite(ss.str(), debugVisualiseSearchRangesIm);
	}
}

void DepthMap::invalidate()
{
	if(activeKeyFrame==0) return;
	activeKeyFrame=0;
	activeKeyFramelock.unlock();
}

void DepthMap::createKeyFrame(Frame* new_keyframe)
{
	assert(isValid());
	assert(new_keyframe != nullptr);
	assert(new_keyframe->hasTrackingParent());

	//boost::shared_lock<boost::shared_mutex> lock = activeKeyFrame->getActiveLock();
	boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

	struct timeval tv_start_all, tv_end_all;
	gettimeofday(&tv_start_all, NULL);


	resetCounters();

	if(plotStereoImages)
	{
		cv::Mat keyFrameImage(new_keyframe->height(), new_keyframe->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
		keyFrameImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
		cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation, CV_GRAY2RGB);
	}



	SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();

	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);
	propagateDepth(new_keyframe);
	gettimeofday(&tv_end, NULL);
	msPropagate = 0.9f*msPropagate + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nPropagate++;

	activeKeyFrame = new_keyframe;
	activeKeyFramelock = activeKeyFrame->getActiveLock();
	activeKeyFrameImageData = new_keyframe->image(0);
	activeKeyFrameIsReactivated = false;



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
	for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
	{
		if(!source->isValid)
			continue;
		sumIdepth += source->idepth_smoothed;
		numIdepth++;
	}
	float rescaleFactor = numIdepth / sumIdepth;
	float rescaleFactor2 = rescaleFactor*rescaleFactor;
	for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
	{
		if(!source->isValid)
			continue;
		source->idepth *= rescaleFactor;
		source->idepth_smoothed *= rescaleFactor;
		source->idepth_var *= rescaleFactor2;
		source->idepth_var_smoothed *= rescaleFactor2;
	}
	activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
	activeKeyFrame->pose->invalidateCache();

	// Update depth in keyframe

	gettimeofday(&tv_start, NULL);
	activeKeyFrame->setDepth(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msCreate = 0.9f*msCreate + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nCreate++;



	if(plotStereoImages)
	{
		//Util::displayImage( "KeyFramePropagation", debugImageHypothesisPropagation );
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

void DepthMap::finalizeKeyFrame()
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
	activeKeyFrame->setDepth(currentDepthMap);
	activeKeyFrame->calculateMeanInformation();
	activeKeyFrame->takeReActivationData(currentDepthMap);
	gettimeofday(&tv_end, NULL);
	msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nSetDepth++;

	gettimeofday(&tv_end_all, NULL);
	msFinalize = 0.9f*msFinalize + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nFinalize++;
}


void DepthMap::saveCurrMapAsPointCloud(const std::string &filename)
{
	ModelLoader modelLoader;
	
	KahanVal<float> meanVar = 0.f;
	for(size_t i = 0; i < width*height; ++i) {
		if(currentDepthMap[i].isValid) {
			meanVar += currentDepthMap[i].idepth_var_smoothed;
		}
	}
	float mVar = meanVar.get();
	mVar /= float(width*height);
	
	for(size_t r = 0; r < height; ++r) {
		for(size_t c = 0; c < width; ++c) {
			if(currentDepthMap[r*width + c].isValid) {
    			vec3 point = model->pixelToCam(
    				vec2(c,r), 1.f / currentDepthMap[r*width + c].idepth_smoothed);
    			if(std::isfinite(point[0]) &&
    			   std::isfinite(point[1]) &&
    			   std::isfinite(point[2])) {
    				
        			float r_var = currentDepthMap[r*width + c].idepth_var_smoothed;
    				if(r_var < mVar*2.f) {
            			modelLoader.vertices().push_back(point);
            			modelLoader.vertColors().push_back(
            						255.f*hueToRgb(0.8f*(r_var) / (mVar * 2.f)));
    				}
    			}
			}
		}
	}
	modelLoader.saveFile(filename);
}

void DepthMap::saveCurrMapAsVectorCloud(const std::string &filename)
{
	const std::array<GLuint, 36> CUBOID_INDICES = {
		3,1,0, 3,2,1,
		0,5,4, 0,1,5,
		1,6,5, 1,2,6,
		3,4,7, 3,0,4,
		2,7,6, 2,3,7,
		4,6,7, 4,5,6
	};
	const float cuboidWidth = 0.01f;

	ModelLoader modelLoader;
	
	for(size_t r = 0; r < height; r += 3) {
		for(size_t c = 0; c < width; c += 3) {
			if(currentDepthMap[r*width + c].isValid) {
       			float var = currentDepthMap[r*width + c].idepth_var_smoothed;
				float depth = 1.f / currentDepthMap[r*width + c].idepth_smoothed;
				vec3 startPt =  model->pixelToCam(vec2(c,r), fmaxf(
					0.001f, depth - 2.f*var));
				vec3 endPt =  model->pixelToCam(vec2(c,r), depth + 2.f*var);
    			if(std::isfinite(startPt[0]) &&
    			   std::isfinite(startPt[1]) &&
    			   std::isfinite(startPt[2]) &&
   					std::isfinite(endPt[0]) &&
    			   std::isfinite(endPt[1]) &&
    			   std::isfinite(endPt[2]) ) {
					vec3 perp1 = (endPt-startPt).cross(vec3(0.f, 0.f, 1.f)).normalized();
					vec3 perp2 = (endPt-startPt).cross(perp1).normalized();
					perp1 *= cuboidWidth*0.5f;
					perp2 *= cuboidWidth*0.5f;

					GLuint startIdx = modelLoader.vertices().size();

           			modelLoader.vertices().push_back((startPt - perp1) - perp2);
           			modelLoader.vertices().push_back((startPt + perp1) - perp2);
           			modelLoader.vertices().push_back((startPt + perp1) + perp2);
           			modelLoader.vertices().push_back((startPt - perp1) + perp2);

           			modelLoader.vertices().push_back((endPt - perp1) - perp2);
           			modelLoader.vertices().push_back((endPt + perp1) - perp2);
           			modelLoader.vertices().push_back((endPt + perp1) + perp2);
           			modelLoader.vertices().push_back((endPt - perp1) + perp2);

					for (size_t i = 0; i < CUBOID_INDICES.size(); ++i) {
						modelLoader.indices().push_back(CUBOID_INDICES[i] + startIdx);
					}
    			}
			}
		}
	}
	modelLoader.saveFile(filename);
}

int DepthMap::debugPlotDepthMap()
{
	if(activeKeyFrame == 0) return 1;

	cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));
	keyFrameImage.convertTo(debugImageDepthGray, CV_8UC1);
	cv::cvtColor(debugImageDepthGray, debugImageDepth, CV_GRAY2RGB);
	//int nchannels = debugImageDepth.channels();

	// debug plot & publish sparse version?
	int refID = referenceFrameByID_offset;

	

	size_t numValid = 0;
	for(size_t y=0;y<height;y++)
		for(size_t x=0;x<width;x++)
		{
			int idx = x + y*width;

			if(currentDepthMap[idx].blacklisted < MIN_BLACKLIST && debugDisplay == 2)
				debugImageDepth.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);

			if(!currentDepthMap[idx].isValid) continue;

			cv::Vec3b color = currentDepthMap[idx].getVisualizationColor(refID);
			++numValid;
			debugImageDepth.at<cv::Vec3b>(y,x) = color;
		}

	return 1;
}

void DepthMap::debugUpdateVisualiseMatchIms(
	const float *keyframe, const float *referenceFrame)
{
	cv::Mat &i = debugVisualiseMatchesIm;
	if (i.cols != model->w * 2 || i.rows != model->h) {
		i = cv::Mat(model->h, model->w * 2, CV_8UC3);
	}
	for (size_t r = 0; r < model->h; ++r) {
		cv::Vec3b *rptr = i.ptr<cv::Vec3b>(r);
		for (size_t c = 0; c < model->w; ++c) {
			uchar kfVal = static_cast<uchar>(keyframe[r*model->w + c]);
			uchar refVal = static_cast<uchar>(referenceFrame[r*model->w + c]);
			rptr[c] = cv::Vec3b(kfVal, kfVal, kfVal);
			rptr[c + model->w] = cv::Vec3b(refVal, refVal, refVal);
		}
	}
}

void DepthMap::debugUpdateVisualiseSearchRangesIms(
	const float *keyframe, const float *referenceFrame)
{
	cv::Mat &i = debugVisualiseSearchRangesIm;
	if (i.cols != model->w * 2 || i.rows != model->h) {
		i = cv::Mat(model->h, model->w * 2, CV_8UC3);
	}
	for (size_t r = 0; r < model->h; ++r) {
		cv::Vec3b *rptr = i.ptr<cv::Vec3b>(r);
		for (size_t c = 0; c < model->w; ++c) {
			uchar kfVal = static_cast<uchar>(keyframe[r*model->w + c]);
			uchar refVal = static_cast<uchar>(referenceFrame[r*model->w + c]);
			rptr[c] = cv::Vec3b(kfVal, kfVal, kfVal);
			rptr[c + model->w] = cv::Vec3b(refVal, refVal, refVal);
		}
	}
}


void DepthMap::debugVisualiseMatch(vec2 keyframePos, vec2 referenceFramePos)
{
	const int matchDisplayInvChance = 100;
	if (rand() % matchDisplayInvChance == 0) {
		//Show match.
		float hue = float(rand() % 256) / 256.f;
		vec3 rgb = hueToRgb(hue);
		cv::line(debugVisualiseMatchesIm,
			cv::Point(int(keyframePos.x()), int(keyframePos.y())),
			cv::Point(int(referenceFramePos.x()) + model->w, int(referenceFramePos.y())),
			cv::Vec3b(uchar(255.f*rgb[0]), uchar(255.f*rgb[1]), uchar(255.f*rgb[2])));
	}
}

}
