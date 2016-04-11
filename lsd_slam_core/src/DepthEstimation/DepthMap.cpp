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
#include "GlobalMapping/KeyFrameGraph.hpp"

namespace lsd_slam
{

DepthMap::DepthMap(const CameraModel &model)
	:model(model.clone()), width(model.w), height(model.h)
{
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

	threadReducer.reduce(boost::bind(&DepthMap::observeDepthRow, this, _1, _2, _3), 3, height-3, 10);

	if(enablePrintDebugInfo && printObserveStatistics)
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


	if(enablePrintDebugInfo && printObservePurgeStatistics)
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
}

bool DepthMap::makeAndCheckEPL(const int x, const int y, const Frame* const ref,
	float* pepx, float* pepy, RunningStats* const stats)
{
	int idx = x+y*width;

	// ======= make epl ========
	// Find direction towards epipole, in the keyframe image.
	vec2 epipole = model->camToPixel(ref->thisToOther_t);
	float epx = x - epipole.x();
	float epy = y - epipole.y();
	
	//Original code for the above:
	//float epx = - fx * ref->thisToOther_t[0] + ref->thisToOther_t[2]*(x - cx);
	//float epy = - fy * ref->thisToOther_t[1] + ref->thisToOther_t[2]*(y - cy);

	if(isnan(epx+epy)) {
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
	float gx = activeKeyFrameImageData[idx+1    ] - activeKeyFrameImageData[idx-1    ];
	float gy = activeKeyFrameImageData[idx+width] - activeKeyFrameImageData[idx-width];
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


	///TODO: rescaling like this assumes the EPL is straight - change this part.
	// ===== DONE - return "normalized" epl =====
	float fac = GRADIENT_SAMPLE_DIST / sqrt(eplLengthSquared);
	*pepx = epx * fac;
	*pepy = epy * fac;

	return true;
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

	float epx, epy;
	bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);
	if(!isGood) return false;

	if(enablePrintDebugInfo) stats->num_observe_create_attempted++;

	float new_u = float(x);
	float new_v = float(y);
	float result_idepth, result_var, result_eplLength;
	float error = doLineStereo(
			new_u,new_v,epx,epy,
			0.0f, 1.0f, 1.0f/MIN_DEPTH,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);

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

	float epx, epy;
	bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);
	if(!isGood) return false;

	// which exact point to track, and where from.
	float sv = sqrt(target->idepth_var_smoothed);
	float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC;
	float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC;
	if(min_idepth < 0) min_idepth = 0;
	if(max_idepth > 1/MIN_DEPTH) max_idepth = 1/MIN_DEPTH;

	stats->num_observe_update_attempted++;

	float result_idepth, result_var, result_eplLength;

	float error = doLineStereo(
			float(x),float(y),epx,epy,
			min_idepth, target->idepth_smoothed ,max_idepth,
			refFrame, refFrame->image(0),
			result_idepth, result_var, result_eplLength, stats);

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

			float new_idepth = 1.0f / pn[2];

			vec2 uv = model->camToPixel(pn);
			float u_new = uv[0];
			float v_new = uv[1];

			// check if still within image, if not: DROP.
			if(!(u_new > 2.1f && v_new > 2.1f && u_new < width-3.1f && v_new < height-3.1f))
			{
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


	if(enablePrintDebugInfo && printPropagationStatistics)
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
	if(enablePrintDebugInfo && printFillHolesStatistics)
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


	if(enablePrintDebugInfo && printRegularizeStatistics)
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
				float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
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
	if(!activeKeyFrame->depthHasBeenUpdatedFlag)
	{
		gettimeofday(&tv_start, NULL);
		activeKeyFrame->setDepth(currentDepthMap);
		gettimeofday(&tv_end, NULL);
		msSetDepth = 0.9f*msSetDepth + 0.1f*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nSetDepth++;
	}


	gettimeofday(&tv_end_all, NULL);
	msUpdate = 0.9f*msUpdate + 0.1f*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
	nUpdate++;


	activeKeyFrame->numMappedOnThis++;
	activeKeyFrame->numMappedOnThisTotal++;


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




int DepthMap::debugPlotDepthMap()
{
	if(activeKeyFrame == 0) return 1;

	cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));
	keyFrameImage.convertTo(debugImageDepth, CV_8UC1);
	cv::cvtColor(debugImageDepth, debugImageDepth, CV_GRAY2RGB);

	// debug plot & publish sparse version?
	int refID = referenceFrameByID_offset;


	for(size_t y=0;y<height;y++)
		for(size_t x=0;x<width;x++)
		{
			int idx = x + y*width;

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
inline float DepthMap::doLineStereo(
	const float u, const float v, const float epxn, const float epyn,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
////	
//	if(enableprintdebuginfo) stats->num_stereo_calls++;
//
//	// calculate epipolar line start and end point in old image
//	vec3 kinvp = model->pixeltocam(vec2(u, v), 1.0f);
//	eigen::vector3f pinf = referenceframe->k_othertothis_r * kinvp;
//	eigen::vector3f preal = pinf / prior_idepth + referenceframe->k_othertothis_t;
//
//	float rescalefactor = preal[2] * prior_idepth;
//
//	float firstx = u - 2*epxn*rescalefactor;
//	float firsty = v - 2*epyn*rescalefactor;
//	float lastx = u + 2*epxn*rescalefactor;
//	float lasty = v + 2*epyn*rescalefactor;
//	// width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
//	if (firstx <= 0 || firstx >= width - 2
//		|| firsty <= 0 || firsty >= height - 2
//		|| lastx <= 0 || lastx >= width - 2
//		|| lasty <= 0 || lasty >= height - 2) {
//		return -1;
//	}
//
//	if(!(rescalefactor > 0.7f && rescalefactor < 1.4f))
//	{
//		if(enableprintdebuginfo) stats->num_stereo_rescale_oob++;
//		return -1;
//	}
//
//	// calculate values to search for
//	float realval_p1 = getinterpolatedelement(activekeyframeimagedata,u + epxn*rescalefactor, v + epyn*rescalefactor, width);
//	float realval_m1 = getinterpolatedelement(activekeyframeimagedata,u - epxn*rescalefactor, v - epyn*rescalefactor, width);
//	float realval = getinterpolatedelement(activekeyframeimagedata,u, v, width);
//	float realval_m2 = getinterpolatedelement(activekeyframeimagedata,u - 2*epxn*rescalefactor, v - 2*epyn*rescalefactor, width);
//	float realval_p2 = getinterpolatedelement(activekeyframeimagedata,u + 2*epxn*rescalefactor, v + 2*epyn*rescalefactor, width);
//
////	if(referenceframe->k_othertothis_t[2] * max_idepth + pinf[2] < 0.01)
//
//
//	eigen::vector3f pclose = pinf + referenceframe->k_othertothis_t*max_idepth;
//	// if the assumed close-point lies behind the
//	// image, have to change that.
//	if(pclose[2] < 0.001f)
//	{
//		max_idepth = (0.001f-pinf[2]) / referenceframe->k_othertothis_t[2];
//		pclose = pinf + referenceframe->k_othertothis_t*max_idepth;
//	}
//	pclose = pclose / pclose[2]; // pos in new image of point (xy), assuming max_idepth
//
//	eigen::vector3f pfar = pinf + referenceframe->k_othertothis_t*min_idepth;
//	// if the assumed far-point lies behind the image or closter than the near-point,
//	// we moved past the point it and should stop.
//	if(pfar[2] < 0.001f || max_idepth < min_idepth)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_inf_oob++;
//		return -1;
//	}
//	pfar = pfar / pfar[2]; // pos in new image of point (xy), assuming min_idepth
//
//
//	// check for nan due to eg division by zero.
//	if(isnan((float)(pfar[0]+pclose[0])))
//		return -4;
//
//	// calculate increments in which we will step through the epipolar line.
//	// they are sampledist (or half sample dist) long
//	float incx = pclose[0] - pfar[0];
//	float incy = pclose[1] - pfar[1];
//	float epllength = sqrt(incx*incx+incy*incy);
//	if (!(epllength > 0) || std::isinf(epllength)) {
//		return -4;
//	}
//
//	if(epllength > max_epl_length_crop)
//	{
//		pclose[0] = pfar[0] + incx*max_epl_length_crop/epllength;
//		pclose[1] = pfar[1] + incy*max_epl_length_crop/epllength;
//	}
//
//	incx *= gradient_sample_dist/epllength;
//	incy *= gradient_sample_dist/epllength;
//
//
//	// extend one sample_dist to left & right.
//	pfar[0] -= incx;
//	pfar[1] -= incy;
//	pclose[0] += incx;
//	pclose[1] += incy;
//
//
//	// make epl long enough (pad a little bit).
//	if(epllength < min_epl_length_crop)
//	{
//		float pad = (min_epl_length_crop - (epllength)) / 2.0f;
//		pfar[0] -= incx*pad;
//		pfar[1] -= incy*pad;
//
//		pclose[0] += incx*pad;
//		pclose[1] += incy*pad;
//	}
//
//	// if inf point is outside of image: skip pixel.
//	if(
//			pfar[0] <= sample_point_to_border ||
//			pfar[0] >= width-sample_point_to_border ||
//			pfar[1] <= sample_point_to_border ||
//			pfar[1] >= height-sample_point_to_border)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_inf_oob++;
//		return -1;
//	}
//
//
//
//	// if near point is outside: move inside, and test length again.
//	if(
//			pclose[0] <= sample_point_to_border ||
//			pclose[0] >= width-sample_point_to_border ||
//			pclose[1] <= sample_point_to_border ||
//			pclose[1] >= height-sample_point_to_border)
//	{
//		if(pclose[0] <= sample_point_to_border)
//		{
//			float toadd = (sample_point_to_border - pclose[0]) / incx;
//			pclose[0] += toadd * incx;
//			pclose[1] += toadd * incy;
//		}
//		else if(pclose[0] >= width-sample_point_to_border)
//		{
//			float toadd = (width-sample_point_to_border - pclose[0]) / incx;
//			pclose[0] += toadd * incx;
//			pclose[1] += toadd * incy;
//		}
//
//		if(pclose[1] <= sample_point_to_border)
//		{
//			float toadd = (sample_point_to_border - pclose[1]) / incy;
//			pclose[0] += toadd * incx;
//			pclose[1] += toadd * incy;
//		}
//		else if(pclose[1] >= height-sample_point_to_border)
//		{
//			float toadd = (height-sample_point_to_border - pclose[1]) / incy;
//			pclose[0] += toadd * incx;
//			pclose[1] += toadd * incy;
//		}
//
//		// get new epl length
//		float fincx = pclose[0] - pfar[0];
//		float fincy = pclose[1] - pfar[1];
//		float newepllength = sqrt(fincx*fincx+fincy*fincy);
//
//		// test again
//		if(
//				pclose[0] <= sample_point_to_border ||
//				pclose[0] >= width-sample_point_to_border ||
//				pclose[1] <= sample_point_to_border ||
//				pclose[1] >= height-sample_point_to_border ||
//				newepllength < 8.0f
//				)
//		{
//			if(enableprintdebuginfo) stats->num_stereo_near_oob++;
//			return -1;
//		}
//
//
//	}
//
//
//	// from here on:
//	// - pinf: search start-point
//	// - p0: search end-point
//	// - incx, incy: search steps in pixel
//	// - epllength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.
//
//
//	float cpx = pfar[0];
//	float cpy =  pfar[1];
//
//	float val_cp_m2 = getinterpolatedelement(referenceframeimage,cpx-2.0f*incx, cpy-2.0f*incy, width);
//	float val_cp_m1 = getinterpolatedelement(referenceframeimage,cpx-incx, cpy-incy, width);
//	float val_cp = getinterpolatedelement(referenceframeimage,cpx, cpy, width);
//	float val_cp_p1 = getinterpolatedelement(referenceframeimage,cpx+incx, cpy+incy, width);
//	float val_cp_p2;
//
//
//
//	/*
//	 * subsequent exact minimum is found the following way:
//	 * - assuming lin. interpolation, the gradient of error at p1 (towards p2) is given by
//	 *   de1 = -2sum(e1*e1 - e1*e2)
//	 *   where e1 and e2 are summed over, and are the residuals (not squared).
//	 *
//	 * - the gradient at p2 (coming from p1) is given by
//	 * 	 de2 = +2sum(e2*e2 - e1*e2)
//	 *
//	 * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
//	 *   p1 + d*(p2-p1) with d = -de1 / (-de1 + de2).
//	 *
//	 *
//	 *
//	 * => i for later exact min calculation, i need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
//	 *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
//	 *    where i is the respective winning index.
//	 */
//
//
//	// walk in equally sized steps, starting at depth=infinity.
//	int loopcounter = 0;
//	float best_match_x = -1;
//	float best_match_y = -1;
//	float best_match_err = flt_max;
//	float second_best_match_err = flt_max;
//
//	// best pre and post errors.
//	float best_match_errpre=nan, best_match_errpost=nan, best_match_differrpre=nan, best_match_differrpost=nan;
//	bool bestwaslastloop = false;
//
//	float eelast = -1; // final error of last comp.
//
//	// alternating intermediate vars
//	float e1a=nan, e1b=nan, e2a=nan, e2b=nan, e3a=nan, e3b=nan, e4a=nan, e4b=nan, e5a=nan, e5b=nan;
//
//	int loopcbest=-1, loopcsecond =-1;
//	while(((incx < 0) == (cpx > pclose[0]) && (incy < 0) == (cpy > pclose[1])) || loopcounter == 0)
//	{
//		// interpolate one new point
//		val_cp_p2 = getinterpolatedelement(referenceframeimage,cpx+2*incx, cpy+2*incy, width);
//
//
//		// hacky but fast way to get error and differential error: switch buffer variables for last loop.
//		float ee = 0;
//		if(loopcounter%2==0)
//		{
//			// calc error and accumulate sums.
//			e1a = val_cp_p2 - realval_p2;ee += e1a*e1a;
//			e2a = val_cp_p1 - realval_p1;ee += e2a*e2a;
//			e3a = val_cp - realval;      ee += e3a*e3a;
//			e4a = val_cp_m1 - realval_m1;ee += e4a*e4a;
//			e5a = val_cp_m2 - realval_m2;ee += e5a*e5a;
//		}
//		else
//		{
//			// calc error and accumulate sums.
//			e1b = val_cp_p2 - realval_p2;ee += e1b*e1b;
//			e2b = val_cp_p1 - realval_p1;ee += e2b*e2b;
//			e3b = val_cp - realval;      ee += e3b*e3b;
//			e4b = val_cp_m1 - realval_m1;ee += e4b*e4b;
//			e5b = val_cp_m2 - realval_m2;ee += e5b*e5b;
//		}
//
//
//		// do i have a new winner??
//		// if so: set.
//		if(ee < best_match_err)
//		{
//			// put to second-best
//			second_best_match_err=best_match_err;
//			loopcsecond = loopcbest;
//
//			// set best.
//			best_match_err = ee;
//			loopcbest = loopcounter;
//
//			best_match_errpre = eelast;
//			best_match_differrpre = e1a*e1b + e2a*e2b + e3a*e3b + e4a*e4b + e5a*e5b;
//			best_match_errpost = -1;
//			best_match_differrpost = -1;
//
//			best_match_x = cpx;
//			best_match_y = cpy;
//			bestwaslastloop = true;
//		}
//		// otherwise: the last might be the current winner, in which case i have to save these values.
//		else
//		{
//			if(bestwaslastloop)
//			{
//				best_match_errpost = ee;
//				best_match_differrpost = e1a*e1b + e2a*e2b + e3a*e3b + e4a*e4b + e5a*e5b;
//				bestwaslastloop = false;
//			}
//
//			// collect second-best:
//			// just take the best of all that are not equal to current best.
//			if(ee < second_best_match_err)
//			{
//				second_best_match_err=ee;
//				loopcsecond = loopcounter;
//			}
//		}
//
//
//		// shift everything one further.
//		eelast = ee;
//		val_cp_m2 = val_cp_m1; val_cp_m1 = val_cp; val_cp = val_cp_p1; val_cp_p1 = val_cp_p2;
//
//		if(enableprintdebuginfo) stats->num_stereo_comparisons++;
//
//		cpx += incx;
//		cpy += incy;
//
//		loopcounter++;
//	}
//
//	// if error too big, will return -3, otherwise -2.
//	if(best_match_err > 4.0f*(float)max_error_stereo)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_invalid_bigerr++;
//		return -3;
//	}
//
//
//	// check if clear enough winner
//	if(abs(loopcbest - loopcsecond) > 1.0f && min_distance_error_stereo * best_match_err > second_best_match_err)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_invalid_unclear_winner++;
//		return -2;
//	}
//
//	bool didsubpixel = false;
//	if(usesubpixelstereo)
//	{
//		// ================== compute exact match =========================
//		// compute gradients (they are actually only half the real gradient)
//		float gradpre_pre = -(best_match_errpre - best_match_differrpre);
//		float gradpre_this = +(best_match_err - best_match_differrpre);
//		float gradpost_this = -(best_match_err - best_match_differrpost);
//		float gradpost_post = +(best_match_errpost - best_match_differrpost);
//
//		// final decisions here.
//		bool interppost = false;
//		bool interppre = false;
//
//		// if one is oob: return false.
//		if(enableprintdebuginfo && (best_match_errpre < 0 || best_match_errpost < 0))
//		{
//			stats->num_stereo_invalid_atend++;
//		}
//
//
//		// - if zero-crossing occurs exactly in between (gradient inconsistent),
//		else if((gradpost_this < 0) ^ (gradpre_this < 0))
//		{
//			// return exact pos, if both central gradients are small compared to their counterpart.
//			if(enableprintdebuginfo && (gradpost_this*gradpost_this > 0.1f*0.1f*gradpost_post*gradpost_post ||
//			   gradpre_this*gradpre_this > 0.1f*0.1f*gradpre_pre*gradpre_pre))
//				stats->num_stereo_invalid_inexistantcrossing++;
//		}
//
//		// if pre has zero-crossing
//		else if((gradpre_pre < 0) ^ (gradpre_this < 0))
//		{
//			// if post has zero-crossing
//			if((gradpost_post < 0) ^ (gradpost_this < 0))
//			{
//				if(enableprintdebuginfo) stats->num_stereo_invalid_twocrossing++;
//			}
//			else
//				interppre = true;
//		}
//
//		// if post has zero-crossing
//		else if((gradpost_post < 0) ^ (gradpost_this < 0))
//		{
//			interppost = true;
//		}
//
//		// if none has zero-crossing
//		else
//		{
//			if(enableprintdebuginfo) stats->num_stereo_invalid_nocrossing++;
//		}
//
//		// do interpolation!
//		// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
//		// the error at that point is also computed by just integrating.
//		if(interppre)
//		{
//			float d = gradpre_this / (gradpre_this - gradpre_pre);
//			best_match_x -= d*incx;
//			best_match_y -= d*incy;
//			best_match_err = best_match_err - 2*d*gradpre_this - (gradpre_pre - gradpre_this)*d*d;
//			if(enableprintdebuginfo) stats->num_stereo_interppre++;
//			didsubpixel = true;
//
//		}
//		else if(interppost)
//		{
//			float d = gradpost_this / (gradpost_this - gradpost_post);
//			best_match_x += d*incx;
//			best_match_y += d*incy;
//			best_match_err = best_match_err + 2*d*gradpost_this + (gradpost_post - gradpost_this)*d*d;
//			if(enableprintdebuginfo) stats->num_stereo_interppost++;
//			didsubpixel = true;
//		}
//		else
//		{
//			if(enableprintdebuginfo) stats->num_stereo_interpnone++;
//		}
//	}
//
//	// sampledist is the distance in pixel at which the realval's were sampled
//	float sampledist = gradient_sample_dist*rescalefactor;
//
//	float gradalongline = 0;
//	float tmp = realval_p2 - realval_p1;  gradalongline+=tmp*tmp;
//	tmp = realval_p1 - realval;  gradalongline+=tmp*tmp;
//	tmp = realval - realval_m1;  gradalongline+=tmp*tmp;
//	tmp = realval_m1 - realval_m2;  gradalongline+=tmp*tmp;
//
//	gradalongline /= sampledist*sampledist;
//
//	// check if interpolated error is ok. use evil hack to allow more error if there is a lot of gradient.
//	if(best_match_err > (float)max_error_stereo + sqrtf( gradalongline)*20)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_invalid_bigerr++;
//		return -3;
//	}
//
//
//	// ================= calc depth (in kf) ====================
//	// * kinvp = kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the kf.
//	// * best_match_x = x-coordinate of found correspondence in the reference frame.
//
//	float idnew_best_match;	// depth in the new image
//	float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
//	vec2 old = model->camtopixel(vec3(best_match_x, best_match_y, 1.f));
//	if(incx*incx>incy*incy)
//	{
//		float oldx = old[0];
//		float nominator = (oldx*referenceframe->othertothis_t[2] 
//			- referenceframe->othertothis_t[0]);
//		float dot0 = kinvp.dot(referenceframe->othertothis_r_row0);
//		float dot2 = kinvp.dot(referenceframe->othertothis_r_row2);
//
//		idnew_best_match = (dot0 - oldx*dot2) / nominator;
//		alpha = incx*fxi*(dot0*referenceframe->othertothis_t[2] 
//			- dot2*referenceframe->othertothis_t[0]) / (nominator*nominator);
//
//	}
//	else
//	{
//		float oldy = old[1];
//
//		float nominator = (oldy*referenceframe->othertothis_t[2] - referenceframe->othertothis_t[1]);
//		float dot1 = kinvp.dot(referenceframe->othertothis_r_row1);
//		float dot2 = kinvp.dot(referenceframe->othertothis_r_row2);
//
//		idnew_best_match = (dot1 - oldy*dot2) / nominator;
//		alpha = incy*fyi*(dot1*referenceframe->othertothis_t[2] - dot2*referenceframe->othertothis_t[1]) / (nominator*nominator);
//
//	}
//
//
//
//
//
//	if(idnew_best_match < 0)
//	{
//		if(enableprintdebuginfo) stats->num_stereo_negative++;
//		if(!allownegativeidepths)
//			return -2;
//	}
//
//	if(enableprintdebuginfo) stats->num_stereo_successfull++;
//
//	// ================= calc var (in new image) ====================
//
//	// calculate error from photometric noise
//	float photodisperror = 4.0f * camerapixelnoise2 / (gradalongline + division_eps);
//
//	float trackingerrorfac = 0.25f*(1.0f+referenceframe->initialtrackedresidual);
//
//	// calculate error from geometric noise (wrong camera pose / calibration)
//	eigen::vector2f gradsinterp = getinterpolatedelement42(activekeyframe->gradients(0), u, v, width);
//	float geodisperror = (gradsinterp[0]*epxn + gradsinterp[1]*epyn) + division_eps;
//	geodisperror = trackingerrorfac*trackingerrorfac*(gradsinterp[0]*gradsinterp[0] + gradsinterp[1]*gradsinterp[1]) / (geodisperror*geodisperror);
//
//
//	//geodisperror *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5 *result_idepth);
//
//	// final error consists of a small constant part (discretization error),
//	// geometric and photometric error.
//	result_var = alpha*alpha*((didsubpixel ? 0.05f : 0.5f)*sampledist*sampledist +  geodisperror + photodisperror);	// square to make variance
//
//	if(plotstereoimages)
//	{
//		if(rand()%5==0)
//		{
//			//if(rand()%500 == 0)
//			//	printf("geo: %f, photo: %f, alpha: %f\n", sqrt(geodisperror), sqrt(photodisperror), alpha, sqrt(result_var));
//
//
//			//int iddiff = (keyframe->pyramidid - referenceframe->id);
//			//cv::scalar color = cv::scalar(0,0, 2*iddiff);// bw
//
//			//cv::scalar color = cv::scalar(sqrt(result_var)*2000, 255-sqrt(result_var)*2000, 0);// bw
//
////			float epllengthf = std::min((float)min_epl_length_crop,(float)epllength);
////			epllengthf = std::max((float)max_epl_length_crop,(float)epllengthf);
////
////			float pixeldistfound = sqrtf((float)((preal[0]/preal[2] - best_match_x)*(preal[0]/preal[2] - best_match_x)
////					+ (preal[1]/preal[2] - best_match_y)*(preal[1]/preal[2] - best_match_y)));
////
//			float fac = best_match_err / ((float)max_error_stereo + sqrtf( gradalongline)*20);
//
//			cv::scalar color = cv::scalar(255*fac, 255-255*fac, 0);// bw
//
//
//			/*
//			if(rescalefactor > 1)
//				color = cv::scalar(500*(rescalefactor-1),0,0);
//			else
//				color = cv::scalar(0,500*(1-rescalefactor),500*(1-rescalefactor));
//			*/
//
//			cv::line(debugimagestereolines,cv::point2f(pclose[0], pclose[1]),cv::point2f(pfar[0], pfar[1]),color,1,8,0);
//		}
//	}
//
//	result_idepth = idnew_best_match;
//
//	result_epllength = epllength;
//
//	return best_match_err;
//	*/
	return 0.f;
}

}
