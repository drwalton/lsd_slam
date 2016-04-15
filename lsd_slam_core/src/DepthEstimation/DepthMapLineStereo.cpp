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

///This file contains the depth estimation functions specific to Projective
/// camera models.

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
#include "ProjCameraModel.hpp"

namespace lsd_slam
{

///\brief Once an initial estimate for the best match has been found, 
///       this method can be used to refine its location to subpixel accuracy.
///\note This version of the method should be applied to PROJ camera models only.
///\note No interpolation is performed if no zero crossing could be found in
///      the surrounding gradients.
///\param[in,out] best_match_x The intial estimate for the best match location,
///               which will be refined by this function.
///\param[in,out] best_match_y The intial estimate for the best match location,
///               which will be refined by this function.
///\param[in,out] best_match_err The initial estimate for the best match error,
///               which will be refined by this function.
///\param best_match_errPre
///\param best_match_bool 
///\return True if subpixel matching was actually performed.
bool subpixelMatchProj(
	float *best_match_x, float *best_match_y, float *best_match_err,
	float best_match_errPre, float best_match_DiffErrPre,
	float best_match_errPost, float best_match_DiffErrPost,
	float incx, float incy,
	RunningStats *stats);

///\brief Given a match location and error, estimate the inverse depth and 
///       variance.
///\note This version of the method should be applied to PROJ camera models only.
float findDepthAndVarProj(
	float *result_idepth, float *result_var,
	const float u, const float v,
	const float epxn, const float epyn,
	const float best_match_x, const float best_match_y,
	const float best_match_err,
	const float incx, const float incy,
	const bool didSubpixel,
	const vec3 &KinvP,
	const vec3 &pClose, const vec3 &pFar,
	const float gradAlongLine,
	const float sampleDist,
	const ProjCameraModel *model,
	const Frame *referenceFrame,
	Frame *activeKeyFrame,
	RunningStats *stats,
	cv::Mat &debugImageStereoLines);


bool DepthMap::makeAndCheckEPL(const int x, const int y, const Frame* const ref,
	float* pepx, float* pepy, RunningStats* const stats)
{
	int idx = x+y*width;

	// ======= make epl ========
	float epx = - model->fx * ref->thisToOther_t[0] + ref->thisToOther_t[2]*(x - model->cx);
	float epy = - model->fy * ref->thisToOther_t[1] + ref->thisToOther_t[2]*(y - model->cy);

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

// find pixel in image (do stereo along epipolar line).
// mat: NEW image
// KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
// trafo: x_old = trafo * x_new; (from new to old image)
// realVal: descriptor in OLD image.
// returns: result_idepth : point depth in new camera's coordinate system
// returns: result_u/v : point's coordinates in new camera's coordinate system
// returns: idepth_var: (approximated) measurement variance of inverse depth of result_point_NEW
// returns error if sucessful; -1 if out of bounds, -2 if not found.
float DepthMap::doLineStereo(
	const float u, const float v, const float epxn, const float epyn,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const Frame* const referenceFrame, const float* referenceFrameImage,
	float &result_idepth, float &result_var, float &result_eplLength,
	RunningStats* stats)
{
	ProjCameraModel &pModel = static_cast<ProjCameraModel&>(*model);
	if (enablePrintDebugInfo) stats->num_stereo_calls++;
	vec3 K_otherToThis_t = model->camToPixelDepth(referenceFrame->otherToThis_t);

	// calculate epipolar line start and end point in old image
	vec3 KinvP = model->pixelToCam(vec2(u, v), 1.0f);
	Eigen::Vector3f pInf = model->camToPixelDepth(referenceFrame->otherToThis_R * KinvP);
	Eigen::Vector3f pReal = pInf / prior_idepth + K_otherToThis_t;

	float rescaleFactor = pReal[2] * prior_idepth;

	float firstX = u - 2 * epxn*rescaleFactor;
	float firstY = v - 2 * epyn*rescaleFactor;
	float lastX = u + 2 * epxn*rescaleFactor;
	float lastY = v + 2 * epyn*rescaleFactor;
	// width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
	if (firstX <= 0 || firstX >= width - 2
		|| firstY <= 0 || firstY >= height - 2
		|| lastX <= 0 || lastX >= width - 2
		|| lastY <= 0 || lastY >= height - 2) {
		return -1;
	}

	if (!(rescaleFactor > 0.7f && rescaleFactor < 1.4f))
	{
		if (enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
		return -1;
	}

	// calculate values to search for
	float realVal_p1 = getInterpolatedElement(activeKeyFrameImageData, u + epxn*rescaleFactor, v + epyn*rescaleFactor, width);
	float realVal_m1 = getInterpolatedElement(activeKeyFrameImageData, u - epxn*rescaleFactor, v - epyn*rescaleFactor, width);
	float realVal = getInterpolatedElement(activeKeyFrameImageData, u, v, width);
	float realVal_m2 = getInterpolatedElement(activeKeyFrameImageData, u - 2 * epxn*rescaleFactor, v - 2 * epyn*rescaleFactor, width);
	float realVal_p2 = getInterpolatedElement(activeKeyFrameImageData, u + 2 * epxn*rescaleFactor, v + 2 * epyn*rescaleFactor, width);

	//	if(referenceFrame->K_otherToThis_t[2] * max_idepth + pInf[2] < 0.01)

	Eigen::Vector3f pClose = pInf + K_otherToThis_t*max_idepth;
	// if the assumed close-point lies behind the
	// image, have to change that.
	if (pClose[2] < 0.001f)
	{
		max_idepth = (0.001f - pInf[2]) / K_otherToThis_t[2];
		pClose = pInf + K_otherToThis_t*max_idepth;
	}
	pClose = pClose / pClose[2]; // pos in new image of point (xy), assuming max_idepth

	Eigen::Vector3f pFar = pInf + K_otherToThis_t*min_idepth;
	// if the assumed far-point lies behind the image or closter than the near-point,
	// we moved past the Point it and should stop.
	if (pFar[2] < 0.001f || max_idepth < min_idepth)
	{
		if (enablePrintDebugInfo) stats->num_stereo_inf_oob++;
		return -1;
	}
	pFar = pFar / pFar[2]; // pos in new image of point (xy), assuming min_idepth


	// check for nan due to eg division by zero.
	if (isnan((float)(pFar[0] + pClose[0])))
		return -4;

	// calculate increments in which we will step through the epipolar line.
	// they are sampleDist (or half sample dist) long
	float incx = pClose[0] - pFar[0];
	float incy = pClose[1] - pFar[1];
	float eplLength = sqrt(incx*incx + incy*incy);
	if (!(eplLength > 0) || std::isinf(eplLength)) {
		return -4;
	}

	if (eplLength > MAX_EPL_LENGTH_CROP)
	{
		pClose[0] = pFar[0] + incx*MAX_EPL_LENGTH_CROP / eplLength;
		pClose[1] = pFar[1] + incy*MAX_EPL_LENGTH_CROP / eplLength;
	}

	incx *= GRADIENT_SAMPLE_DIST / eplLength;
	incy *= GRADIENT_SAMPLE_DIST / eplLength;


	// extend one sample_dist to left & right.
	pFar[0] -= incx;
	pFar[1] -= incy;
	pClose[0] += incx;
	pClose[1] += incy;


	// make epl long enough (pad a little bit).
	if (eplLength < MIN_EPL_LENGTH_CROP)
	{
		float pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2.0f;
		pFar[0] -= incx*pad;
		pFar[1] -= incy*pad;

		pClose[0] += incx*pad;
		pClose[1] += incy*pad;
	}

	// if inf point is outside of image: skip pixel.
	if (
		pFar[0] <= SAMPLE_POINT_TO_BORDER ||
		pFar[0] >= width - SAMPLE_POINT_TO_BORDER ||
		pFar[1] <= SAMPLE_POINT_TO_BORDER ||
		pFar[1] >= height - SAMPLE_POINT_TO_BORDER)
	{
		if (enablePrintDebugInfo) stats->num_stereo_inf_oob++;
		return -1;
	}

	// if near point is outside: move inside, and test length again.
	if (
		pClose[0] <= SAMPLE_POINT_TO_BORDER ||
		pClose[0] >= width - SAMPLE_POINT_TO_BORDER ||
		pClose[1] <= SAMPLE_POINT_TO_BORDER ||
		pClose[1] >= height - SAMPLE_POINT_TO_BORDER)
	{
		if (pClose[0] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		} else if (pClose[0] >= width - SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (width - SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		if (pClose[1] <= SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		} else if (pClose[1] >= height - SAMPLE_POINT_TO_BORDER)
		{
			float toAdd = (height - SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
			pClose[0] += toAdd * incx;
			pClose[1] += toAdd * incy;
		}

		// get new epl length
		float fincx = pClose[0] - pFar[0];
		float fincy = pClose[1] - pFar[1];
		float newEplLength = sqrt(fincx*fincx + fincy*fincy);

		// test again
		if (
			pClose[0] <= SAMPLE_POINT_TO_BORDER ||
			pClose[0] >= width - SAMPLE_POINT_TO_BORDER ||
			pClose[1] <= SAMPLE_POINT_TO_BORDER ||
			pClose[1] >= height - SAMPLE_POINT_TO_BORDER ||
			newEplLength < 8.0f
			)
		{
			if (enablePrintDebugInfo) stats->num_stereo_near_oob++;
			return -1;
		}


	}


	// from here on:
	// - pInf: search start-point
	// - p0: search end-point
	// - incx, incy: search steps in pixel
	// - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.


	float cpx = pFar[0];
	float cpy = pFar[1];

	float val_cp_m2 = getInterpolatedElement(referenceFrameImage, cpx - 2.0f*incx, cpy - 2.0f*incy, width);
	float val_cp_m1 = getInterpolatedElement(referenceFrameImage, cpx - incx, cpy - incy, width);
	float val_cp = getInterpolatedElement(referenceFrameImage, cpx, cpy, width);
	float val_cp_p1 = getInterpolatedElement(referenceFrameImage, cpx + incx, cpy + incy, width);
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
	float best_match_errPre = NAN, best_match_errPost = NAN, best_match_DiffErrPre = NAN, best_match_DiffErrPost = NAN;
	bool bestWasLastLoop = false;

	float eeLast = -1; // final error of last comp.

	// alternating intermediate vars
	float e1A = NAN, e1B = NAN, e2A = NAN, e2B = NAN, e3A = NAN, e3B = NAN, e4A = NAN, e4B = NAN, e5A = NAN, e5B = NAN;

	int loopCBest = -1, loopCSecond = -1;
	while (((incx < 0) == (cpx > pClose[0]) && (incy < 0) == (cpy > pClose[1])) || loopCounter == 0)
	{
		// interpolate one new point
		val_cp_p2 = getInterpolatedElement(referenceFrameImage, cpx + 2 * incx, cpy + 2 * incy, width);


		// hacky but fast way to get error and differential error: switch buffer variables for last loop.
		float ee = 0;
		if (loopCounter % 2 == 0)
		{
			// calc error and accumulate sums.
			e1A = val_cp_p2 - realVal_p2; ee += e1A*e1A;
			e2A = val_cp_p1 - realVal_p1; ee += e2A*e2A;
			e3A = val_cp - realVal;      ee += e3A*e3A;
			e4A = val_cp_m1 - realVal_m1; ee += e4A*e4A;
			e5A = val_cp_m2 - realVal_m2; ee += e5A*e5A;
		} else
		{
			// calc error and accumulate sums.
			e1B = val_cp_p2 - realVal_p2; ee += e1B*e1B;
			e2B = val_cp_p1 - realVal_p1; ee += e2B*e2B;
			e3B = val_cp - realVal;      ee += e3B*e3B;
			e4B = val_cp_m1 - realVal_m1; ee += e4B*e4B;
			e5B = val_cp_m2 - realVal_m2; ee += e5B*e5B;
		}


		// do I have a new winner??
		// if so: set.
		if (ee < best_match_err)
		{
			// put to second-best
			second_best_match_err = best_match_err;
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
			if (bestWasLastLoop)
			{
				best_match_errPost = ee;
				best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
				bestWasLastLoop = false;
			}

			// collect second-best:
			// just take the best of all that are NOT equal to current best.
			if (ee < second_best_match_err)
			{
				second_best_match_err = ee;
				loopCSecond = loopCounter;
			}
		}


		// shift everything one further.
		eeLast = ee;
		val_cp_m2 = val_cp_m1; val_cp_m1 = val_cp; val_cp = val_cp_p1; val_cp_p1 = val_cp_p2;

		if (enablePrintDebugInfo) stats->num_stereo_comparisons++;

		cpx += incx;
		cpy += incy;

		loopCounter++;
	}

	// if error too big, will return -3, otherwise -2.
	if (best_match_err > 4.0f*(float)MAX_ERROR_STEREO)
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}


	// check if clear enough winner
	if (abs(loopCBest - loopCSecond) > 1.0f && MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err)
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
		return -2;
	}

	bool didSubpixel = false;
	if (useSubpixelStereo)
	{
		didSubpixel = subpixelMatchProj(&best_match_x, &best_match_y, &best_match_err,
			best_match_errPre, best_match_DiffErrPre,
			best_match_errPost, best_match_DiffErrPost,
			incx, incy, stats);
	}


	// sampleDist is the distance in pixel at which the realVal's were sampled
	float sampleDist = GRADIENT_SAMPLE_DIST*rescaleFactor;

	float gradAlongLine = 0;
	float tmp = realVal_p2 - realVal_p1;  gradAlongLine += tmp*tmp;
	tmp = realVal_p1 - realVal;  gradAlongLine += tmp*tmp;
	tmp = realVal - realVal_m1;  gradAlongLine += tmp*tmp;
	tmp = realVal_m1 - realVal_m2;  gradAlongLine += tmp*tmp;

	gradAlongLine /= sampleDist*sampleDist;

	// check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
	if (best_match_err > (float)MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20)
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
		return -3;
	}

	float r = findDepthAndVarProj(&result_idepth, &result_var, u, v, epxn,
		epyn, best_match_x, best_match_y, best_match_err,
		incx, incy, didSubpixel, KinvP, pClose, pFar,
		gradAlongLine, sampleDist, &pModel, referenceFrame, activeKeyFrame,
		stats, debugImageStereoLines);
	if (r != 0.f) return r;

	result_eplLength = eplLength;
	return best_match_err;
}

bool subpixelMatchProj(
	float *best_match_x, float *best_match_y, float *best_match_err,
	float best_match_errPre, float best_match_DiffErrPre,
	float best_match_errPost, float best_match_DiffErrPost,
	float incx, float incy,
	RunningStats *stats)
{
	bool didSubpixel = false;

	// ================== compute exact match =========================
	// compute gradients (they are actually only half the real gradient)
	float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
	float gradPre_this = +(*best_match_err - best_match_DiffErrPre);
	float gradPost_this = -(*best_match_err - best_match_DiffErrPost);
	float gradPost_post = +(best_match_errPost - best_match_DiffErrPost);

	// final decisions here.
	bool interpPost = false;
	bool interpPre = false;

	// if one is oob: return false.
	if (enablePrintDebugInfo && (best_match_errPre < 0 || best_match_errPost < 0))
	{
		stats->num_stereo_invalid_atEnd++;
	}


	// - if zero-crossing occurs exactly in between (gradient Inconsistent),
	else if ((gradPost_this < 0) ^ (gradPre_this < 0))
	{
		// return exact pos, if both central gradients are small compared to their counterpart.
		if (enablePrintDebugInfo && (gradPost_this*gradPost_this > 0.1f*0.1f*gradPost_post*gradPost_post ||
			gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre))
			stats->num_stereo_invalid_inexistantCrossing++;
	}

	// if pre has zero-crossing
	else if ((gradPre_pre < 0) ^ (gradPre_this < 0))
	{
		// if post has zero-crossing
		if ((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			if (enablePrintDebugInfo) stats->num_stereo_invalid_twoCrossing++;
		} else
			interpPre = true;
	}

	// if post has zero-crossing
	else if ((gradPost_post < 0) ^ (gradPost_this < 0))
	{
		interpPost = true;
	}

	// if none has zero-crossing
	else
	{
		if (enablePrintDebugInfo) stats->num_stereo_invalid_noCrossing++;
	}


	// DO interpolation!
	// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
	// the error at that point is also computed by just integrating.
	if (interpPre)
	{
		float d = gradPre_this / (gradPre_this - gradPre_pre);
		*best_match_x -= d*incx;
		*best_match_y -= d*incy;
		*best_match_err = *best_match_err - 2 * d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
		if (enablePrintDebugInfo) stats->num_stereo_interpPre++;
		didSubpixel = true;

	} else if (interpPost)
	{
		float d = gradPost_this / (gradPost_this - gradPost_post);
		*best_match_x += d*incx;
		*best_match_y += d*incy;
		*best_match_err = *best_match_err + 2 * d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
		if (enablePrintDebugInfo) stats->num_stereo_interpPost++;
		didSubpixel = true;
	} else
	{
		if (enablePrintDebugInfo) stats->num_stereo_interpNone++;
	}

	return didSubpixel;
}

float findDepthAndVarProj(
	float *result_idepth, float *result_var,
	const float u, const float v,
	const float epxn, const float epyn,
	const float best_match_x, const float best_match_y,
	const float best_match_err,
	const float incx, const float incy,
	const bool didSubpixel,
	const vec3 &KinvP,
	const vec3 &pClose, const vec3 &pFar,
	const float gradAlongLine,
	const float sampleDist,
	const ProjCameraModel *model,
	const Frame *referenceFrame,
	Frame *activeKeyFrame,
	RunningStats *stats,
	cv::Mat &debugImageStereoLines)
{

	// ================= calc depth (in KF) ====================
	// * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the KF.
	// * best_match_x = x-coordinate of found correspondence in the reference frame.

	float idnew_best_match;	// depth in the new image
	float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
	vec2 old = model->camToPixel(vec3(best_match_x, best_match_y, 1.f));
	if (incx*incx > incy*incy)
	{
		float oldX = old[0];
		float nominator = (oldX*referenceFrame->otherToThis_t[2]
			- referenceFrame->otherToThis_t[0]);
		float dot0 = KinvP.dot(referenceFrame->otherToThis_R_row0);
		float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

		idnew_best_match = (dot0 - oldX*dot2) / nominator;
		alpha = incx*model->fxi()*(dot0*referenceFrame->otherToThis_t[2]
			- dot2*referenceFrame->otherToThis_t[0]) / (nominator*nominator);
	} else
	{
		float oldY = old[1];

		float nominator = (oldY*referenceFrame->otherToThis_t[2] - referenceFrame->otherToThis_t[1]);
		float dot1 = KinvP.dot(referenceFrame->otherToThis_R_row1);
		float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);

		idnew_best_match = (dot1 - oldY*dot2) / nominator;
		alpha = incy*model->fyi()*(dot1*referenceFrame->otherToThis_t[2] - dot2*referenceFrame->otherToThis_t[1]) / (nominator*nominator);
	}

	if (idnew_best_match < 0)
	{
		if (enablePrintDebugInfo) stats->num_stereo_negative++;
		if (!allowNegativeIdepths)
			return -2;
	}

	if (enablePrintDebugInfo) stats->num_stereo_successfull++;

	// ================= calc var (in NEW image) ====================

	// calculate error from photometric noise
	float photoDispError = 4.0f * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);

	float trackingErrorFac = 0.25f*(1.0f + referenceFrame->initialTrackedResidual);

	// calculate error from geometric noise (wrong camera pose / calibration)
	Eigen::Vector2f gradsInterp = getInterpolatedElement42(activeKeyFrame->gradients(0), u, v, model->w);
	float geoDispError = (gradsInterp[0] * epxn + gradsInterp[1] * epyn) + DIVISION_EPS;
	geoDispError = trackingErrorFac*trackingErrorFac*(gradsInterp[0] * gradsInterp[0] + gradsInterp[1] * gradsInterp[1]) / (geoDispError*geoDispError);

	// final error consists of a small constant part (discretization error),
	// geometric and photometric error.
	*result_var = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist + geoDispError + photoDispError);	// square to make variance

	if (plotStereoImages)
	{
		if (rand() % 5 == 0)
		{
			float fac = best_match_err / ((float)MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20);

			cv::Scalar color = cv::Scalar(255 * fac, 255 - 255 * fac, 0);// bw

			cv::line(debugImageStereoLines, cv::Point2f(pClose[0], pClose[1]), cv::Point2f(pFar[0], pFar[1]), color, 1, 8, 0);
		}
	}

	*result_idepth = idnew_best_match;
	return 0.f;
}

}
