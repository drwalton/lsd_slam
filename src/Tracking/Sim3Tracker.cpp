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

#include "Sim3Tracker.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "DataStructures/Frame.hpp"
#include "Tracking/TrackingReference.hpp"
#include "Util/globalFuncs.hpp"
#include "IOWrapper/ImageDisplay.hpp"
#include "Tracking/LGSX.hpp"
#include "System/Win32Compatibility.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"

namespace lsd_slam
{


#if defined(ENABLE_NEON)
	#define callOptimized(function, arguments) function##NEON arguments
#else
	#if defined(ENABLE_SSE)
		#define callOptimized(function, arguments) (USESSE ? function##SSE arguments : function arguments)
	#else
		#define callOptimized(function, arguments) function arguments
	#endif
#endif



Sim3Tracker::Sim3Tracker(const CameraModel &model)
	//TODO TMP TEST
	:camModel(model.clone())
	//:camModel(model.makeOmniCamModel())
{
	int w = model.w, h = model.h;
	settings = DenseDepthTrackerSettings();

	buf_warped_residual = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_weights = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dx = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dy = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_x = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_y = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_z = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_residual_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_p = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_weight_Huber = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_VarP = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_VarD = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_warped_size = 0;

	debugImageWeights = cv::Mat(h,w,CV_8UC3);
	debugImageResiduals = cv::Mat(h,w,CV_8UC3);
	debugImageSecondFrame = cv::Mat(h,w,CV_8UC3);
	debugImageOldImageWarped = cv::Mat(h,w,CV_8UC3);
	debugImageOldImageSource = cv::Mat(h,w,CV_8UC3);
	debugImageExternalWeights = cv::Mat(h,w,CV_8UC3);
	debugImageDepthResiduals = cv::Mat(h,w,CV_8UC3);
	debugImageScaleEstimation = cv::Mat(h,w,CV_8UC3);

	debugImageHuberWeight = cv::Mat(h,w,CV_8UC3);
	debugImageWeightD = cv::Mat(h,w,CV_8UC3);
	debugImageWeightP = cv::Mat(h,w,CV_8UC3);
	debugImageWeightedResP = cv::Mat(h,w,CV_8UC3);
	debugImageWeightedResD = cv::Mat(h,w,CV_8UC3);

	
	lastResidual = 0;
	iterationNumber = 0;
	lastDepthResidual = lastPhotometricResidual = lastDepthResidualUnweighted = lastPhotometricResidualUnweighted = lastResidualUnweighted = 0;
	pointUsage = 0;

}

Sim3Tracker::~Sim3Tracker()
{
	debugImageResiduals.release();
	debugImageWeights.release();
	debugImageSecondFrame.release();
	debugImageOldImageSource.release();
	debugImageOldImageWarped.release();
	debugImageExternalWeights.release();
	debugImageDepthResiduals.release();
	debugImageScaleEstimation.release();

	debugImageHuberWeight.release();
	debugImageWeightD.release();
	debugImageWeightP.release();
	debugImageWeightedResP.release();
	debugImageWeightedResD.release();


	Eigen::internal::aligned_free((void*)buf_warped_residual);
	Eigen::internal::aligned_free((void*)buf_warped_weights);
	Eigen::internal::aligned_free((void*)buf_warped_dx);
	Eigen::internal::aligned_free((void*)buf_warped_dy);
	Eigen::internal::aligned_free((void*)buf_warped_x);
	Eigen::internal::aligned_free((void*)buf_warped_y);
	Eigen::internal::aligned_free((void*)buf_warped_z);

	Eigen::internal::aligned_free((void*)buf_d);
	Eigen::internal::aligned_free((void*)buf_residual_d);
	Eigen::internal::aligned_free((void*)buf_idepthVar);
	Eigen::internal::aligned_free((void*)buf_warped_idepthVar);
	Eigen::internal::aligned_free((void*)buf_weight_p);
	Eigen::internal::aligned_free((void*)buf_weight_d);

	Eigen::internal::aligned_free((void*)buf_weight_Huber);
	Eigen::internal::aligned_free((void*)buf_weight_VarP);
	Eigen::internal::aligned_free((void*)buf_weight_VarD);
}


Sim3 Sim3Tracker::trackFrameSim3(
		TrackingKeyframe* reference,
		Frame* frame,
		const Sim3& frameToReference_initialEstimate,
		int startLevel, int finalLevel)
{
	boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();

	diverged = false;


	affineEstimation_a = 1; affineEstimation_b = 0;


	// ============ track frame ============
    Sim3 referenceToFrame = frameToReference_initialEstimate.inverse();
	LGS7 ls7;


	int numCalcResidualCalls[PYRAMID_LEVELS];
	int numCalcWarpUpdateCalls[PYRAMID_LEVELS];

	Sim3ResidualStruct finalResidual;

	bool warp_update_up_to_date = false;

	for(int lvl=startLevel;lvl >= finalLevel;lvl--)
	{
		numCalcResidualCalls[lvl] = 0;
		numCalcWarpUpdateCalls[lvl] = 0;

		if(settings.maxItsPerLvl[lvl] == 0)
			continue;

		reference->makePointCloud(lvl);

		// evaluate baseline-residual.
		callOptimized(calcSim3Buffers, (reference, frame, referenceToFrame, lvl));
		if(buf_warped_size < 0.5 * MIN_GOODPERALL_PIXEL_ABSMIN * (camModel->w>>lvl)*(camModel->h>>lvl) || buf_warped_size < 10)
		{
			diverged = true;
			return Sim3();
		}

		Sim3ResidualStruct lastErr = callOptimized(calcSim3WeightsAndResidual,(referenceToFrame));
		if(plotSim3TrackingIterationInfo) callOptimized(calcSim3Buffers,(reference, frame, referenceToFrame, lvl, true));
		numCalcResidualCalls[lvl]++;

		if(useAffineLightningEstimation)
		{
			affineEstimation_a = affineEstimation_a_lastIt;
			affineEstimation_b = affineEstimation_b_lastIt;
		}

		float LM_lambda = settings.lambdaInitial[lvl];

		warp_update_up_to_date = false;
		for(int iteration=0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
		{

			// calculate LS System, result is saved in ls.
			callOptimized(calcSim3LGS,(ls7));
			warp_update_up_to_date = true;
			numCalcWarpUpdateCalls[lvl]++;

			iterationNumber = iteration;


			int incTry=0;
			while(true)
			{
				// solve LS system with current lambda
				Vector7 b = - ls7.b / static_cast<float>(ls7.num_constraints);
				Matrix7x7 A = ls7.A / static_cast<float>(ls7.num_constraints);
				for(int i=0;i<7;i++) A(i,i) *= 1+LM_lambda;
				Vector7 inc = A.ldlt().solve(b);
				incTry++;

				float absInc = inc.dot(inc);
				if(!(absInc >= 0 && absInc < 1))
				{
					// ERROR tracking diverged.
					lastSim3Hessian.setZero();
					return Sim3();
				}

				// apply increment. pretty sure this way round is correct, but hard to test.
				Sim3 new_referenceToFrame =Sim3::exp(inc.cast<sophusType>()) * referenceToFrame;
				//Sim3 new_referenceToFrame = referenceToFrame * Sim3::exp((inc));


				// re-evaluate residual
				callOptimized(calcSim3Buffers,(reference, frame, new_referenceToFrame, lvl));
				if(buf_warped_size < 0.5 * MIN_GOODPERALL_PIXEL_ABSMIN * (camModel->w>>lvl)*(camModel->h>>lvl) || buf_warped_size < 10)
				{
					diverged = true;
					return Sim3();
				}

				Sim3ResidualStruct error = callOptimized(calcSim3WeightsAndResidual,(new_referenceToFrame));
				if(plotSim3TrackingIterationInfo) callOptimized(calcSim3Buffers,(reference, frame, new_referenceToFrame, lvl, true));
				numCalcResidualCalls[lvl]++;


				// accept inc?
				if(error.mean < lastErr.mean)
				{
					// accept inc
					referenceToFrame = new_referenceToFrame;
					warp_update_up_to_date = false;

					if(useAffineLightningEstimation)
					{
						affineEstimation_a = affineEstimation_a_lastIt;
						affineEstimation_b = affineEstimation_b_lastIt;
					}

					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						// debug output
						printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);

						printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
								referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
								referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
								referenceToFrame.log()[6]);
					}

					// converged?
					if(error.mean / lastErr.mean > settings.convergenceEps[lvl])
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
					}

					finalResidual = lastErr = error;

					if(LM_lambda <= 0.2)
						LM_lambda = 0;
					else
						LM_lambda *= settings.lambdaSuccessFac;

					break;
				}
				else
				{
					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						printf("(%d-%d): REJECTED increment of %f with lambda %.1f, (residual: %f -> %f)\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr.mean, error.mean);
					}

					if(!(inc.dot(inc) > settings.stepSizeMin[lvl]))
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (stepsize too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
						break;
					}

					if(LM_lambda == 0)
						LM_lambda = 0.2f;
					else
						LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
				}
			}
		}
	}



	if(enablePrintDebugInfo && printTrackingIterationInfo)
	{
		printf("Tracking: ");
			for(int lvl=PYRAMID_LEVELS-1;lvl >= 0;lvl--)
			{
				printf("lvl %d: %d (%d); ",
					lvl,
					numCalcResidualCalls[lvl],
					numCalcWarpUpdateCalls[lvl]);
			}

		printf("\n");


		printf("pOld = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
				frameToReference_initialEstimate.inverse().log()[0],frameToReference_initialEstimate.inverse().log()[1],frameToReference_initialEstimate.inverse().log()[2],
				frameToReference_initialEstimate.inverse().log()[3],frameToReference_initialEstimate.inverse().log()[4],frameToReference_initialEstimate.inverse().log()[5],
				frameToReference_initialEstimate.inverse().log()[6]);
		printf("pNew = %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
				referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
				referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5],
				referenceToFrame.log()[6]);
		printf("final res mean: %f meanD %f, meanP %f\n", finalResidual.mean, finalResidual.meanD, finalResidual.meanP);
	}


	// Make sure that there is a warp update at the final position to get the correct information matrix
	if (!warp_update_up_to_date)
	{
		reference->makePointCloud(finalLevel);
		callOptimized(calcSim3Buffers,(reference, frame, referenceToFrame, finalLevel));
	    finalResidual = callOptimized(calcSim3WeightsAndResidual,(referenceToFrame));
	    callOptimized(calcSim3LGS,(ls7));
	}

	lastSim3Hessian = ls7.A;


	if(referenceToFrame.scale() <= 0 )
	{
		diverged = true;
		return Sim3();
	}

	lastResidual = finalResidual.mean;
	lastDepthResidual = finalResidual.meanD;
	lastPhotometricResidual = finalResidual.meanP;


	return referenceToFrame.inverse();
}

void Sim3Tracker::calcSim3Buffers(
		const TrackingKeyframe* reference,
		Frame* frame,
		const Sim3& referenceToFrame,
		int level, bool plotWeights)
{
	//TODO Omni Code
	if(plotSim3TrackingIterationInfo)
	{
		cv::Vec3b col = cv::Vec3b(255,170,168);
		fillCvMat(&debugImageResiduals,col);
		fillCvMat(&debugImageOldImageSource,col);
		fillCvMat(&debugImageOldImageWarped,col);
		fillCvMat(&debugImageDepthResiduals,col);
	}
	if(plotWeights && plotSim3TrackingIterationInfo)
	{
		cv::Vec3b col = cv::Vec3b(255,170,168);
		fillCvMat(&debugImageHuberWeight,col);
		fillCvMat(&debugImageWeightD,col);
		fillCvMat(&debugImageWeightP,col);
		fillCvMat(&debugImageWeightedResP,col);
		fillCvMat(&debugImageWeightedResD,col);
	}

	// get static values
	int w = frame->width(level);
	int h = frame->height(level);
	const CameraModel &m = frame->model(level);

	Eigen::Matrix3f rotMat = referenceToFrame.rxso3().matrix().cast<float>();
	Eigen::Matrix3f rotMatUnscaled = referenceToFrame.rotationMatrix().cast<float>();
	Eigen::Vector3f transVec = referenceToFrame.translation().cast<float>();

	// Calculate rotation around optical axis for rotating source frame gradients
	Eigen::Vector3f forwardVector(0, 0, -1);
	Eigen::Vector3f rotatedForwardVector = rotMatUnscaled * forwardVector;
	Eigen::Quaternionf shortestBackRotation;
	shortestBackRotation.setFromTwoVectors(rotatedForwardVector, forwardVector);
	Eigen::Matrix3f rollMat = shortestBackRotation.toRotationMatrix() * rotMatUnscaled;
	float xRoll0 = rollMat(0, 0);
	float xRoll1 = rollMat(0, 1);
	float yRoll0 = rollMat(1, 0);
	float yRoll1 = rollMat(1, 1);


	const Eigen::Vector3f* refPoint_max = reference->posData[level] + reference->numData[level];
	const Eigen::Vector3f* refPoint = reference->posData[level];
	const Eigen::Vector2f* refColVar = reference->colorAndVarData[level];
	const Eigen::Vector2f* refGrad = reference->gradData[level];

	const float* 			frame_idepth = frame->idepth(level);
	const float* 			frame_idepthVar = frame->idepthVar(level);
	const Eigen::Vector4f* 	frame_intensityAndGradients = frame->gradients(level);


	float sxx=0,syy=0,sx=0,sy=0,sw=0;

	float usageCount = 0;

	int idx=0;
	bool isOmni = (camModel->getType() == CameraModelType::OMNI);
	for(;refPoint<refPoint_max; refPoint++, refGrad++, refColVar++)
	{
		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
		vec2 uv = m.camToPixel(Wxp);
		float u_new = uv[0];
		float v_new = uv[1];

		// step 1a: coordinates have to be in image:
		// (inverse test to exclude NANs)
		if(!(u_new > 1 && v_new > 1 && u_new < w-2 && v_new < h-2))
			continue;

		*(buf_warped_x+idx) = Wxp(0);
		*(buf_warped_y+idx) = Wxp(1);
		*(buf_warped_z+idx) = Wxp(2);

		Eigen::Vector3f resInterp = getInterpolatedElement43(frame_intensityAndGradients, u_new, v_new, w);


		// save values
#if USE_ESM_TRACKING == 1
		// get rotated gradient of point
		float rotatedGradX = xRoll0 * (*refGrad)[0] + xRoll1 * (*refGrad)[1];
		float rotatedGradY = yRoll0 * (*refGrad)[0] + yRoll1 * (*refGrad)[1];

		//TODO double check this is OK for omni
		*(buf_warped_dx+idx) = m.fx * 0.5f * (resInterp[0] + rotatedGradX);
		*(buf_warped_dy+idx) = m.fy * 0.5f * (resInterp[1] + rotatedGradY);
#else
		*(buf_warped_dx+idx) = fx_l * resInterp[0];
		*(buf_warped_dy+idx) = fy_l * resInterp[1];
#endif


		float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
		//TODO change this for OMNI
		float c2 = resInterp[2];
		float residual_p = c1 - c2;

		float weight = fabsf(residual_p) < 2.0f ? 1 : 2.0f / fabsf(residual_p);
		sxx += c1*c1*weight;
		syy += c2*c2*weight;
		sx += c1*weight;
		sy += c2*weight;
		sw += weight;


		*(buf_warped_residual+idx) = residual_p;
		*(buf_idepthVar+idx) = (*refColVar)[1];


		// new (only for Sim3):
		int idx_rounded = (int)(u_new+0.5f) + w*(int)(v_new+0.5f);
		float var_frameDepth = frame_idepthVar[idx_rounded];
		float ref_idepth;
		//Adapt depth interpretation for OMNI vs PROJ camera models
		if (isOmni) {
			ref_idepth = 1.0f / Wxp.norm();
			*(buf_d + idx) = 1.0f / refPoint->norm();
		}
		else {
			ref_idepth = 1.0f / Wxp[2];
			*(buf_d + idx) = 1.0f / (*refPoint)[2];
		}
		if(var_frameDepth > 0)
		{
			float residual_d = ref_idepth - frame_idepth[idx_rounded];
			*(buf_residual_d+idx) = residual_d;
			*(buf_warped_idepthVar+idx) = var_frameDepth;
		}
		else
		{
			*(buf_residual_d+idx) = -1;
			*(buf_warped_idepthVar+idx) = -1;
		}


		// DEBUG STUFF
		if(plotSim3TrackingIterationInfo)
		{
			// for debug plot only: find x,y again.
			// horribly inefficient, but who cares at this point...
			Eigen::Vector2f point = frame->model(level).camToPixel((*refPoint));
			int x = static_cast<int>(point.x());
			int y = static_cast<int>(point.y());

			//TODO change this for OMNI (not urgent).
			setPixelInCvMat(&debugImageOldImageSource,getGrayCvPixel((float)resInterp[2]),
				static_cast<int>(u_new+0.5f),static_cast<int>(v_new+0.5f),(camModel->w/w));
			setPixelInCvMat(&debugImageOldImageWarped,getGrayCvPixel((float)resInterp[2]),x,y,(camModel->w/w));
			setPixelInCvMat(&debugImageResiduals,getGrayCvPixel(residual_p+128),x,y,(camModel->w/w));

			if(*(buf_warped_idepthVar+idx) >= 0)
			{
				setPixelInCvMat(&debugImageDepthResiduals,getGrayCvPixel(128 + 800 * *(buf_residual_d+idx)),x,y,(camModel->w/w));

				if(plotWeights)
				{
					setPixelInCvMat(&debugImageWeightD,getGrayCvPixel(255 * (1/60.0f) * sqrtf(*(buf_weight_VarD+idx))),x,y,(camModel->w/w));
					setPixelInCvMat(&debugImageWeightedResD,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarD+idx)) * *(buf_residual_d+idx)),x,y,(camModel->w/w));
				}
			}


			if(plotWeights)
			{
				setPixelInCvMat(&debugImageWeightP,getGrayCvPixel(255 * 4 * sqrtf(*(buf_weight_VarP+idx))),x,y,(camModel->w/w));
				setPixelInCvMat(&debugImageHuberWeight,getGrayCvPixel(255 * *(buf_weight_Huber+idx)),x,y,(camModel->w/w));
				setPixelInCvMat(&debugImageWeightedResP,getGrayCvPixel(128 + (128/5.0f) * sqrtf(*(buf_weight_VarP+idx)) * *(buf_warped_residual+idx)),x,y,(camModel->w/w));
			}
		}

		idx++;

		//Adapting depth calculation for OMNI.
		float depthChange;
		if (isOmni) {
			depthChange = refPoint->norm() / Wxp.norm();
		} else {
			depthChange = (*refPoint)[2] / Wxp[2];
		}
		usageCount += depthChange < 1 ? depthChange : 1;
	}
	buf_warped_size = idx;


	pointUsage = usageCount / (float)reference->numData[level];

	affineEstimation_a_lastIt = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
	affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx)/sw;



	if(plotSim3TrackingIterationInfo)
	{
		Util::displayImage( "P Residuals", debugImageResiduals );
		Util::displayImage( "D Residuals", debugImageDepthResiduals );

		if(plotWeights)
		{
			Util::displayImage( "Huber Weights", debugImageHuberWeight );
			Util::displayImage( "DV Weights", debugImageWeightD );
			Util::displayImage( "IV Weights", debugImageWeightP );
			Util::displayImage( "WP Res", debugImageWeightedResP );
			Util::displayImage( "WD Res", debugImageWeightedResD );
		}
	}
}

Sim3ResidualStruct Sim3Tracker::calcSim3WeightsAndResidual(
		const Sim3& referenceToFrame)
{
	float tx = float(referenceToFrame.translation()[0]);
	float ty = float(referenceToFrame.translation()[1]);
	float tz = float(referenceToFrame.translation()[2]);

	Sim3ResidualStruct sumRes;
	memset(&sumRes, 0, sizeof(Sim3ResidualStruct));


	float sum_rd=0, sum_rp=0, sum_wrd=0, sum_wrp=0, sum_wp=0, sum_wd=0, sum_num_d=0, sum_num_p=0;

	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);	// x'
		float py = *(buf_warped_y+i);	// y'
		float pz = *(buf_warped_z+i);	// z'

		float d = *(buf_d+i);	// d

		float rp = *(buf_warped_residual+i); // r_p
		float rd = *(buf_residual_d+i);	 // r_d

		float gx = *(buf_warped_dx+i);	// \delta_x I
		float gy = *(buf_warped_dy+i);  // \delta_y I

		float s = settings.var_weight * *(buf_idepthVar+i);	// \sigma_d^2
		float sv = settings.var_weight * *(buf_warped_idepthVar+i);	// \sigma_d^2'


		// calc dw/dd (first 2 components):
		float g0 = (tx * pz - tz * px) / (pz*pz*d);
		float g1 = (ty * pz - tz * py) / (pz*pz*d);
		float g2 = (pz - tz) / (pz*pz*d);

		// calc w_p
		float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		float w_p = 1.0f / (cameraPixelNoise2 + s * drpdd * drpdd);

		float w_d = 1.0f / (sv + g2*g2*s);

		float weighted_rd = fabs(rd*sqrtf(w_d));
		float weighted_rp = fabs(rp*sqrtf(w_p));


		float weighted_abs_res = sv > 0 ? weighted_rd+weighted_rp : weighted_rp;
		float wh = fabs(weighted_abs_res < settings.huber_d ? 1 : settings.huber_d / weighted_abs_res);

		if(sv > 0)
		{
			sumRes.sumResD += wh * w_d * rd*rd;
			sumRes.numTermsD++;
		}

		sumRes.sumResP += wh * w_p * rp*rp;
		sumRes.numTermsP++;


		if(plotSim3TrackingIterationInfo)
		{
			// for debug
			*(buf_weight_Huber+i) = wh;
			*(buf_weight_VarP+i) = w_p;
			*(buf_weight_VarD+i) = sv > 0 ? w_d : 0;


			sum_rp += fabs(rp);
			sum_wrp += fabs(weighted_rp);
			sum_wp += sqrtf(w_p);
			sum_num_p++;

			if(sv > 0)
			{
				sum_rd += fabs(weighted_rd);
				sum_wrd += fabs(rd);
				sum_wd += sqrtf(w_d);
				sum_num_d++;
			}
		}

		*(buf_weight_p+i) = wh * w_p;

		if(sv > 0)
			*(buf_weight_d+i) = wh * w_d;
		else
			*(buf_weight_d+i) = 0;

	}

	sumRes.mean = (sumRes.sumResD + sumRes.sumResP) / (sumRes.numTermsD + sumRes.numTermsP);
	sumRes.meanD = (sumRes.sumResD) / (sumRes.numTermsD);
	sumRes.meanP = (sumRes.sumResP) / (sumRes.numTermsP);

	if(plotSim3TrackingIterationInfo)
	{
		printf("rd %f, rp %f, wrd %f, wrp %f, wd %f, wp %f\n ",
				sum_rd/sum_num_d,
				sum_rp/sum_num_p,
				sum_wrd/sum_num_d,
				sum_wrp/sum_num_p,
				sum_wd/sum_num_d,
				sum_wp/sum_num_p);
	}
	return sumRes;
}

void Sim3Tracker::calcSim3LGS(LGS7 &ls7)
{
	LGS4 ls4;
	LGS6 ls6;
	ls6.initialize(camModel->w*camModel->h);
	ls4.initialize(camModel->w*camModel->h);

	if (camModel->getType() == CameraModelType::PROJ) {
		for (int i = 0; i < buf_warped_size; i++)
		{
			float px = *(buf_warped_x + i);	// x'
			float py = *(buf_warped_y + i);	// y'
			float pz = *(buf_warped_z + i);	// z'

			float wp = *(buf_weight_p + i);	// wr/wp
			float wd = *(buf_weight_d + i);	// wr/wd

			float rp = *(buf_warped_residual + i); // r_p
			float rd = *(buf_residual_d + i);	 // r_d

			float gx = *(buf_warped_dx + i);	// \delta_x I
			float gy = *(buf_warped_dy + i);  // \delta_y I


			float z = 1.0f / pz;
			float z_sqr = 1.0f / (pz*pz);
			Vector6 v;
			Vector4 v4;
			v[0] = z*gx + 0;
			v[1] = 0 + z*gy;
			v[2] = (-px * z_sqr) * gx +
				(-py * z_sqr) * gy;
			v[3] = (-px * py * z_sqr) * gx +
				(-(1.0f + py * py * z_sqr)) * gy;
			v[4] = (1.0f + px * px * z_sqr) * gx +
				(px * py * z_sqr) * gy;
			v[5] = (-py * z) * gx +
				(px * z) * gy;

			// new:
			v4[0] = z_sqr;
			v4[1] = z_sqr * py;
			v4[2] = -z_sqr * px;
			v4[3] = z;

			ls6.update(v, rp, wp);		// Jac = - v
			ls4.update(v4, rd, wd);	// Jac = v4

		}
	}
	else  /* CameraModelType::OMNI */ {
		const OmniCameraModel *m = static_cast<const OmniCameraModel*>(camModel.get());
		float e = m->e;
		for (int i = 0; i < buf_warped_size; i++)
		{
			float px = *(buf_warped_x + i);	// x'
			float py = *(buf_warped_y + i);	// y'
			float pz = *(buf_warped_z + i);	// z'

			float wp = *(buf_weight_p + i);	// wr/wp
			float wd = *(buf_weight_d + i);	// wr/wd

			float rp = *(buf_warped_residual + i); // r_p
			float rd = *(buf_residual_d + i);	 // r_d

			float gx = *(buf_warped_dx + i);	// \delta_x I
			float gy = *(buf_warped_dy + i);  // \delta_y I


			float z = 1.0f / pz;
			float n = vec3(px, py, pz).norm();
			float n_sqr = n*n;
			float in = 1.f / n;
			float den = 1.f / ((pz + n*e)*(pz + n*e));
			Vector6 v;

			float gxd = gx * den, gyd = gy * den;

			float J00 = gxd * (pz + e*(n - (px*px)*in));
			float J01 = -gyd * (e*px*py * in);

			float J10 = -gxd * (e*px*py * in);
			float J11 = gyd * (pz + e*(n - (py*py)*in));

			float J20 = -gxd * px*(1.f + pz*e*in);
			float J21 = -gyd * py*(1.f + pz*e*in);

			v[0] = J00 + J01;
			v[1] = J10 + J11;
			v[2] = J20 + J21;

			v[3] = -pz*v[1] + py*v[2];
			v[4] = pz*v[0] - px*v[2];
			v[5] = -py*v[0] + px*v[1];

			Vector4 v4;

			// new:
			v4[0] = n_sqr;
			v4[1] = n_sqr * py;
			v4[2] = -n_sqr * px;
			v4[3] = z;

			ls6.update(v, rp, wp);		// Jac = - v
			ls4.update(v4, rd, wd);	// Jac = v4

		}
	}

	ls4.finishNoDivide();
	ls6.finishNoDivide();


	ls7.initializeFrom(ls6, ls4);
}

void Sim3Tracker::calcResidualAndBuffers_debugStart()
{
	if(plotTrackingIterationInfo || saveAllTrackingStagesInternal)
	{
		int other = saveAllTrackingStagesInternal ? 255 : 0;
		fillCvMat(&debugImageResiduals,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageExternalWeights,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageWeights,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageSource,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageWarped,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageScaleEstimation,cv::Vec3b(255,other,other));
		fillCvMat(&debugImageDepthResiduals,cv::Vec3b(other,other,255));
	}
}

void Sim3Tracker::calcResidualAndBuffers_debugFinish(int w)
{
	if(plotTrackingIterationInfo)
	{
		Util::displayImage( "Weights", debugImageWeights );
		Util::displayImage( "second_frame", debugImageSecondFrame );
		Util::displayImage( "Intensities of second_frame at transformed positions", debugImageOldImageSource );
		Util::displayImage( "Intensities of second_frame at pointcloud in first_frame", debugImageOldImageWarped );
		Util::displayImage( "Residuals", debugImageResiduals );
		Util::displayImage( "DepthVar Weights", debugImageExternalWeights );
		Util::displayImage( "Depth Residuals", debugImageDepthResiduals );

		// wait for key and handle it
		bool looping = true;
		while(looping)
		{
			int k = Util::waitKey(1);
			if(k == -1)
			{
				if(autoRunWithinFrame)
					break;
				else
					continue;
			}

			char key = k;
			if(key == ' ')
				looping = false;
			else
				handleKey(k);
		}
	}

	if(saveAllTrackingStagesInternal)
	{
		char charbuf[500];
 
		snprintf(charbuf,500,"save/%sresidual-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageResiduals);

		snprintf(charbuf,500,"save/%swarped-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageOldImageWarped);

		snprintf(charbuf,500,"save/%sweights-%d-%d.png",packagePath.c_str(),w,iterationNumber);
		cv::imwrite(charbuf,debugImageWeights);

		printf("saved three images for lvl %d, iteration %d\n",w,iterationNumber);
	}
}
}
