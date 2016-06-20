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

#include "SE3Tracker.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "DataStructures/Frame.hpp"
#include "Tracking/TrackingReference.hpp"
#include "util/globalFuncs.hpp"
#include "IOWrapper/ImageDisplay.hpp"
#include "Tracking/LGSX.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"

#include <Eigen/Core>

#if _WIN32
#define snprintf _snprintf_s
#endif 

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


SE3Tracker::SE3Tracker(const CameraModel &model)
	//TODO TEMP TESTING WITH PROJ
	//:camModel(model.clone())
	:camModel(model.makeOmniCamModel())
{
	int w = model.w, h = model.h;
	settings = DenseDepthTrackerSettings();
	//settings.maxItsPerLvl[0] = 2;

	buf_warped_residual = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dx = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dy = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_x = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_y = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_z = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_p = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_warped_size = 0;

	debugImageWeights = cv::Mat(h,w,CV_8UC3);
	debugImageResiduals = cv::Mat(h,w,CV_8UC3);
	debugImageSecondFrame = cv::Mat(h,w,CV_8UC3);
	debugImageOldImageWarped = cv::Mat(h,w,CV_8UC3);
	debugImageOldImageSource = cv::Mat(h,w,CV_8UC3);



	lastResidual = 0;
	iterationNumber = 0;
	pointUsage = 0;
	lastGoodCount = lastBadCount = 0;

	diverged = false;
}

SE3Tracker::~SE3Tracker()
{
	debugImageResiduals.release();
	debugImageWeights.release();
	debugImageSecondFrame.release();
	debugImageOldImageSource.release();
	debugImageOldImageWarped.release();


	Eigen::internal::aligned_free((void*)buf_warped_residual);
	Eigen::internal::aligned_free((void*)buf_warped_dx);
	Eigen::internal::aligned_free((void*)buf_warped_dy);
	Eigen::internal::aligned_free((void*)buf_warped_x);
	Eigen::internal::aligned_free((void*)buf_warped_y);
	Eigen::internal::aligned_free((void*)buf_warped_z);

	Eigen::internal::aligned_free((void*)buf_d);
	Eigen::internal::aligned_free((void*)buf_idepthVar);
	Eigen::internal::aligned_free((void*)buf_weight_p);
}



// tracks a frame.
// first_frame has depth, second_frame DOES NOT have depth.
float SE3Tracker::checkPermaRefOverlap(
		Frame* reference,
		const SE3 &referenceToFrameOrg)
{
	Sophus::SE3f referenceToFrame = referenceToFrameOrg.cast<float>();
	boost::unique_lock<boost::mutex> lock2 = boost::unique_lock<boost::mutex>(reference->permaRef_mutex);

	int w2 = reference->width(QUICK_KF_CHECK_LVL) - 1;
	int h2 = reference->height(QUICK_KF_CHECK_LVL) - 1;
	const CameraModel &m = reference->model(QUICK_KF_CHECK_LVL);

	Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	Eigen::Vector3f transVec = referenceToFrame.translation();

	const Eigen::Vector3f* refPoint_max = reference->permaRef_posData + reference->permaRefNumPts;
	const Eigen::Vector3f* refPoint = reference->permaRef_posData;

	float usageCount = 0;
	for (; refPoint<refPoint_max; refPoint++)
	{
		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
		vec2 uv = m.camToPixel(Wxp);
		float u_new = uv[0];
		float v_new = uv[1];
		if ((u_new > 0 && v_new > 0 && u_new < w2 && v_new < h2))
		{
			//TODO check this, make sure it's OK for OMNI
			float depthChange = (*refPoint)[2] / Wxp[2];
			usageCount += depthChange < 1 ? depthChange : 1;
		}
	}

	pointUsage = usageCount / (float)reference->permaRefNumPts;
	return pointUsage;
}


// tracks a frame.
// first_frame has depth, second_frame DOES NOT have depth.
SE3 SE3Tracker::trackFrameOnPermaref(
		Frame* reference,
		Frame* frame,
		const SE3 &referenceToFrameOrg)
{

	Sophus::SE3f referenceToFrame = referenceToFrameOrg.cast<float>();

	boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
	boost::unique_lock<boost::mutex> lock2 = boost::unique_lock<boost::mutex>(reference->permaRef_mutex);

	affineEstimation_a = 1; affineEstimation_b = 0;

	LGS6 ls;
	diverged = false;
	trackingWasGood = true;

	callOptimized(calcResidualAndBuffers, (reference->permaRef_posData, reference->permaRef_colorAndVarData, 0, reference->permaRefNumPts, frame, referenceToFrame, QUICK_KF_CHECK_LVL, false));
	if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (camModel->w>>QUICK_KF_CHECK_LVL)*(camModel->h>>QUICK_KF_CHECK_LVL))
	{
		diverged = true;
		trackingWasGood = false;
		return SE3();
	}
	if(useAffineLightningEstimation)
	{
		affineEstimation_a = affineEstimation_a_lastIt;
		affineEstimation_b = affineEstimation_b_lastIt;
	}
	float lastErr = callOptimized(calcWeightsAndResidual,(referenceToFrame));

	float LM_lambda = settings.lambdaInitialTestTrack;

	for(int iteration=0; iteration < settings.maxItsTestTrack; iteration++)
	{
		callOptimized(calculateWarpUpdate,(ls));


		int incTry=0;
		while(true)
		{
			// solve LS system with current lambda
			Vector6 b = -ls.b;
			Matrix6x6 A = ls.A;
			for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
			Vector6 inc = A.ldlt().solve(b);
			incTry++;

			// apply increment. pretty sure this way round is correct, but hard to test.
			Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;

			// re-evaluate residual
			callOptimized(calcResidualAndBuffers, (reference->permaRef_posData, reference->permaRef_colorAndVarData, 0, reference->permaRefNumPts, frame, new_referenceToFrame, QUICK_KF_CHECK_LVL, false));
			if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (camModel->w>>QUICK_KF_CHECK_LVL)*(camModel->h>>QUICK_KF_CHECK_LVL))
			{
				diverged = true;
				trackingWasGood = false;
				return SE3();
			}
			float error = callOptimized(calcWeightsAndResidual,(new_referenceToFrame));


			// accept inc?
			if(error < lastErr)
			{
				// accept inc
				referenceToFrame = new_referenceToFrame;
				if(useAffineLightningEstimation)
				{
					affineEstimation_a = affineEstimation_a_lastIt;
					affineEstimation_b = affineEstimation_b_lastIt;
				}
				// converged?
				if(error / lastErr > settings.convergenceEpsTestTrack)
					iteration = static_cast<int>(settings.maxItsTestTrack);


				lastErr = error;


				if(LM_lambda <= 0.2)
					LM_lambda = 0;
				else
					LM_lambda *= settings.lambdaSuccessFac;

				break;
			}
			else
			{
				if(!(inc.dot(inc) > settings.stepSizeMinTestTrack))
				{
					iteration = static_cast<int>(settings.maxItsTestTrack);
					break;
				}

				if(LM_lambda == 0.f)
					LM_lambda = 0.2f;
				else
					LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
			}
		}
	}

	lastResidual = lastErr;

	trackingWasGood = !diverged
			&& lastGoodCount / (frame->width(QUICK_KF_CHECK_LVL)*frame->height(QUICK_KF_CHECK_LVL)) > MIN_GOODPERALL_PIXEL
			&& lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

	return toSophus(referenceToFrame);
}





// tracks a frame.
// first_frame has depth, second_frame DOES NOT have depth.
SE3 SE3Tracker::trackFrame(
		TrackingReference* reference,
		Frame* frame,
		const SE3& frameToReference_initialEstimate)
{

	boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
	diverged = false;
	trackingWasGood = true;
	affineEstimation_a = 1; affineEstimation_b = 0;

	if(saveAllTrackingStages)
	{
		saveAllTrackingStages = false;
		saveAllTrackingStagesInternal = true;
	}
	
	if (plotTrackingIterationInfo)
	{
		const float* frameImage = frame->image();
		for (size_t row = 0; row < camModel->h; ++ row)
			for (size_t col = 0; col < camModel->w; ++ col)
				setPixelInCvMat(&debugImageSecondFrame,getGrayCvPixel(frameImage[col+row*camModel->w]), col, row, 1);
	}

	// ============ track frame ============
	Sophus::SE3f referenceToFrame = frameToReference_initialEstimate.inverse().cast<float>();
	LGS6 ls;


	int numCalcResidualCalls[PYRAMID_LEVELS];
	int numCalcWarpUpdateCalls[PYRAMID_LEVELS];

	float last_residual = 0;


	for(int lvl=SE3TRACKING_MAX_LEVEL-1;lvl >= SE3TRACKING_MIN_LEVEL;lvl--)
	{
		numCalcResidualCalls[lvl] = 0;
		numCalcWarpUpdateCalls[lvl] = 0;

		reference->makePointCloud(lvl);

		callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
		if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (camModel->w>>lvl)*(camModel->h>>lvl))
		{
			diverged = true;
			trackingWasGood = false;
			return SE3();
		}

		if(useAffineLightningEstimation)
		{
			affineEstimation_a = affineEstimation_a_lastIt;
			affineEstimation_b = affineEstimation_b_lastIt;
		}
		float lastErr = callOptimized(calcWeightsAndResidual,(referenceToFrame));

		numCalcResidualCalls[lvl]++;


		float LM_lambda = settings.lambdaInitial[lvl];

		for(int iteration=0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
		{

			callOptimized(calculateWarpUpdate,(ls));

			numCalcWarpUpdateCalls[lvl]++;

			iterationNumber = iteration;

			int incTry=0;
			while(true)
			{
				// solve LS system with current lambda
				Vector6 b = -ls.b;
				Matrix6x6 A = ls.A;
				for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
				Vector6 inc = A.ldlt().solve(b);
				incTry++;

				// apply increment. pretty sure this way round is correct, but hard to test.
				Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;
				//Sophus::SE3f new_referenceToFrame = referenceToFrame * Sophus::SE3f::exp((inc));


				// re-evaluate residual
				callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, new_referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
				if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN* (camModel->w>>lvl)*(camModel->h>>lvl))
				{
					diverged = true;
					trackingWasGood = false;
					return SE3();
				}

				float error = callOptimized(calcWeightsAndResidual,(new_referenceToFrame));
				numCalcResidualCalls[lvl]++;


				// accept inc?
				if(error < lastErr)
				{
					// accept inc
					referenceToFrame = new_referenceToFrame;
					if(useAffineLightningEstimation)
					{
						affineEstimation_a = affineEstimation_a_lastIt;
						affineEstimation_b = affineEstimation_b_lastIt;
					}


					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						// debug output
						printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr, error);

						printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f\n",
								referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
								referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5]);
					}

					// converged?
					if(error / lastErr > settings.convergenceEps[lvl])
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
					}

					last_residual = lastErr = error;


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
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr, error);
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


	if(plotTracking)
		Util::displayImage("TrackingResidual", debugImageResiduals, false);


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
	}

	saveAllTrackingStagesInternal = false;

	lastResidual = last_residual;

	trackingWasGood = !diverged
			&& lastGoodCount / (frame->width(SE3TRACKING_MIN_LEVEL)*frame->height(SE3TRACKING_MIN_LEVEL)) > MIN_GOODPERALL_PIXEL
			&& lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

	if(trackingWasGood)
		reference->keyframe->numFramesTrackedOnThis++;

	frame->initialTrackedResidual = lastResidual / pointUsage;
	frame->pose->thisToParent_raw = sim3FromSE3(toSophus(referenceToFrame.inverse()),1);
	frame->pose->trackingParent = reference->keyframe->pose;
	return toSophus(referenceToFrame.inverse());
}

float SE3Tracker::calcWeightsAndResidual(
		const Sophus::SE3f& referenceToFrame)
{
	float tx = referenceToFrame.translation()[0];
	float ty = referenceToFrame.translation()[1];
	float tz = referenceToFrame.translation()[2];

	float sumRes = 0;

	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);	// x'
		float py = *(buf_warped_y+i);	// y'
		float pz = *(buf_warped_z+i);	// z'
		float d = *(buf_d+i);	// d
		float rp = *(buf_warped_residual+i); // r_p
		float gx = *(buf_warped_dx+i);	// \delta_x I
		float gy = *(buf_warped_dy+i);  // \delta_y I
		float s = settings.var_weight * *(buf_idepthVar+i);	// \sigma_d^2


		// calc dw/dd (first 2 components):
		float g0 = (tx * pz - tz * px) / (pz*pz*d);
		float g1 = (ty * pz - tz * py) / (pz*pz*d);


		// calc w_p
		float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);

		float weighted_rp = fabs(rp*sqrtf(w_p));

		float wh = fabs(weighted_rp < (settings.huber_d/2) ? 1 : (settings.huber_d/2) / weighted_rp);

		sumRes += wh * w_p * rp*rp;


		*(buf_weight_p+i) = wh * w_p;
	}

	return sumRes / buf_warped_size;
}


void SE3Tracker::calcResidualAndBuffers_debugStart()
{
	if(plotTrackingIterationInfo || saveAllTrackingStagesInternal)
	{
		int other = saveAllTrackingStagesInternal ? 255 : 0;
		fillCvMat(&debugImageResiduals,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageWeights,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageSource,cv::Vec3b(other,other,255));
		fillCvMat(&debugImageOldImageWarped,cv::Vec3b(other,other,255));
	}
}

void SE3Tracker::calcResidualAndBuffers_debugFinish(int w)
{
	if(plotTrackingIterationInfo)
	{
		Util::displayImage( "Weights", debugImageWeights );
		Util::displayImage( "second_frame", debugImageSecondFrame );
		Util::displayImage( "Intensities of second_frame at transformed positions", debugImageOldImageSource );
		Util::displayImage( "Intensities of second_frame at pointcloud in first_frame", debugImageOldImageWarped );
		Util::displayImage( "Residuals", debugImageResiduals );


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

#if defined(ENABLE_SSE)
float SE3Tracker::calcResidualAndBuffersSSE(
		const Eigen::Vector3f* refPoint,
		const Eigen::Vector2f* refColVar,
		int* idxBuf,
		int refNum,
		Frame* frame,
		const Sophus::SE3f& referenceToFrame,
		int level,
		bool plotResidual)
{
	return calcResidualAndBuffers(refPoint, refColVar, idxBuf, refNum, frame, referenceToFrame, level, plotResidual);
}
#endif

#if defined(ENABLE_NEON)
float SE3Tracker::calcResidualAndBuffersNEON(
		const Eigen::Vector3f* refPoint,
		const Eigen::Vector2f* refColVar,
		int* idxBuf,
		int refNum,
		Frame* frame,
		const Sophus::SE3f& referenceToFrame,
		int level,
		bool plotResidual)
{
	return calcResidualAndBuffers(refPoint, refColVar, idxBuf, refNum, frame, referenceToFrame, level, plotResidual);
}
#endif


float SE3Tracker::calcResidualAndBuffers(
		const Eigen::Vector3f* refPoint,
		const Eigen::Vector2f* refColVar,
		int* idxBuf,
		int refNum,
		Frame* frame,
		const Sophus::SE3f& referenceToFrame,
		int level,
		bool plotResidual)
{
	calcResidualAndBuffers_debugStart();

	if (plotResidual)
		debugImageResiduals.setTo(0);


	int w = frame->width(level);
	int h = frame->height(level);
	const CameraModel &m = frame->model(level);

	Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	Eigen::Vector3f transVec = referenceToFrame.translation();

	const Eigen::Vector3f* refPoint_max = refPoint + refNum;

	const Eigen::Vector4f* frame_gradients = frame->gradients(level);

	int idx = 0;

	float sumResUnweighted = 0;

	bool* isGoodOutBuffer = idxBuf != 0 ? frame->refPixelWasGood() : 0;

	int goodCount = 0;
	int badCount = 0;

	float sumSignedRes = 0;

	float sxx = 0, syy = 0, sx = 0, sy = 0, sw = 0;

	float usageCount = 0;

	for (; refPoint<refPoint_max; refPoint++, refColVar++, idxBuf++)
	{

		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
		vec2 uv = m.camToPixel(Wxp);
		float u_new = uv[0];
		float v_new = uv[1];

		// step 1a: coordinates have to be in image:
		// (inverse test to exclude NANs)
		if (!(u_new > 1 && v_new > 1 && u_new < w - 2 && v_new < h - 2))
		{
			if (isGoodOutBuffer != 0)
				isGoodOutBuffer[*idxBuf] = false;
			continue;
		}

		Eigen::Vector3f resInterp = getInterpolatedElement43(frame_gradients, u_new, v_new, w);

		float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
		//TODO check this is OK for OMNI
		float c2 = resInterp[2];
		float residual = c1 - c2;

		float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
		sxx += c1*c1*weight;
		syy += c2*c2*weight;
		sx += c1*weight;
		sy += c2*weight;
		sw += weight;

		bool isGood = residual*residual /
			(MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*(resInterp[0] * resInterp[0]
				+ resInterp[1] * resInterp[1])) < 1;

		if (isGoodOutBuffer != 0)
			isGoodOutBuffer[*idxBuf] = isGood;

		*(buf_warped_x + idx) = Wxp(0);
		*(buf_warped_y + idx) = Wxp(1);
		*(buf_warped_z + idx) = Wxp(2);

		*(buf_warped_dx + idx) = m.fx * resInterp[0];
		*(buf_warped_dy + idx) = m.fy * resInterp[1];
		//const OmniCameraModel *omModel;
		//if (omModel = dynamic_cast<const OmniCameraModel*>(&m)) {
		//	float d = Wxp.norm() * omModel->e;
		//	*(buf_warped_dx+idx) /= d;
		//	*(buf_warped_dy+idx) /= d;
		//}
		*(buf_warped_residual + idx) = residual;

		//TODO change this for OMNI
		*(buf_d + idx) = 1.0f / (*refPoint)[2];
		*(buf_idepthVar + idx) = (*refColVar)[1];
		idx++;


		if (isGood)
		{
			sumResUnweighted += residual*residual;
			sumSignedRes += residual;
			goodCount++;
		}
		else
			badCount++;

		//TODO change this for OMNI
		float depthChange = (*refPoint)[2] / Wxp[2];	// if depth becomes larger: pixel becomes "smaller", hence count it less.
		usageCount += depthChange < 1 ? depthChange : 1;


		// DEBUG STUFF
		if (plotTrackingIterationInfo || plotResidual)
		{
			// for debug plot only: find x,y again.
			// horribly inefficient, but who cares at this point...
			Eigen::Vector2f point = m.camToPixel((*refPoint));
			int x = static_cast<int>(point[0]);
			int y = static_cast<int>(point[1]);

			if (plotTrackingIterationInfo)
			{
				setPixelInCvMat(&debugImageOldImageSource, getGrayCvPixel((float)resInterp[2]),
					static_cast<int>(u_new + 0.5f), static_cast<int>(v_new + 0.5f), (camModel->w / w));
				//TODO change this for OMNI (not urgent).
				setPixelInCvMat(&debugImageOldImageWarped, getGrayCvPixel(
					(float)resInterp[2]), x, y, (camModel->w / w));
			}
			if (isGood)
				setPixelInCvMat(&debugImageResiduals, getGrayCvPixel(residual + 128), x, y, (camModel->w / w));
			else
				setPixelInCvMat(&debugImageResiduals, cv::Vec3b(0, 0, 255), x, y, (camModel->w / w));

		}
	}

	buf_warped_size = idx;

	pointUsage = usageCount / (float)refNum;
	lastGoodCount = float(goodCount);
	lastBadCount = float(badCount);
	lastMeanRes = sumSignedRes / goodCount;

	affineEstimation_a_lastIt = sqrtf((syy - sy*sy / sw) / (sxx - sx*sx / sw));
	affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx) / sw;

	calcResidualAndBuffers_debugFinish(w);

	return sumResUnweighted / goodCount;
}

void SE3Tracker::calculateWarpUpdate(
		LGS6 &ls)
{
	ls.initialize(camModel->w*camModel->h);

	if (camModel->getType() == CameraModelType::PROJ) {
		for (int i = 0; i < buf_warped_size; i++)
		{
			float px = *(buf_warped_x + i);
			float py = *(buf_warped_y + i);
			float pz = *(buf_warped_z + i);
			float r = *(buf_warped_residual + i);
			float gx = *(buf_warped_dx + i);
			float gy = *(buf_warped_dy + i);
			// step 3 + step 5 comp 6d error vector

			float z = 1.0f / pz;
			float z_sqr = 1.0f / (pz*pz);
			Vector6 v;
			v[0] = z*gx + 0;
			v[1] = 0 + z*gy;
			v[2] = (-px * z_sqr) * gx +
				(-py * z_sqr) * gy;

			v[3] = -pz*v[1] + py*v[2];
			v[4] = pz*v[0] - px*v[2];
			v[5] = -py*v[0] + px*v[1];

			/*
			v[3] = (-px * py * z_sqr) * gx +
			(-(1.0f + py * py * z_sqr)) * gy;
			v[4] = (1.0f + px * px * z_sqr) * gx +
			(px * py * z_sqr) * gy;
			v[5] = (-py * z) * gx +
			(px * z) * gy;
			*/

			// step 6: integrate into A and b:
			ls.update(v, r, *(buf_weight_p + i));
		}
	}
	else /* CameraModelType::OMNI */ {
		const OmniCameraModel *m = static_cast<const OmniCameraModel*>(camModel.get());
		float e = m->e;
		for (int i = 0; i < buf_warped_size; i++)
		{
			float px = *(buf_warped_x + i);
			float py = *(buf_warped_y + i);
			float pz = *(buf_warped_z + i);
			float r = *(buf_warped_residual + i);
			float gx = *(buf_warped_dx + i);
			float gy = *(buf_warped_dy + i);
			// step 3 + step 5 comp 6d error vector

			//float z_sqr = 1.0f / (pz*pz);
			float n = vec3(px, py, pz).norm();
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

			// step 6: integrate into A and b:
			ls.update(v, r, *(buf_weight_p + i));
		}
	}

	// solve ls
	ls.finish();
}



}

