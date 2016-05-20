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

#pragma once
#include <opencv2/core/core.hpp>
#include "util/settings.hpp"
#include "util/DenseDepthTrackerSettings.hpp"
#include "util/EigenCoreInclude.hpp"
#include "util/SophusUtil.hpp"
#include "Tracking/LGSX.hpp"
#include "OmniCameraModel.hpp"

namespace lsd_slam
{

class TrackingReference;
class Frame;

///\brief Tracker designed to align frames using rigid transforms (elements
///       of SE3).
class SE3Tracker
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	/// Model for current camera.
	std::unique_ptr<CameraModel> model;

	DenseDepthTrackerSettings settings;


	// debug images
	cv::Mat debugImageResiduals;
	cv::Mat debugImageWeights;
	cv::Mat debugImageSecondFrame;
	cv::Mat debugImageOldImageSource;
	cv::Mat debugImageOldImageWarped;


	SE3Tracker(const CameraModel &model);
	SE3Tracker(const SE3Tracker&) = delete;
	SE3Tracker& operator=(const SE3Tracker&) = delete;
	~SE3Tracker();

	///\brief Apply whole tracking procedure to a single frame.
	///\param reference The keyframe against which to track the new frame.
	///\param frame The new frame to track
	///\param frameToReference_initialEstimate The initial estimate for the transform
	///    from frame to reference.
	///\note The reference frame has an associated depth, but the tracked frame
	///      does not.
	SE3 trackFrame(
			TrackingReference* reference,
			Frame* frame,
			const SE3& frameToReference_initialEstimate);
	

	SE3 trackFrameOnPermaref(
			Frame* reference,
			Frame* frame,
			const SE3 &referenceToFrame);


	float checkPermaRefOverlap(
			Frame* reference,
			const SE3 &referenceToFrame);


	float pointUsage;
	float lastGoodCount;
	float lastMeanRes;
	float lastBadCount;
	float lastResidual;

	float affineEstimation_a;
	float affineEstimation_b;


	bool diverged;
	bool trackingWasGood;

	///Used for plotting tracking when debugging.
	bool plotTracking;
private:

	//========================= INTERNAL BUFFERS ==============================
	float* buf_warped_residual;
	float* buf_warped_dx;
	float* buf_warped_dy;

	///These three buffers contain 3D points from the reference frame, to which 
	///    the transform referenceToFrame has been applied.
	float* buf_warped_x;
	float* buf_warped_y;
	float* buf_warped_z;

	float* buf_d;
	float* buf_idepthVar;
	float* buf_weight_p;

	int buf_warped_size;


	float calcResidualAndBuffers(
			const Eigen::Vector3f* refPoint,
			const Eigen::Vector2f* refColVar,
			int* idxBuf,
			int refNum,
			Frame* frame,
			const Sophus::SE3f& referenceToFrame,
			int level,
			bool plotResidual = false);

	float calcWeightsAndResidual(
			const Sophus::SE3f& referenceToFrame);
	void calculateWarpUpdate(
			LGS6 &ls);

	void calcResidualAndBuffers_debugStart();
	void calcResidualAndBuffers_debugFinish(int w);


	/// used for image saving
	int iterationNumber;

	float affineEstimation_a_lastIt;
	float affineEstimation_b_lastIt;
};

}

