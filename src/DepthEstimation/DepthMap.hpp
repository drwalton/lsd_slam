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
#include <memory>
#include "util/EigenCoreInclude.hpp"
#include "util/settings.hpp"
#include "util/IndexThreadReduce.hpp"
#include "util/SophusUtil.hpp"
#include "g2o/stuff/timeutil.h"
#include "DepthMapDebugImages.hpp"
#include "DepthMapInitMode.hpp"
#ifdef _WIN32
using g2o::timeval;
#endif

namespace lsd_slam
{

class DepthMapPixelHypothesis;
class Frame;
class KeyframeGraph;
class CameraModel;

struct DepthMapDebugSettings
{
	explicit DepthMapDebugSettings();
	bool convertDepths;
	int drawMatchInvChance;
};

/**
 * Keeps a detailed depth map (consisting of DepthMapPixelHypothesis) and does
 * stereo comparisons and regularization to update it.
 */
class DepthMap
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	DepthMap(const CameraModel &model, DepthMapInitMode mode = DepthMapInitMode::RANDOM);
	DepthMap(const DepthMap&) = delete;
	DepthMap& operator=(const DepthMap&) = delete;
	~DepthMap();

	/** Resets everything. */
	void reset();
	
	/**
	 * does obervation and regularization only.
	 **/
	void updateKeyframe(std::deque< std::shared_ptr<Frame>, std::allocator<std::shared_ptr<Frame> > > referenceFrames);

	/**
	 * does propagation and whole-filling-regularization (no observation, for that need to call updateKeyframe()!)
	 **/
	void createKeyframe(Frame* new_keyframe);
	
	/**
	 * does one fill holes iteration
	 */
	void finalizeKeyframe();

	void invalidate();
	inline bool isValid() {return activeKeyframe!=0;};

	int debugPlotDepthMap();

	// ONLY for debugging, their memory is managed (created & deleted) by this object.
	cv::Mat debugImageHypothesisHandling;
	cv::Mat debugImageHypothesisPropagation;
	cv::Mat debugImageStereoLines;
	cv::Mat debugImageDepth;

	void initializeFromGTDepth(Frame* new_frame);
	void initializeRandomly(Frame* new_frame);

	void setFromExistingKF(Frame* kf);

	void addTimingSample();
	float msUpdate, msCreate, msFinalize;
	float msObserve, msRegularize, msPropagate, msFillHoles, msSetDepth;
	int nUpdate, nCreate, nFinalize;
	int nObserve, nRegularize, nPropagate, nFillHoles, nSetDepth;
	struct timeval lastHzUpdate;
	float nAvgUpdate, nAvgCreate, nAvgFinalize;
	float nAvgObserve, nAvgRegularize, nAvgPropagate, nAvgFillHoles, nAvgSetDepth;



	// pointer to global keyframe graph
	IndexThreadReduce threadReducer;

	DepthMapDebugSettings settings;
	DepthMapDebugImages debugImages;

private:
	std::unique_ptr<CameraModel> camModel_;
	const DepthMapInitMode initMode_;

	// ============= parameter copies for convenience ===========================
	// these are just copies of the pointers given to this function, for convenience.
	// these are NOT managed by this object!
	Frame* activeKeyframe;
	boost::shared_lock<boost::shared_mutex> activeKeyframelock;
	const float* activeKeyframeImageData;
	bool activeKeyframeIsReactivated;

	Frame* oldest_referenceFrame;
	Frame* newest_referenceFrame;
	std::vector<Frame*> referenceFrameByID;
	int referenceFrameByID_offset;

	// ============= internally used buffers for intermediate calculations etc. =============
	// for internal depth tracking, their memory is managed (created & deleted) by this object.
	DepthMapPixelHypothesis* otherDepthMap;
	DepthMapPixelHypothesis* currentDepthMap;
	int* validityIntegralBuffer;

	

	// ============ internal functions ==================================================
	// does the line-stereo seeking.
	// takes a lot of parameters, because they all have been pre-computed before.
	inline float doStereoProj(
			const float u, const float v, const float epxn, const float epyn,
			const float min_idepth, const float prior_idepth, float max_idepth,
			const Frame* const referenceFrame, const float* referenceFrameImage,
			float &result_idepth, float &result_var, float &result_eplLength,
			RunningStats* const stats);

	float DepthMap::doStereoOmni(
		const float u, const float v, const vec3 &epDir,
		const float min_idepth, const float prior_idepth, float max_idepth,
		const Frame* const referenceFrame, const float* referenceFrameImage,
		float &result_idepth, float &result_var, float &result_eplLength,
		RunningStats* stats);


	void propagateDepth(Frame* new_keyframe);
	

	void observeDepth();
	void observeDepthRow(int yMin, int yMax, RunningStats* stats);
	bool observeDepthCreate(const int &x, const int &y, const int &idx, RunningStats* const &stats);
	bool observeDepthUpdate(const int &x, const int &y, const int &idx, const float* keyframeMaxGradBuf, RunningStats* const &stats);
	bool makeAndCheckEPLProj(const int x, const int y, const Frame* const ref, float* pepx, float* pepy, RunningStats* const stats);
	bool makeAndCheckEPLOmni(const int x, const int y, const Frame* const ref,
		vec3 *epDir, RunningStats* const stats);


	void regularizeDepthMap(bool removeOcclusion, int validityTH);
	template<bool removeOcclusions> void regularizeDepthMapRow(int validityTH, int yMin, int yMax, RunningStats* stats);


	void buildRegIntegralBuffer();
	void buildRegIntegralBufferRow1(int yMin, int yMax, RunningStats* stats);
	void regularizeDepthMapFillHoles();
	void regularizeDepthMapFillHolesRow(int yMin, int yMax, RunningStats* stats);


	void resetCounters();

	//float clocksPropagate, clocksPropagateKF, clocksObserve, msObserve, clocksReg1, clocksReg2, msReg1, msReg2, clocksFinalize;
};

}
