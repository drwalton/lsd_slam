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
#include "Util/EigenCoreInclude.hpp"
#include "opencv2/core/core.hpp"
#include "Util/settings.hpp"
#include "Util/IndexThreadReduce.hpp"
#include "Util/SophusUtil.hpp"
#include "g2o/stuff/timeutil.h"
#include "CameraModel/CameraModel.hpp"
#ifdef _WIN32
using g2o::timeval;
#endif

namespace lsd_slam
{

class DepthMapPixelHypothesis;
class Frame;
class KeyFrameGraph;

struct DepthMapDebugSettings final {
	explicit DepthMapDebugSettings();
	~DepthMapDebugSettings() throw();
	bool debugShowEstimatedDepths;
	bool printPropagationStatistics;
    bool printFillHolesStatistics;
    bool printObserveStatistics;
    bool printObservePurgeStatistics;
    bool printRegularizeStatistics;
    bool printLineStereoStatistics;
    bool printLineStereoFails;
};

///\brief Maintains a depth map (consisting of DepthMapPixelHypothesis), which
///       may be updated via stereo comparisons and regularisation.
///\note A SlamSystem object maintains one DepthMap, which it uses to hold the 
///      estimated depths for the current keyframe.
class DepthMap
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	DepthMap(const CameraModel &model);
	DepthMap(const DepthMap&) = delete;
	DepthMap& operator=(const DepthMap&) = delete;
	~DepthMap() throw();

	///\brief Reset depth map (cause it to be recalculated).
	///\note Labels all points in depth map as "Invalid", meaning they will be
	///      updated from scratch at the next update.
	void reset();
	
	///\brief does obervation and regularization only.
	void updateKeyframe(std::deque< std::shared_ptr<Frame>, std::allocator<std::shared_ptr<Frame> > > referenceFrames);

	///\brief does propagation and whole-filling-regularization (no observation, for that need to call updateKeyframe()!)
	void createKeyFrame(Frame* new_keyframe);
	
	///\brief Perform one iteration of hole-filling.
	void finalizeKeyFrame();

	void invalidate();
	inline bool isValid() {return activeKeyFrame!=0;};

	int debugPlotDepthMap();

	// ONLY for debugging, their memory is managed (created & deleted) by this object.
	cv::Mat debugImageHypothesisHandling;
	cv::Mat debugImageHypothesisPropagation;
	cv::Mat debugImageStereoLines;
	cv::Mat debugImageDepth, debugImageDepthGray;

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
	
	
	void propagateDepth(Frame* new_keyframe);
	Frame* activeKeyFrame;

private:
	// camera matrix etc.
	std::unique_ptr<CameraModel> model;
	CameraModelType modelType;
	const size_t width, height;

	// ============= parameter copies for convenience ===========================
	// these are just copies of the pointers given to this function, for convenience.
	// these are NOT managed by this object!
	boost::shared_lock<boost::shared_mutex> activeKeyFramelock;
	const float* activeKeyFrameImageData;
	bool activeKeyFrameIsReactivated;

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
	
	///\brief does the line-stereo seeking.
	///\note takes a lot of parameters, because they all have been pre-computed before.
    ///\param mat: NEW image
    ///\param KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
    ///\param trafo: x_old = trafo * x_new; (from new to old image)
    ///\param realVal: descriptor in OLD image.
    ///\param[out] result_idepth : point depth in new camera's coordinate system
    ///\param[out] returns: result_u/v : point's coordinates in new camera's coordinate system
    ///\param[out] returns: idepth_var: (approximated) measurement variance of inverse depth of result_point_NEW
	///\return If a good match is found, the associated (positive-valued) error is returned.
	///        If no good match is found, a (negative) error code indicating
	///        the reason for the failure is returned.
	///        -4: Epipolar line length invalid (zero or NaN).
	///        -1: if out of bounds
	///        -2: if not found
	float doLineStereo(
			const float u, const float v, const float epxn, const float epyn,
			const float min_idepth, const float prior_idepth, float max_idepth,
			const Frame* const referenceFrame, const float* referenceFrameImage,
			float &result_idepth, float &result_var, float &result_eplLength,
			RunningStats* const stats);

	float doOmniStereo(
			const float u, const float v, const vec3 &epDir,
			const float min_idepth, const float prior_idepth, float max_idepth,
			const Frame* const referenceFrame, const float* referenceFrameImage,
			float &result_idepth, float &result_var, float &result_eplLength,
			RunningStats* const stats);

	

	void observeDepth();
	void observeDepthRow(size_t yMin, size_t yMax, RunningStats* stats);
	bool observeDepthCreate(const int &x, const int &y, const int &idx, RunningStats* const &stats);
	bool observeDepthUpdate(const int &x, const int &y, const int &idx, const float* keyFrameMaxGradBuf, RunningStats* const &stats);
	
	///\brief Check an epipolar line segment for validity
	///\return A boolean value indicating if the epipolar line segment is valid.
	bool makeAndCheckEPL(const int x, const int y, const Frame* const ref, float* pepx, float* pepy, RunningStats* const stats);

	bool makeAndCheckEPLOmni(const int x, const int y, const Frame* const ref, 
		vec3 *epDir, RunningStats* const stats);

	void regularizeDepthMap(bool removeOcclusion, int validityTH);
	template<bool removeOcclusions> void regularizeDepthMapRow(int validityTH, int yMin, int yMax, RunningStats* stats);


	void buildRegIntegralBuffer();
	void buildRegIntegralBufferRow1(size_t yMin, size_t yMax, RunningStats* stats);
	void regularizeDepthMapFillHoles();
	void regularizeDepthMapFillHolesRow(size_t yMin, size_t yMax, RunningStats* stats);


	void resetCounters();

	//float clocksPropagate, clocksPropagateKF, clocksObserve, msObserve, clocksReg1, clocksReg2, msReg1, msReg2, clocksFinalize;
};

}
