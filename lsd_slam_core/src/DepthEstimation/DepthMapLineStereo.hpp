#ifndef DEPTHESTIMATION_DEPTHMAPLINESTEREO_HPP_INCLUDED
#define DEPTHESTIMATION_DEPTHMAPLINESTEREO_HPP_INCLUDED

#include "ProjCameraModel.hpp"
#include <array>
#include "util/Constants.hpp"
#include "util/settings.hpp"
#include "VectorTypes.hpp"

namespace lsd_slam {

bool makeAndCheckEPL(const int x, const int y, const float* const ref,
	const float *const keyframe,
	const RigidTransform &keyframeToReference,
	float* pepx, float* pepy, RunningStats* const stats,
	const ProjCameraModel &model);

float doLineStereo(
	const float u, const float v, const float epxn, const float epyn,
	const float min_idepth, const float prior_idepth, float max_idepth,
	const float* const keyframeImage, const float* referenceFrameImage,
	const ProjCameraModel &model, const RigidTransform &keyframeToReference,
	float &result_idepth, float &result_var, float &result_eplLength,
	const Eigen::Vector4f *keyframeGradients,
	float initialTrackedResidual,
	RunningStats* const stats,
	cv::Mat &drawIm = emptyMat);

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
	const RigidTransform &keyframeToReference,
	const float *referenceFrame,
	const float *activeKeyFrame,
	float initialTrackedResidual,
	const Eigen::Vector4f *keyframeGradients,
	RunningStats *stats);
}

#endif