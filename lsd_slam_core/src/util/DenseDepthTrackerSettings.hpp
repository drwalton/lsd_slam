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

#include <stdio.h>

namespace lsd_slam
{

/** ============== Depth Variance Handling ======================= */
#define SUCC_VAR_INC_FAC (1.01f) // before an ekf-update, the variance is increased by this factor.
#define FAIL_VAR_INC_FAC 1.1f // after a failed stereo observation, the variance is increased by this factor.
#define MAX_VAR (0.5f*0.5f) // initial variance on creation - if variance becomes larter than this, hypothesis is removed.

#define VAR_GT_INIT_INITIAL 0.01f*0.01f	// initial variance vor Ground Truth Initialization
#define VAR_RANDOM_INIT_INITIAL (0.5f*MAX_VAR)	// initial variance vor Random Initialization

// Whether to use the gradients of source and target frame for tracking,
// or only the target frame gradient
#define USE_ESM_TRACKING 1


#ifdef ANDROID
	// tracking pyramid levels.
	#define MAPPING_THREADS 2
	#define RELOCALIZE_THREADS 4
#else
	// tracking pyramid levels.
	#define MAPPING_THREADS 4
	#define RELOCALIZE_THREADS 6
#endif

#define SE3TRACKING_MIN_LEVEL 1
#define SE3TRACKING_MAX_LEVEL 5

#define SIM3TRACKING_MIN_LEVEL 1
#define SIM3TRACKING_MAX_LEVEL 5

#define QUICK_KF_CHECK_LVL 4

#define PYRAMID_LEVELS (SE3TRACKING_MAX_LEVEL > SIM3TRACKING_MAX_LEVEL ? SE3TRACKING_MAX_LEVEL : SIM3TRACKING_MAX_LEVEL)

class DenseDepthTrackerSettings
{
public:
	inline DenseDepthTrackerSettings()
	{
		// Set default settings
		if (PYRAMID_LEVELS > 6)
			printf("WARNING: Sim3Tracker(): default settings are intended for a maximum of 6 levels!");

		lambdaSuccessFac = 0.5f;
		lambdaFailFac = 2.0f;

		const float stepSizeMinc[6] = {1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f, 1e-8f};
		const int maxIterations[6] = {5, 20, 50, 100, 100, 100};


		for (int level = 0; level < PYRAMID_LEVELS; ++ level)
		{
			lambdaInitial[level] = 0;
			stepSizeMin[level] = stepSizeMinc[level];
			convergenceEps[level] = 0.999f;
			maxItsPerLvl[level] = maxIterations[level];
		}

		lambdaInitialTestTrack = 0;
		stepSizeMinTestTrack = 1e-3f;
		convergenceEpsTestTrack = 0.98f;
		maxItsTestTrack = 5;

		var_weight = 1.0;
		huber_d = 3;
	}

	float lambdaSuccessFac;
	float lambdaFailFac;
	float lambdaInitial[PYRAMID_LEVELS];
	float stepSizeMin[PYRAMID_LEVELS];
	float convergenceEps[PYRAMID_LEVELS];
	int maxItsPerLvl[PYRAMID_LEVELS];

	float lambdaInitialTestTrack;
	float stepSizeMinTestTrack;
	float convergenceEpsTestTrack;
	float maxItsTestTrack;

	float huber_d;
	float var_weight;
};

}
