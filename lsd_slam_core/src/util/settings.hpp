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

#include <string.h>
#include <string>



namespace lsd_slam
{

std::string resourcesDir();

#define ALIGN __attribute__((__aligned__(16)))
#define SSEE(val,idx) (*(((float*)&val)+idx))
#define DIVISION_EPS 1e-10f
#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))

#if defined(ENABLE_SSE)
	#define USESSE true
#else
	#define USESSE false
#endif


#if defined(NDEBUG)
	#define enablePrintDebugInfo false
#else
	#define enablePrintDebugInfo true
#endif

/** ============== constants for validity handeling ======================= */

// validity can take values between 0 and X, where X depends on the abs. gradient at that location:
// it is calculated as VALIDITY_COUNTER_MAX + (absGrad/255)*VALIDITY_COUNTER_MAX_VARIABLE
#define VALIDITY_COUNTER_MAX (5.0f)		// validity will never be higher than this
#define VALIDITY_COUNTER_MAX_VARIABLE (250.0f)		// validity will never be higher than this

#define VALIDITY_COUNTER_INC 5		// validity is increased by this on sucessful stereo
#define VALIDITY_COUNTER_DEC 5		// validity is decreased by this on failed stereo
#define VALIDITY_COUNTER_INITIAL_OBSERVE 5	// initial validity for first observations


#define VAL_SUM_MIN_FOR_CREATE (30) // minimal summed validity over 5x5 region to create a new hypothesis for non-blacklisted pixel (hole-filling)
#define VAL_SUM_MIN_FOR_KEEP (24) // minimal summed validity over 5x5 region to keep hypothesis (regularization)
#define VAL_SUM_MIN_FOR_UNBLACKLIST (100) // if summed validity surpasses this, a pixel is un-blacklisted.

#define MIN_BLACKLIST -1	// if blacklist is SMALLER than this, pixel gets ignored. blacklist starts with 0.


// ============== RE-LOCALIZATION, KF-REACTIVATION etc. ======================
// defines the level on which we do the quick tracking-check for relocalization.

#define MAX_DIFF_CONSTANT (40.0f*40.0f)
#define MAX_DIFF_GRAD_MULT (0.5f*0.5f)

#define MIN_GOODPERGOODBAD_PIXEL (0.5f)
#define MIN_GOODPERALL_PIXEL (0.04f)
#define MIN_GOODPERALL_PIXEL_ABSMIN (0.01f)

#define INITIALIZATION_PHASE_COUNT 5

#define MIN_NUM_MAPPED 5

// settings variables
// controlled via keystrokes
extern bool autoRun;
extern bool autoRunWithinFrame;
extern int debugDisplay;
extern bool displayDepthMap;
extern bool onSceenInfoDisplay;
extern bool dumpMap;
extern bool doFullReConstraintTrack;

// dyn config
extern bool printThreadingInfo;

extern bool printKeyframeSelectionInfo;
extern bool printNonKeyframeInfo;
extern bool printConstraintSearchInfo;
extern bool printOptimizationInfo;
extern bool printRelocalizationInfo;

extern bool printFrameBuildDebugInfo;
extern bool printMemoryDebugInfo;

extern bool printMappingTiming;
extern bool printOverallTiming;
extern bool plotStereoImages;

extern bool allowNegativeIdepths;
extern bool useMotionModel;
extern bool useSubpixelStereo;
extern bool multiThreading;
extern bool useAffineLightningEstimation;

extern float freeDebugParam1;
extern float freeDebugParam2;
extern float freeDebugParam3;
extern float freeDebugParam4;
extern float freeDebugParam5;

extern float KFDistWeight;
extern float KFUsageWeight;
extern int maxLoopClosureCandidates;
extern int propagateKeyFrameDepthCount;
extern float loopclosureStrictness;
extern float relocalizationTH;

extern float minUseGrad;
extern float cameraPixelNoise2;
extern float depthSmoothingFactor;

extern bool useFabMap;
extern bool doSlam;
extern bool doKFReActivation;
extern bool doMapping;

extern bool saveKeyframes;
extern bool saveAllTracked;
extern bool saveLoopClosureImages;
extern bool saveAllTrackingStages;
extern bool saveAllTrackingStagesInternal;

extern bool continuousPCOutput;


/// Relative path of calibration file, map saving directory etc. for live_odometry
extern std::string packagePath;

extern bool fullResetRequested;
extern bool manualTrackingLossIndicated;

void handleKey(char k);

}
