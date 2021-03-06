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

#include "Util/settings.hpp"
#include <opencv2/opencv.hpp>
#include <boost/bind.hpp>
#include "DepthEstimation/DepthMapDebugDefines.hpp"
#include "CameraModel/CameraModel.hpp"
#include "globalFuncs.hpp"


namespace lsd_slam
{
RunningStats runningStats;

std::string resourcesDir() {
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
	static std::string dir = getenv("HOMEDRIVE") + std::string(getenv("HOMEPATH")) + "/Documents/lsd_slam_new/resources/";
#else
	static std::string dir = getenv("HOME") + std::string("/lsd_slam_new/resources/");
#endif
	return dir;
}

void makeDebugDirectories(const CameraModel *m)
{
	std::cout << "Clearing debug directories..."; std::cout.flush();
#if DEBUG_SAVE_VAR_IMS
	makeEmptyDirectory(resourcesDir() + "VarIms" + m->getTypeName() +  "/");
#endif
#if DEBUG_SAVE_IDEPTH_IMS
	makeEmptyDirectory(resourcesDir() + "DepthIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_MATCH_IMS
	makeEmptyDirectory(resourcesDir() + "MatchIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_SEARCH_RANGE_IMS
	makeEmptyDirectory(resourcesDir() + "RangeIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_RESULT_IMS
	makeEmptyDirectory(resourcesDir() + "ResultIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_PIXEL_DISPARITY_IMS
	makeEmptyDirectory(resourcesDir() + "PixelDispIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_FRAME_STEREO_POINT_CLOUDS
	makeEmptyDirectory(resourcesDir() + "FramePtClouds" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_KEYFRAME_PROPAGATION_CLOUDS
	makeEmptyDirectory(resourcesDir() + "KeyframePropagationPtClouds" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_GRAD_ALONG_LINE_IMS
	makeEmptyDirectory(resourcesDir() + "GradAlongLineIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_GEO_DISP_ERROR_IMS
	makeEmptyDirectory(resourcesDir() + "GeoDispErrIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_PHOTO_DISP_ERROR_IMS
	makeEmptyDirectory(resourcesDir() + "PhotoDispErrIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_DISCRETIZATION_ERROR_IMS
	makeEmptyDirectory(resourcesDir() + "DiscretizeErrIms" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_KEYFRAME_POINT_CLOUDS_EACH_FRAME
	makeEmptyDirectory(resourcesDir() + "KeyframePtClouds" + m->getTypeName() + "/");
#endif
#if DEBUG_SAVE_EPL_LENGTH_IMS
	makeEmptyDirectory(resourcesDir() + "EplLengths" + m->getTypeName() + "/");
#endif
}

bool autoRun = true;
bool autoRunWithinFrame = true;

int debugDisplay = 0;

bool onSceenInfoDisplay = true;
bool displayDepthMap = true;
bool dumpMap = false;
bool doFullReConstraintTrack = false;

// dyn config
bool printPropagationStatistics = true;
bool printFillHolesStatistics = false;
bool printObserveStatistics = false;
bool printObservePurgeStatistics = false;
bool printRegularizeStatistics = false;
bool printLineStereoStatistics = false;
bool printLineStereoFails = false;

bool printTrackingIterationInfo = false;

bool printFrameBuildDebugInfo = false;
bool printMemoryDebugInfo = false;

bool printKeyframeSelectionInfo = true;
bool printNonKeyframeInfo = false;
bool printConstraintSearchInfo = false;
bool printOptimizationInfo = false;
bool printRelocalizationInfo = false;

bool printThreadingInfo = false;
bool printMappingTiming = false;
bool printOverallTiming = false;

bool plotTrackingIterationInfo = false;
bool plotSim3TrackingIterationInfo = false;
bool plotStereoImages = false;
bool plotTracking = false;


float freeDebugParam1 = 1;
float freeDebugParam2 = 1;
float freeDebugParam3 = 1;
float freeDebugParam4 = 1;
float freeDebugParam5 = 1;

float KFDistWeight = 4;
float KFUsageWeight = 3;

float minUseGrad = 5;
float cameraPixelNoise2 = 4*4;
float depthSmoothingFactor = 1;

bool allowNegativeIdepths = true;
bool useMotionModel = false;
bool useSubpixelStereo = true;
bool multiThreading = true;
bool useAffineLightningEstimation = true;



bool useFabMap = false;
bool doSlam = true;
bool doKFReActivation = true;
bool doMapping = true;

int maxLoopClosureCandidates = 10;
int maxOptimizationIterations = 100;
int propagateKeyframeDepthCount = 0;
float loopclosureStrictness = 1.5f;
float relocalizationTH = 0.7f;


bool saveKeyframes =  false;
bool saveAllTracked =  false;
bool saveLoopClosureImages =  false;
bool saveAllTrackingStages = false;
bool saveAllTrackingStagesInternal = false;

bool continuousPCOutput = false;


bool fullResetRequested = false;
bool manualTrackingLossIndicated = false;


std::string packagePath = "";


void handleKey(char k)
{
	char kkk = k;
	switch(kkk)
	{
	case 'a': case 'A':
//		autoRun = !autoRun;		// disabled... only use for debugging & if you really, really know what you are doing
		break;
	case 's': case 'S':
//		autoRunWithinFrame = !autoRunWithinFrame; 	// disabled... only use for debugging & if you really, really know what you are doing
		break;
	case 'd': case 'D':
		debugDisplay = (debugDisplay+1)%6;
		printf("debugDisplay is now: %d\n", debugDisplay);
		break;
	case 'e': case 'E':
		debugDisplay = (debugDisplay-1+6)%6;
		printf("debugDisplay is now: %d\n", debugDisplay);
		break;
	case 'o': case 'O':
		onSceenInfoDisplay = !onSceenInfoDisplay;
		break;
	case 'r': case 'R':
		printf("requested full reset!\n");
		fullResetRequested = true;
		break;
	case 'm': case 'M':
		printf("Dumping Map!\n");
		dumpMap = true;
		break;
	case 'p': case 'P':
		printf("Tracking all Map-Frames again!\n");
		doFullReConstraintTrack = true;
		break;
	case 'l': case 'L':
		printf("Manual Tracking Loss Indicated!\n");
		manualTrackingLossIndicated = true;
		break;
	}

}

}
