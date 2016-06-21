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
#include "Win32Compatibility.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

#include "IOWrapper/Timestamp.hpp"
#include "IOWrapper/NotifyBuffer.hpp"
#include "IOWrapper/TimestampedObject.hpp"
#include "util/SophusUtil.hpp"
#include "DepthEstimation/DepthMapInitMode.hpp"

namespace cv {
	class Mat;
}



namespace lsd_slam
{

class SlamSystem;
class LiveSLAMWrapperROS;
class InputImageStream;
class Output3DWrapper;
class CameraModel;

class LiveSLAMWrapper : public Notifiable
{
friend class LiveSLAMWrapperROS;
public:
	enum class ThreadingMode {SINGLE, MULTI};
	enum class LoopClosureMode {ENABLED, DISABLED};
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	LiveSLAMWrapper(InputImageStream* imageStream, Output3DWrapper* outputWrapper, 
		ThreadingMode threadMode = ThreadingMode::MULTI,
		LoopClosureMode loopClosureMode = LoopClosureMode::ENABLED,
		DepthMapInitMode depthMapInitMode = DepthMapInitMode::RANDOM,
		bool saveTrackingInfo = false);

	/** Destructor. */
	~LiveSLAMWrapper();
	
	
	void start();

	void stop();
	
	/** Requests a reset from a different thread. */
	void requestReset();
	
	/** Resets everything, starting the odometry from the beginning again. */
	void resetAll();

	/** Callback function for new RGB images. */
	void newImageCallback(const cv::Mat& img, Timestamp imgTime);

	/** Writes the given time and pose to the outFile. */
	void logCameraPose(const SE3& camToWorld, double time);
	
	
	inline SlamSystem* getSlamSystem() {return monoOdometry;}

	// Threading stuff
	bool blockTrackUntilMapped;

	void plotTracking(bool p);
	bool plotTracking() const;

private:
	bool running_;
	std::thread mainSlamLoopThread_;
	/** Runs the main processing loop. Will never return. */
	void Loop();
	
	InputImageStream* imageStream;
	Output3DWrapper* outputWrapper;

	// initialization stuff
	bool isInitialized;

	// monoOdometry
	SlamSystem* monoOdometry;

	std::string outFileName;
	std::ofstream* outFile;
	
	int imageSeqNumber;
	std::unique_ptr<CameraModel> camModel_;

};

}
