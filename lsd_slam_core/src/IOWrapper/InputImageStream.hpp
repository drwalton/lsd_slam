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

#include <string>
#include "IOWrapper/NotifyBuffer.hpp"
#include "IOWrapper/TimestampedObject.hpp"
#include "CameraModel/CameraModel.hpp"

namespace lsd_slam
{

/**
 * Virtual ImageStream. Can be from OpenCV's ImageCapture, ROS or Android.
 * Also has to provide the camera calibration for that stream, as well as the respective undistorter object (if required).
 * Runs in it's own thread, and has a NotifyBuffer, in which received images are stored.
 */
class InputImageStream
{
public:
	explicit InputImageStream() :dropFrames(true) {};
	virtual ~InputImageStream() {};
	
	/**
	 * Starts the thread.
	 */
	virtual void run() {};

	virtual bool running() { return true; };

	virtual void setCalibration(const std::string &file) { model = CameraModel::loadFromFile(file); };

	///Defines the behaviour when the frame buffer is full (i.e. the processing
	/// thread is not pulling images quickly enough). If true, frames will be
	/// pulled from the capture device but not added to the queue. If false,
	/// frames will only be pulled when there is space in the buffer (useful for
	/// offline processing of videos).
	bool dropFrames;

	/**
	 * Gets the NotifyBuffer to which incoming images are stored.
	 */
	inline NotifyBuffer<TimestampedMat>* getBuffer() {return imageBuffer;};

	const CameraModel &camModel() const { return *model; };

protected:
	std::unique_ptr<CameraModel> model;
	NotifyBuffer<TimestampedMat>* imageBuffer;
};
}
