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

#include "LiveSLAMWrapper.hpp"
#include <vector>
#include <thread>
#include "Util/SophusUtil.hpp"

#include "SlamSystem.hpp"

#include "IOWrapper/ImageDisplay.hpp"
#include "IOWrapper/Output3DWrapper.hpp"
#include "IOWrapper/InputImageStream.hpp"
#include "Util/globalFuncs.hpp"

#include <iostream>
#include <cstdio>
#include "Win32Compatibility.hpp"
#include <opencv2/opencv.hpp>

namespace lsd_slam
{

LiveSLAMWrapper::LiveSLAMWrapper(InputImageStream* imageStream, 
	Output3DWrapper* outputWrapper)
	:model(imageStream->camModel().clone())
{
	this->imageStream = imageStream;
	this->outputWrapper = outputWrapper;
	imageStream->getBuffer()->setReceiver(this);

	outFileName = packagePath+"estimated_poses.txt";

	isInitialized = false;

	outFile = nullptr;

	// make Odometry
	monoOdometry = new SlamSystem(*model, doSlam);

	monoOdometry->setVisualization(outputWrapper);

	imageSeqNumber = 0;
}


LiveSLAMWrapper::~LiveSLAMWrapper()
{
	if(monoOdometry != 0)
		delete monoOdometry;
	if(outFile != 0)
	{
		outFile->flush();
		outFile->close();
		delete outFile;
	}
}

void LiveSLAMWrapper::start()
{
	running = true;
	paused = false;
	slamThread = std::thread([&](){
		std::cout << "Starting live SLAM thread..." << std::endl;
		while (running) {
			boost::unique_lock<boost::recursive_mutex> waitLock(imageStream->getBuffer()->getMutex());
			while (running && !fullResetRequested && !(imageStream->getBuffer()->size() > 0)) {
				notifyCondition.wait(waitLock);
				while (paused) {

				}
			}
			waitLock.unlock();


			if (fullResetRequested)
			{
				resetAll();
				fullResetRequested = false;
				if (!(imageStream->getBuffer()->size() > 0))
					continue;
			}

			if (!imageStream->running()) break;
			TimestampedMat image = imageStream->getBuffer()->first();
			imageStream->getBuffer()->popFront();

			// process image
			//Util::displayImage("MyVideo", image.data);
			newImageCallback(image.data, image.timestamp);
		}
		std::cout << "Finishing live SLAM thread..." << std::endl;
	});
}

void LiveSLAMWrapper::stop() 
{
	running = false;
	//
	slamThread.join();
}


void LiveSLAMWrapper::newImageCallback(const cv::Mat& img, Timestamp imgTime)
{
	++ imageSeqNumber;

	// Convert image to grayscale, if necessary
	cv::Mat grayImg;
	if (img.channels() == 1)
		grayImg = img;
	else
		cvtColor(img, grayImg, CV_RGB2GRAY);
	
	// Assert that we work with 8 bit images
	assert(grayImg.elemSize() == 1);
	//assert(fx != 0 || fy != 0);
	assert(model->fx != 0 && model->fy != 0);


	// need to initialize
	if(!isInitialized)
	{
		monoOdometry->randomInit(grayImg.data, imgTime.toSec(), 1);
		isInitialized = true;
	}
	else if(isInitialized && monoOdometry != nullptr)
	{
		monoOdometry->trackFrame(grayImg.data,imageSeqNumber,false,imgTime.toSec());
	}
}

void LiveSLAMWrapper::logCameraPose(const SE3& camToWorld, double time)
{
	Sophus::Quaternionf quat = camToWorld.unit_quaternion().cast<float>();
	Eigen::Vector3f trans = camToWorld.translation().cast<float>();

	char buffer[1000];

	int num = snprintf(buffer, 1000, "%f %f %f %f %f %f %f %f\n",
			time,
			trans[0],
			trans[1],
			trans[2],
			quat.x(),
			quat.y(),
			quat.z(),
			quat.w());

	if(outFile == 0)
		outFile = new std::ofstream(outFileName.c_str());
	outFile->write(buffer,num);
	outFile->flush();
}

void LiveSLAMWrapper::requestReset()
{
	fullResetRequested = true;
	notifyCondition.notify_all();
}

void LiveSLAMWrapper::resetAll()
{
	if(monoOdometry != nullptr)
	{
		delete monoOdometry;
		printf("Deleted SlamSystem Object!\n");

		monoOdometry = new SlamSystem(*model, doSlam);
		monoOdometry->setVisualization(outputWrapper);

	}
	imageSeqNumber = 0;
	isInitialized = false;

	Util::closeAllWindows();

}

bool LiveSLAMWrapper::saveKeyframeCloudsToDisk() const
{
	return outputWrapper->saveKeyframeCloudsToDisk();
}
void LiveSLAMWrapper::saveKeyframeCloudsToDisk(bool b)
{
	outputWrapper->saveKeyframeCloudsToDisk(b);
}

void LiveSLAMWrapper::depthMapImageViewer(lsd_slam::ImageViewer *v)
{
	monoOdometry->depthMapImageViewer(v);
}

}
