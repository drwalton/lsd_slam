#include "OpenCVImageStream.hpp"
#include "Util/Undistorter.hpp"
#include "System/Win32Compatibility.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include <iostream>
#include "IOWrapper/ImageDisplay.hpp"
#include "Viewer/ImageViewer.hpp"


namespace lsd_slam {

OpenCVImageStream::OpenCVImageStream()
{}

OpenCVImageStream::~OpenCVImageStream() throw()
{
	stop();
}

void OpenCVImageStream::run()
{
	if (!cap_.isOpened()) {
		throw std::runtime_error("Cannot run OpenCVImageStream: video capture"
			" has not been opened!");
	}
	cap_.grab();
	running_ = true;
	if (!hasCalib_) throw std::runtime_error(
		"OpenCVImageStream has had no calibration file supplied!");
	thread_.reset(new std::thread(&OpenCVImageStream::operator(), this));
}

void OpenCVImageStream::setCalibration(const std::string &file)
{
	InputImageStream::setCalibration(file);
	this->cap_.set(CV_CAP_PROP_FRAME_WIDTH , undistorter_->getInputWidth ());
	this->cap_.set(CV_CAP_PROP_FRAME_HEIGHT, undistorter_->getInputHeight());
	//TODO check size is correct.
}

cv::VideoCapture &OpenCVImageStream::capture()
{
	return this->cap_;
}


void OpenCVImageStream::operator()()
{
	std::cout << "Starting image retrieving thread..." << std::endl;
	while(running_) {
		TimestampedMat newFrame;
		newFrame.timestamp = Timestamp::now();
		if(imageBuffer->isFull() && !dropFrames) {
			//Wait to pull from the capture device until there is space in the
			// buffer.
			usleep(33000);
			continue;
		}
		
		if (cap_.grab()) {
			static cv::Mat rawFrame;
			cap_.retrieve(rawFrame);
			if (undistorter_) {
				undistorter_->undistort(rawFrame, newFrame.data);
			} else {
				newFrame.data = rawFrame;
			}

			tryToShowImages(rawFrame, newFrame.data);

			if(!imageBuffer->pushBack(newFrame)) {
				std::cout << "Frame dropped!\n";
			}
			usleep(33000);
		} else {
			std::cout << "No new frames available; terminating OpenCVImageStream..." << std::endl;
			running_ = false;
		}
	}
	std::cout << "Ending image retrieving thread..." << std::endl;
}

}
