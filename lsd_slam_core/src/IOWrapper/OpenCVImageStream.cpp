#include "OpenCVImageStream.hpp"
#include "Util/Undistorter.hpp"
#include "Win32Compatibility.hpp"
#include "CameraModel/ProjCameraModel.hpp"
#include <iostream>
#include "IOWrapper/ImageDisplay.hpp"
#include "ImageViewer.hpp"

const size_t NOTIFY_BUFFER_SIZE = 16;

namespace lsd_slam {

OpenCVImageStream::OpenCVImageStream()
	:undistorter_(nullptr), running_(false), hasCalib_(false),
	dropFrames(true), undistortedImageViewer_(nullptr),
	rawImageViewer_(nullptr)
{
	imageBuffer = new NotifyBuffer<TimestampedMat>(
		NOTIFY_BUFFER_SIZE);
}

OpenCVImageStream::~OpenCVImageStream() throw()
{
	stop();
	delete imageBuffer;
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

bool OpenCVImageStream::running()
{
	return running_;
}

void OpenCVImageStream::stop()
{
	running_ = false;
	if (thread_->joinable()) {
		thread_->join();
	}
}

void OpenCVImageStream::setCalibration(const std::string &file)
{
	undistorter_.reset(Undistorter::getUndistorterForFile(file.c_str()));
	model = CameraModel::loadFromFile(file);
	this->cap_.set(CV_CAP_PROP_FRAME_WIDTH , undistorter_->getInputWidth ());
	this->cap_.set(CV_CAP_PROP_FRAME_HEIGHT, undistorter_->getInputHeight());

	if (!undistorter_ || !model)
	{
		throw std::runtime_error("Unable to read camera calibration from file!");
	}

	hasCalib_ = true;
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
			if (rawImageViewer_) {
				rawImageViewer_->setImage(rawFrame);
			}
			usleep(33000);
			if (undistorter_) {
				undistorter_->undistort(rawFrame, newFrame.data);
			} else {
				newFrame.data = rawFrame;
			}
			if(undistortedImageViewer_) {
				undistortedImageViewer_->setImage(newFrame.data);
			}
			if(!imageBuffer->pushBack(newFrame)) {
				std::cout << "Frame dropped!\n";
			}
		} else {
			std::cout << "No new frames available; terminating OpenCVImageStream..." << std::endl;
			running_ = false;
		}
	}
	std::cout << "Ending image retrieving thread..." << std::endl;
}


void OpenCVImageStream::undistortedImageViewer(ImageViewer *v)
{
	undistortedImageViewer_ = v;
}

}
