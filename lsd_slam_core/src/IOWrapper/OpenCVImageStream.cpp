#include "OpenCVImageStream.hpp"
#include "util/Undistorter.hpp"

namespace lsd_slam {


OpenCVImageStream::OpenCVImageStream()
:undistorter_(nullptr)
{

}

OpenCVImageStream::~OpenCVImageStream() throw()
{
	stop();
}

void OpenCVImageStream::run()
{
	running = true;
	thread_.reset(new std::thread(&OpenCVImageStream::operator(), this));
}

void OpenCVImageStream::stop()
{
	running = false;
	thread_->join();
	thread_.reset(nullptr);
}

void OpenCVImageStream::setCalibration(std::string file)
{
	//TODO make undistorter using calibration from file.
}

void OpenCVImageStream::setCameraCapture(cv::VideoCapture cap)
{
	this->cap_ = cap;
}

void OpenCVImageStream::operator()()
{
	while(running) {
		
	}
}

}
