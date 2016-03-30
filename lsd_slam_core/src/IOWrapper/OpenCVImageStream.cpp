#include "OpenCVImageStream.hpp"
#include "util/Undistorter.hpp"

const size_t NOTIFY_BUFFER_SIZE = 8;

namespace lsd_slam {

OpenCVImageStream::OpenCVImageStream()
	:undistorter_(nullptr), running_(false), hasCalib_(false)
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
	running_ = true;
	if (!hasCalib_) throw std::runtime_error(
		"OpenCVImageStream has had no calibration file supplied!");
	thread_.reset(new std::thread(&OpenCVImageStream::operator(), this));
}

void OpenCVImageStream::stop()
{
	running_ = false;
	thread_->join();
	thread_.reset(nullptr);
}

void OpenCVImageStream::setCalibration(const std::string &file)
{
	undistorter_.reset(Undistorter::getUndistorterForFile(file.c_str()));

	if (!undistorter_)
	{
		throw std::runtime_error("Unable to read camera calibration from file!");
	}

	fx_ = float(undistorter_->getK().at<double>(0, 0));
	fy_ = float(undistorter_->getK().at<double>(1, 1));
	cx_ = float(undistorter_->getK().at<double>(2, 0));
	cy_ = float(undistorter_->getK().at<double>(2, 1));

	width_ = undistorter_->getOutputWidth();
	height_ = undistorter_->getOutputHeight();
	
	hasCalib_ = true;
}

cv::VideoCapture &OpenCVImageStream::capture()
{
	return this->cap_;
}

void OpenCVImageStream::operator()()
{
	while(running_) {
		TimestampedMat newFrame;
		newFrame.timestamp = Timestamp::now();
		if (cap_.grab()) {
			static cv::Mat rawFrame;
			cap_.retrieve(rawFrame);
			cv::imshow("CVStream", rawFrame);
			cv::waitKey(1);
			if (undistorter_) {
				undistorter_->undistort(rawFrame, newFrame.data);
			} else {
				newFrame.data = rawFrame;
			}
			imageBuffer->pushBack(newFrame);
		} else {

		}
	}
}

}
