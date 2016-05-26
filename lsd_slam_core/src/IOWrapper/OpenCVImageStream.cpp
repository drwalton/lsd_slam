#include "OpenCVImageStream.hpp"
#include "util/Undistorter.hpp"
#include "Win32Compatibility.hpp"
#include "ProjCameraModel.hpp"

const size_t NOTIFY_BUFFER_SIZE = 16;

namespace lsd_slam {

OpenCVImageStream::OpenCVImageStream()
	:undistorter_(nullptr), running_(false), hasCalib_(false),
	showRawStream_(false), showUndistortedStream_(true)
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
	thread_->join();
	thread_.reset(nullptr);
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

bool OpenCVImageStream::showRawStream() const
{
	return showRawStream_;
}

void OpenCVImageStream::showRawStream(bool s)
{
	showRawStream_ = s;
}
bool OpenCVImageStream::showUndistortedStream() const
{
	return showUndistortedStream_;
}

void OpenCVImageStream::showUndistortedStream(bool s)
{
	showUndistortedStream_ = s;
}

void OpenCVImageStream::operator()()
{
	while(running_) {
		TimestampedMat newFrame;
		newFrame.timestamp = Timestamp::now();
		if (cap_.grab()) {
			static cv::Mat rawFrame;
			cap_.retrieve(rawFrame);
			if (showRawStream_) {
				cv::imshow("OpenCVImageStream (Raw)", rawFrame);
				cv::waitKey(1);
			}
			usleep(33000);
			if (undistorter_) {
				undistorter_->undistort(rawFrame, newFrame.data);
				if (showUndistortedStream_) {
					cv::imshow("OpenCVImageStream (Undistorted)", newFrame.data);
					cv::waitKey(1);
				}
			} else {
				newFrame.data = rawFrame;
			}
			if(!imageBuffer->pushBack(newFrame)) {
				std::cout << "Frame dropped!\n";
			}
		} else {
			std::cout << "No new frames available; terminating OpenCVImageStream..." << std::endl;
			cv::destroyWindow("OpenCVImageStream");
			running_ = false;
		}
	}
}

}
