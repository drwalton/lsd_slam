#pragma once

#include "InputImageStream.hpp"
#include <opencv2/opencv.hpp>
#include <thread>

namespace lsd_slam {

class Undistorter;
class ImageViewer;

class OpenCVImageStream : public InputImageStream
{
public:
	OpenCVImageStream();
	virtual ~OpenCVImageStream() throw();

	virtual void run();

	virtual bool running();
	
	void stop();

	virtual void setCalibration(const std::string &file);

	cv::VideoCapture &capture();

	void operator()();
	
	///Defines the behaviour when the frame buffer is full (i.e. the processing
	/// thread is not pulling images quickly enough). If true, frames will be
	/// pulled from the capture device but not added to the queue. If false,
	/// frames will only be pulled when there is space in the buffer (useful for
	/// offline processing of videos).
	bool dropFrames;
	
	void undistortedImageViewer(ImageViewer *v);

private:
	cv::VideoCapture cap_;
	std::unique_ptr<Undistorter> undistorter_;
	std::unique_ptr<std::thread> thread_;
	bool running_, hasCalib_;
	ImageViewer *undistortedImageViewer_, *rawImageViewer_;
};

}
