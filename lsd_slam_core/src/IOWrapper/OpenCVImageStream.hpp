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
	
	virtual void setCalibration(const std::string &file);

	cv::VideoCapture &capture();

	void operator()();

private:
	cv::VideoCapture cap_;
	bool running_, hasCalib_;
};

}
