#ifndef LSD_SLAM_OPENCVIMAGESTREAM_HPP_INCLUDED
#define LSD_SLAM_OPENCVIMAGESTREAM_HPP_INCLUDED

#include "InputImageStream.hpp"
#include <opencv2/opencv.hpp>
#include <thread>

namespace lsd_slam {

class Undistorter;

class OpenCVImageStream : public InputImageStream
{
public:
	OpenCVImageStream();
	virtual ~OpenCVImageStream() throw();

	virtual void run();
	
	void stop();

	virtual void setCalibration(std::string file);

	virtual void setCameraCapture(cv::VideoCapture cap);

	void operator()();

private:
	cv::VideoCapture cap_;
	std::unique_ptr<Undistorter> undistorter_;
	std::unique_ptr<std::thread> thread_;
	bool running;
};

}

#endif
