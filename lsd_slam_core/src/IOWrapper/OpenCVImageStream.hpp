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

	virtual bool running();
	
	void stop();

	virtual void setCalibration(const std::string &file);

	cv::VideoCapture &capture();

	void operator()();
	bool showRawStream() const;
	void showRawStream(bool s);
	bool showUndistortedStream() const;
	void showUndistortedStream(bool s);

private:
	cv::VideoCapture cap_;
	std::unique_ptr<Undistorter> undistorter_;
	std::unique_ptr<std::thread> thread_;
	bool running_, hasCalib_, showRawStream_, showUndistortedStream_;
};

}

#endif
