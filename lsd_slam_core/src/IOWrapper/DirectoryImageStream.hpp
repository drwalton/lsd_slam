#pragma once

#include "InputImageStream.hpp"
#include <thread>

namespace lsd_slam {

class Undistorter;
class ImageViewer;

class DirectoryImageStream : public InputImageStream
{
public:
	DirectoryImageStream();
	virtual ~DirectoryImageStream() throw();

	virtual void run();

	virtual void setCalibration(const std::string &file);

	void operator()();
	
	void openDirectory(const std::string &dir);

private:
	struct Impl;
	std::unique_ptr<Impl> pimpl_;
	bool dirOpened_;
};

}
