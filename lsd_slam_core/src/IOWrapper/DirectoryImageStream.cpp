#include "DirectoryImageStream.hpp"
#include "Util/Undistorter.hpp"
#include "ImageViewer.hpp"
#include "Win32Compatibility.hpp"
#include <boost/filesystem.hpp>

namespace lsd_slam {

struct DirectoryImageStream::Impl {
	boost::filesystem::directory_iterator imFileIter;
	boost::filesystem::directory_iterator endFileIter;
};

DirectoryImageStream::DirectoryImageStream()
	:pimpl_(new Impl()), dirOpened_(false)
{}

DirectoryImageStream::~DirectoryImageStream() throw()
{
	stop();
}



void DirectoryImageStream::run()
{
	if (!hasCalib_) throw std::runtime_error(
		"DirectoryImageStream has had no calibration file supplied!");
	thread_.reset(new std::thread(&DirectoryImageStream::operator(), this));
	running_ = true;
}

void DirectoryImageStream::setCalibration(const std::string &file)
{
	InputImageStream::setCalibration(file);
	//TODO check size is correct.
}

void DirectoryImageStream::operator()()
{
	if (!dirOpened_) {
		throw std::runtime_error("Cannot start DirectoryImageStream until dir is "
			"opened!");
	}

	std::cout << "Starting image retrieving thread..." << std::endl;
	while(running_) {
		TimestampedMat newFrame;
		newFrame.timestamp = Timestamp::now();
		if(imageBuffer->isFull() && !dropFrames) {
			//Wait to load new images until there is space in the buffer.
			usleep(33000);
			continue;
		}
		
		if (pimpl_->imFileIter != pimpl_->endFileIter) {
			if (boost::filesystem::is_regular_file(*(pimpl_->imFileIter))) {
				static cv::Mat rawFrame;
				rawFrame = cv::imread(pimpl_->imFileIter->path().string());

				if (rawFrame.size() != cv::Size(
					undistorter_->getInputWidth(), undistorter_->getInputHeight())) {
					throw std::runtime_error("Image found in directory not of"
						" required size!");
				}

				if (undistorter_) {
					undistorter_->undistort(rawFrame, newFrame.data);
				}
				else {
					newFrame.data = rawFrame;
				}
				tryToShowImages(rawFrame, newFrame.data);
				if (!imageBuffer->pushBack(newFrame)) {
					std::cout << "Frame dropped!\n";
				}
				usleep(33000);
			}
			++pimpl_->imFileIter;
		} else {
			std::cout << "No new frames available; terminating ..." << std::endl;
			running_ = false;
		}
	}
	std::cout << "Ending image retrieving thread..." << std::endl;
}

void DirectoryImageStream::openDirectory(const std::string &dir)
{
	boost::filesystem::path path(dir);
	if (!boost::filesystem::is_directory(path)) {
		throw std::runtime_error("Path supplied to DirectoryImageStream::"
			"openDirectory() is not a directory!");
	}
	pimpl_->imFileIter = boost::filesystem::directory_iterator(path);

	dirOpened_ = true;
}

}
