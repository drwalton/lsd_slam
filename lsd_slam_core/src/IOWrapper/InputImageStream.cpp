#include "InputImageStream.hpp"
#include "ImageViewer.hpp"
#include "Util/Undistorter.hpp"
#include <boost/filesystem.hpp>
#include "OpenCVImageStream.hpp"
#include "DirectoryImageStream.hpp"

namespace lsd_slam
{

const size_t NOTIFY_BUFFER_SIZE = 16;

InputImageStream::InputImageStream()
	:dropFrames(true),
	undistortedImageViewer_(nullptr), rawImageViewer_(nullptr),
	hasCalib_(false), running_(false)
{
	imageBuffer.reset(new NotifyBuffer<TimestampedMat>(
		NOTIFY_BUFFER_SIZE));
}
InputImageStream::~InputImageStream() throw()
{}
	
void InputImageStream::run() 
{}

bool InputImageStream::running()
{ 
	return running_; 
}

void InputImageStream::setCalibration(const std::string &file) 
{	
	model = CameraModel::loadFromFile(file); 
	undistorter_.reset(Undistorter::getUndistorterForFile(file.c_str()));
	if (!undistorter_ || !model)
	{
		throw std::runtime_error("Unable to read camera calibration from file!");
	}
	hasCalib_ = true;
}

const CameraModel &InputImageStream::camModel() const 
{	
	return *model; 
}

void InputImageStream::undistortedImageViewer(ImageViewer *v)
{
	undistortedImageViewer_ = v;
}
void InputImageStream::rawImageViewer(ImageViewer *v)
{
	rawImageViewer_ = v;
}

void InputImageStream::tryToShowImages(
	const cv::Mat &rawIm, const cv::Mat &undistortedIm)
{
	if (rawImageViewer_) {
		rawImageViewer_->setImage(rawIm);
	}
	if (undistortedImageViewer_) {
		undistortedImageViewer_->setImage(undistortedIm);
	}
}

std::unique_ptr<InputImageStream> InputImageStream::openImageStream(
	const std::string &pathname)
{
	boost::filesystem::path path(pathname);
	if (boost::filesystem::is_directory(path)) {
		DirectoryImageStream *stream = new DirectoryImageStream();
		stream->openDirectory(pathname);
		return std::unique_ptr<InputImageStream>(stream);
	} else if (boost::filesystem::is_regular_file(path)) {
		OpenCVImageStream *stream = new OpenCVImageStream();
		stream->capture().open(pathname);
		return std::unique_ptr<InputImageStream>(stream);
	} else {
		throw std::runtime_error("Supplied path is neither a directory"
			" nor a regular file.");
	}
}

void InputImageStream::stop()
{
	running_ = false;
	if (thread_->joinable()) {
		thread_->join();
	}
}

}
