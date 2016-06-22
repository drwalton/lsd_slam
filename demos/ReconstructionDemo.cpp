#include <qapplication.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/DirectoryImageStream.hpp"
#include "IOWrapper/ViewerOutput3DWrapper.hpp"
#include "System/LiveSLAMWrapper.hpp"
#include "System/Win32Compatibility.hpp"
#include "Util/settings.hpp"
#include <boost/filesystem.hpp>

int main(int argc, char **argv)
{
	if (argc < 3) {
		std::cout << "Usage: ReconstructionDemo: [calibrationfile] [input]\n"
			"Input can either be a video file, or a number, specifying a live camera." << std::endl;
		return 1;
	}

	std::string fullPathStr = lsd_slam::resourcesDir() + argv[2];
	std::unique_ptr<lsd_slam::InputImageStream> imageStream;
	boost::filesystem::path path(fullPathStr);
	std::cout << "Opening stream: \"" << argv[2] << "\"..."; std::cout.flush();
	if (boost::filesystem::is_directory(path)) {
		lsd_slam::DirectoryImageStream *dirImStream = new lsd_slam::DirectoryImageStream();
		imageStream.reset(dirImStream);
		dirImStream->dropFrames = false;
		dirImStream->openDirectory(fullPathStr);
	} else if (boost::filesystem::is_regular_file(path)) {
		lsd_slam::OpenCVImageStream *cvImStream = new lsd_slam::OpenCVImageStream();
		imageStream.reset(cvImStream);
		cvImStream->dropFrames = false;
		cvImStream->capture().open(fullPathStr);
	} else {
		//Not file or directory - assume it's a camera index.
		lsd_slam::OpenCVImageStream *cvImStream = new lsd_slam::OpenCVImageStream();
		imageStream.reset(cvImStream);
		cvImStream->dropFrames = false;
		cvImStream->capture().open(argv[2]);
	}

	std::cout << "Done!" << std::endl;

	std::cout << "Opening calibration file: \"" << argv[1] << "\"..."; std::cout.flush();
	imageStream->setCalibration(lsd_slam::resourcesDir() + argv[1]);
	std::cout << "Done!" << std::endl;
	std::cout << "Loaded calibration:\n" << imageStream->camModel() << std::endl;

	lsd_slam::makeDebugDirectories(&imageStream->camModel());

	std::cout << "Running stream..."; std::cout.flush();
	imageStream->run();
	std::cout << "Done!" << std::endl;

	QApplication qapp(argc, argv);

	std::cout << "Creating output wrapper..."; std::cout.flush();
	lsd_slam::ViewerOutput3DWrapper outWrapper(true, 640, 480);
	std::cout << "Done!" << std::endl;

	std::cout << "Creating SLAM wrapper..."; std::cout.flush();
	lsd_slam::LiveSLAMWrapper slamWrapper(imageStream.get(), &outWrapper,
		lsd_slam::LiveSLAMWrapper::ThreadingMode::SINGLE,
		lsd_slam::LiveSLAMWrapper::LoopClosureMode::ENABLED,
		lsd_slam::DepthMapInitMode::CONSTANT,
		true);

	std::cout << "Starting SLAM wrapper..." << std::endl;
	slamWrapper.start();
	std::cout << "Done!" << std::endl;
	std::cout << "Starting QT App loop..." << std::endl;
	qapp.exec();
	slamWrapper.stop();

	std::cout << "Stopping stream..."; std::cout.flush();
	imageStream->stop();
	std::cout << "Done!" << std::endl;
	
	std::cout << "Exiting program..." << std::endl;
	return 0;
}
