#include <qapplication.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/ViewerOutput3DWrapper.hpp"
#include "System/LiveSLAMWrapper.hpp"
#include "System/Win32Compatibility.hpp"
#include "Util/settings.hpp"

int main(int argc, char **argv)
{
	if (argc < 3) {
		std::cout << "Usage: ReconstructionDemo: [calibrationfile] [input]\n"
			"Input can either be a video file, or a number, specifying a live camera." << std::endl;
		return 1;
	}

	std::cout << "Opening stream: \"" << argv[2] << "\"..."; std::cout.flush();
	lsd_slam::OpenCVImageStream stream;
	stream.dropFrames = false;
	stream.capture().open(lsd_slam::resourcesDir() + argv[2]);
	std::cout << "Done!" << std::endl;

	std::cout << "Opening calibration file: \"" << argv[1] << "\"..."; std::cout.flush();
	stream.setCalibration(lsd_slam::resourcesDir() + argv[1]);
	std::cout << "Done!" << std::endl;
	std::cout << "Loaded calibration:\n" << stream.camModel() << std::endl;

	std::cout << "Running stream..."; std::cout.flush();
	stream.run();
	std::cout << "Done!" << std::endl;

	//QApplication qapp(argc, argv);

	std::cout << "Creating output wrapper..."; std::cout.flush();
	std::unique_ptr<lsd_slam::ViewerOutput3DWrapper> outWrapper(
		new lsd_slam::ViewerOutput3DWrapper(true, 640, 480));
	std::cout << "Done!" << std::endl;

	std::cout << "Creating SLAM wrapper..."; std::cout.flush();
	{
		lsd_slam::LiveSLAMWrapper slamWrapper(&stream, outWrapper.get(),
			outWrapper->running,
			lsd_slam::LiveSLAMWrapper::ThreadingMode::SINGLE,
			lsd_slam::DepthMapInitMode::CONSTANT,
			true);
		slamWrapper.saveStereoSearchIms(true);
		slamWrapper.saveStereoResultIms(true);
		slamWrapper.plotTracking(true);
		std::cout << "Done!" << std::endl;

		std::cout << "Starting SLAM wrapper..." << std::endl;
		slamWrapper.Loop();
	}

	std::cout << "Stopping stream..."; std::cout.flush();
	stream.stop();
	std::cout << "Done!" << std::endl;
	
	std::cout << "Exiting program..." << std::endl;
	return 0;
}
