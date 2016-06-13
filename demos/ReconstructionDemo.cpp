#include <qapplication.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/ViewerOutput3DWrapper.hpp"
#include "System/LiveSLAMWrapper.hpp"
#include "System/Win32Compatibility.hpp"

int main(int argc, char **argv)
{
	if (argc < 3) {
		std::cout << "Usage: ReconstructionDemo: [calibrationfile] [input]\n"
			"Input can either be a video file, or a number, specifying a live camera." << std::endl;
		return 1;
	}

	lsd_slam::OpenCVImageStream stream;
	stream.capture().open(argv[2]);

	stream.setCalibration(argv[1]);
	stream.run();

	std::unique_ptr<lsd_slam::ViewerOutput3DWrapper> outWrapper(
		new lsd_slam::ViewerOutput3DWrapper(true, 640, 480));
	lsd_slam::LiveSLAMWrapper slamWrapper(&stream, outWrapper.get());

	slamWrapper.Loop();
	
	stream.stop();
	
	return 0;
}
