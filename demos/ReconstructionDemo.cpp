#include <qapplication.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/ViewerOutput3DWrapper.hpp"
#include "LiveSLAMWrapper.hpp"
#include "SlamSystem.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include <FL/Fl_Native_File_Chooser.H>
#include "Win32Compatibility.hpp"
#include "Util/settings.hpp"
#include "ImageViewer.hpp"

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cout << "Need cailbration file!" << std::endl;
		WIN_WAIT_BEFORE_EXIT
			return 1;
	}

	std::unique_ptr<lsd_slam::InputImageStream> stream;

	if (argc >= 3) {
		stream = lsd_slam::InputImageStream::openImageStream(
			lsd_slam::resourcesDir() + argv[2]);
		stream->dropFrames = false;
	}
	else {
		while (true) {
			std::cout << "Please select video source ([C]amera, [F]ile): ";
			std::string selection;
			std::cin >> selection;
			if (selection[0] == 'C' || selection[0] == 'c') {
				int i;
				std::cout << "Please enter number of camera (0=default): ";
				std::cin >> i;
				lsd_slam::OpenCVImageStream *s = new lsd_slam::OpenCVImageStream();
				s->capture().open(i);
				stream.reset(s);
				if (s->capture().isOpened()) {
					break;
				}
				else {
					std::cout << "Unable to open video device " << i << std::endl;
				}
			}
			else if (selection[0] == 'F' || selection[0] == 'f') {
				std::string filename;
				Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
				ch.title("Select video file");
				ch.show();
				filename = pathToForwardSlashes(ch.filename());

				stream = lsd_slam::InputImageStream::openImageStream(
					filename);
				stream->dropFrames = false;
				break;
			}
			else {
				std::cout << "Could not parse response; please try again..." << std::endl;
			}
		}
	}
	if (argc >= 2) {
		stream->setCalibration(lsd_slam::resourcesDir() + argv[1]);
		std::cout << "Calibration file loaded: " << argv[1] <<
			"\n Width: " << stream->camModel().w << ", Height: " <<
			stream->camModel().h << std::endl;

	} else {
		 std::string filename;
		 Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
		 ch.title("Select calibration file");
		 ch.show();
		 stream->setCalibration(ch.filename());
	}

	stream->run();
	QApplication qapp(argc, argv);
	lsd_slam::ViewerOutput3DWrapper outWrapper(true, 640, 480);
	lsd_slam::LiveSLAMWrapper slamWrapper(stream.get(), &outWrapper);
	lsd_slam::ImageViewer depthMapViewer("Est. Depth Map");
	lsd_slam::ImageViewer inputImageViewer("Input (Undistorted)");
	//lsd_slam::ImageViewer rawInputImageViewer("Input (Raw)");
	slamWrapper.saveKeyframeCloudsToDisk(true);
	slamWrapper.depthMapImageViewer(&depthMapViewer);
	//slamWrapper.getSlamSystem()->depthMapSettings().saveAllFramesAsPointClouds
	//	= true;
	//slamWrapper.getSlamSystem()->depthMapSettings().saveAllFramesAsVectorClouds
	//	= true;
	slamWrapper.getSlamSystem()->depthMapSettings().saveMatchesImages
		= true;
	slamWrapper.getSlamSystem()->depthMapSettings().printPropagationStatistics
		= true;
	stream->undistortedImageViewer(&inputImageViewer);
	stream->dropFrames = false;
	//stream->rawImageViewer(&rawInputImageViewer);
	

	slamWrapper.start();
	qapp.exec();
	slamWrapper.stop();
	
	stream->stop();
	return 0;
}
