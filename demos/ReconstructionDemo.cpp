#include <qapplication.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/ViewerOutput3DWrapper.hpp"
#include "LiveSLAMWrapper.hpp"
#include <FL/Fl_Native_File_Chooser.H>
#include "Win32Compatibility.hpp"
#include "util/settings.hpp"

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cout << "Need cailbration file!" << std::endl;
		WIN_WAIT_BEFORE_EXIT
			return 1;
	}

	lsd_slam::OpenCVImageStream stream;

	if (argc >= 3) {
		stream.capture().open(lsd_slam::resourcesDir() + argv[2]);
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
				stream.capture().open(i);
				if (stream.capture().isOpened()) {
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
				stream.capture().open(lsd_slam::resourcesDir() + filename);
				stream.capture() = cv::VideoCapture(lsd_slam::resourcesDir() + filename);
				if (stream.capture().isOpened()) {
					break;
				}
				else {
					std::cout << "Unable to open file \"" << lsd_slam::resourcesDir() + filename << "\"." << std::endl;
				}
			}
			else {
				std::cout << "Could not parse response; please try again..." << std::endl;
			}

			std::string filename;
			Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
			ch.title("Select calibration file");
			ch.show();
		}
	}
	if (argc > 2) {
		stream.setCalibration(lsd_slam::resourcesDir() + argv[1]);
	} else {
		 std::string filename;
		 Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
		 ch.title("Select calibration file");
		 ch.show();
		 stream.setCalibration(lsd_slam::resourcesDir() + ch.filename());
	}

	stream.run();
	std::unique_ptr<lsd_slam::ViewerOutput3DWrapper> outWrapper(
		new lsd_slam::ViewerOutput3DWrapper(true, 640, 480));
	lsd_slam::LiveSLAMWrapper slamWrapper(&stream, outWrapper.get());

	slamWrapper.Loop();
	
	stream.stop();
	
	WIN_WAIT_BEFORE_EXIT
	return 0;
}
