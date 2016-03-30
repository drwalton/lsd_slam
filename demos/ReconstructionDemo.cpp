#include <opencv2/opencv.hpp>
#include <iostream>
#include "IOWrapper/OpenCVImageStream.hpp"
#include "IOWrapper/Output3DWrapper.hpp"
#include "LiveSLAMWrapper.hpp"
#include <FL/Fl_Native_File_Chooser.H>
#include "Win32Compatibility.hpp"

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cout << "Need cailbration file!" << std::endl;
		return 1;
	}

	lsd_slam::OpenCVImageStream stream;
	
	while(true) {
    	std::cout << "Please select video source ([C]amera, [F]ile): ";
    	std::string selection;
    	std::cin >> selection;
    	if(selection[0] == 'C' || selection[0] == 'c') {
    		int i;
    		std::cout << "Please enter number of camera (0=default): ";
    		std::cin >> i;
    		stream.capture().open(i);
			if (stream.capture().isOpened()) {
    			break;
			} else {
				std::cout << "Unable to open video device " << i << std::endl;
			}
    	} else if(selection[0] == 'F' || selection[0] == 'f') {
			std::string filename;
			Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
			ch.title("Load first image");
			ch.show();
			filename = pathToForwardSlashes(ch.filename());
			stream.capture().open(filename);
			stream.capture() = cv::VideoCapture(filename);
			if (stream.capture().isOpened()) {
				break;
			} else {
				std::cout << "Unable to open file \"" << filename << "\"." << std::endl;
			}
		} else {
			std::cout << "Could not parse response; please try again..." << std::endl;
		}
	}

	stream.setCalibration(argv[1]);
	stream.run();

	lsd_slam::Output3DWrapper outWrapper;
	lsd_slam::LiveSLAMWrapper slamWrapper(&stream, &outWrapper);

	slamWrapper.Loop();
	
	stream.stop();
	
	return 0;
}
