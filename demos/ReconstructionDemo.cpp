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
    		break;
    	} else if(selection[0] == 'F' || selection[0] == 'f') {
			std::string filename;
			Fl_Native_File_Chooser ch(Fl_Native_File_Chooser::BROWSE_FILE);
			ch.title("Load first image");
			ch.show();
			filename = ch.filename();
			stream.capture().open(pathToForwardSlashes(filename));
			break;
//			if (stream.capture().isOpened()) {
//				break;
//			}
    	}
    	std::cout << "Could not parse response; please try again..." << std::endl;
	}

	stream.setCalibration(argv[1]);
	stream.run();

	lsd_slam::Output3DWrapper outWrapper;
	lsd_slam::LiveSLAMWrapper slamWrapper(&stream, &outWrapper);

	slamWrapper.Loop();
	
	int key = 0;
	cv::Mat frame;
	while(key != 27) {
		cv::imshow("Input", stream.getBuffer()->popFront().data);
		key = cv::waitKey(1);
	}

	stream.stop();
	
	return 0;
}
