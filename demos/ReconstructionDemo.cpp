#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
	cv::VideoCapture cap;
	
	while(true) {
    	std::cout << "Please select video source ([C]amera, [F]ile): ";
    	std::string selection;
    	std::cin >> selection;
    	if(selection[0] == 'C' || selection[0] == 'c') {
    		int i;
    		std::cout << "Please enter number of camera (0=default): ";
    		std::cin >> i;
    		cap.open(i);
    		break;
    	} else if(selection[0] == 'F' || selection[0] == 'f') {
			std::string filename;
    		std::cout << "Please enter filename: ";
			std::cin >> filename;
			cap.open(filename);
			break;
    	}
    	std::cout << "Could not parse response; please try again..." << std::endl;
	}
	
	int key = 0;
	cv::Mat frame;
	while(key != 27) {
		cap >> frame;
		cv::imshow("Input", frame);
		key = cv::waitKey(30);
	}
	
	return 0;
}
