#include <QApplication>
#include "ImageViewer.hpp"
#include "Util/globalFuncs.hpp"

int main(int argc, char **argv)
{
	cv::Mat toShow;
	if(argc >= 2) {
		toShow = cv::imread(lsd_slam::resourcesDir() + argv[1]);
	} else {
		std::cout << "No image supplied" << std::endl;
		return 1;
	}
	QApplication qapp(argc, argv);
	lsd_slam::ImageViewer viewer("Test Image");
	viewer.setImage(toShow);
	qapp.exec();
	return 0;
}