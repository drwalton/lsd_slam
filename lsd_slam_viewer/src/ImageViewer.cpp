#include "ImageViewer.hpp"
#include <QVBoxLayout>
#include <QStatusBar>

namespace lsd_slam
{

ImageViewer::ImageViewer(const std::string &title)
	:newImage_(false)
{
	imageLabel.setParent(this);
	imageLabel.setBackgroundRole(QPalette::Base);
	imageLabel.setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	imageLabel.setScaledContents(true);
	imageLabel.setText("Displayed Image");
	setCentralWidget(&imageLabel);
	setWindowTitle(QString(title.c_str()));
	
	cv::Mat matBgr8(400, 400, CV_8UC3);
	matBgr8.setTo(cv::Scalar(0,0,0));
	
	QImage image(matBgr8.ptr<uchar>(0),
		matBgr8.cols, matBgr8.rows, QImage::Format_RGB888);
	setFixedSize(image.width(), image.height());
	imageLabel.resize(image.width(), image.height());
	pixmap = QPixmap::fromImage(image);
	imageLabel.setPixmap(pixmap);
	imageLabel.repaint();
	imageLabel.update();
	timer_.setInterval(30);
	connect(&timer_, SIGNAL(timeout()), this, SLOT(checkForNewImage()));
	timer_.start();
	show();
}

ImageViewer::~ImageViewer() throw()
{}

void ImageViewer::setImage(const cv::Mat &mat)
{
	mutex_.lock();
	mat.copyTo(image_);
	newImage_ = true;
	mutex_.unlock();
}

void ImageViewer::checkForNewImage()
{
	mutex_.lock();
	if(newImage_) {
		newImage_ = false;
    	cv::Mat matBgr;
    	if(image_.channels() == 1) {
    		cv::cvtColor(image_, matBgr, CV_GRAY2BGR);
    	} else {
    		cv::cvtColor(image_, matBgr, CV_BGR2RGB);
    	}
    	cv::Mat matBgr8;
    	if(matBgr.type() != CV_8UC3) {
    		matBgr.convertTo(matBgr8, CV_8UC3);
    	} else {
    		matBgr8 = matBgr;
    	}
		
    	QImage image(matBgr8.ptr<uchar>(0),
    		matBgr8.cols, matBgr8.rows, QImage::Format_RGB888);
    	pixmap = QPixmap::fromImage(image);
    	
    	imageLabel.setPixmap(pixmap);
    	
    	setFixedSize(image.width(), image.height());
    	imageLabel.resize(image.width(), image.height());
		
		update();
	}
	mutex_.unlock();
}

}
