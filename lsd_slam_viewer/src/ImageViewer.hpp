#pragma once

#include <opencv2/opencv.hpp>
#include <QMainWindow>
#include <QLabel>
#include <mutex>
#include <QTimer>

namespace lsd_slam
{

///\brief Image viewer intended for use in multi-threaded programs, with a
/// QApplication running in the main thread.
///\note This viewer can be supplied a new image to display at any time, but only
/// checks for new images and displays them at set intervals.
class ImageViewer : public QMainWindow
{
	Q_OBJECT
public:
	///\brief Construct the imageviewer, and set the window title.
	///\note Also calls show().
	explicit ImageViewer(const std::string &title);
	~ImageViewer() throw();
	
	///\brief Set the image to be shown in the viewer.
	///\note Can accept input of any 1 or 3-channel type. Note that images
	/// will be converted to RGB888 before being displayed.
	void setImage(const cv::Mat &im);
private slots:
	///\brief Runs at intervals, on main thread. Updates image if a new one is
	/// available.
	void checkForNewImage();
	
private:
	///Storage for the new image to be displayed in the viewer.
	cv::Mat image_;
	
	///Set to true if a new image is available.
	bool newImage_;
	
	QPixmap pixmap;
	QLabel imageLabel;
	QTimer timer_;
	std::mutex mutex_;
};
	
}
