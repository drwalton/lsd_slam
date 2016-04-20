#include "Tracking/SE3Tracker.hpp"
#include "Tracking/TrackingReference.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "DataStructures/Frame.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"

using namespace lsd_slam;

int main(int argc, char **argv)
{
	cv::Mat image1 = cv::imread(resourcesDir() + argv[1]);
	cv::Mat image2 = cv::imread(resourcesDir() + argv[2]);
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[3]);

	cv::Mat fltImage1;
	image1.convertTo(fltImage1, CV_32FC1);
	cv::Mat fltImage2;
	image2.convertTo(fltImage2, CV_32FC1);

	lsd_slam::Frame referenceFrame(0, *model, 0.0, fltImage1.ptr<float>(0));
	lsd_slam::DepthMap depthMap(*model);
	depthMap.initializeRandomly(&referenceFrame);

	lsd_slam::TrackingReference reference;
	reference.importFrame(&referenceFrame);

	lsd_slam::Frame newFrame(1, *model, 1.0, fltImage2.ptr<float>(0));


	SE3 initialEstimate;

	SE3Tracker tracker(*model);

	SE3 trackedEstimate = tracker.trackFrame(&reference, &newFrame, initialEstimate);

	std::cout << trackedEstimate;

	return 0;
}
