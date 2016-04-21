#include "Tracking/SE3Tracker.hpp"
#include "Tracking/TrackingReference.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "DataStructures/Frame.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"
#include "util/ImgProc.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.hpp"

using namespace lsd_slam;

int main(int argc, char **argv)
{
	if (argc < 5) return -1;
	cv::Mat image1 = cv::imread(resourcesDir() + argv[1]);
	cv::Mat image2 = cv::imread(resourcesDir() + argv[2]);
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[3]);
	cv::Mat depth1 = imreadFloat(resourcesDir() + argv[4]);

	cv::Mat fltImage1;
	image1.convertTo(fltImage1, CV_32FC1);
	cv::Mat fltImage2;
	image2.convertTo(fltImage2, CV_32FC1);

	lsd_slam::Frame keyframe(0, *model, 0.0, fltImage1.ptr<float>(0));

	{
		lsd_slam::DepthMapPixelHypothesis *arr = new lsd_slam::DepthMapPixelHypothesis[model->w*model->h];

		for (size_t r = 0; r < model->h; ++r) {
			for (size_t c = 0; c < model->w; ++c) {
				arr[r*model->w + c].idepth =
					arr[r*model->w + c].idepth_smoothed = 
					1.f / depth1.at<float>(r, c);
				arr[r*model->w + c].idepth_var = 
					arr[r*model->w + c].idepth_var_smoothed = 0.01f;
				arr[r*model->w + c].isValid = true;
			}
		}

		keyframe.setDepth(arr);


		delete[] arr;
	}

	lsd_slam::DepthMap depthMap(*model);
	depthMap.initializeFromGTDepth(&keyframe);

	lsd_slam::TrackingReference reference;
	reference.importFrame(&keyframe);

	lsd_slam::Frame newFrame(1, *model, 1.0, fltImage2.ptr<float>(0));


	SE3 initialEstimate;

	SE3Tracker tracker(*model);

	SE3 trackedEstimate = tracker.trackFrame(&reference, &newFrame, initialEstimate);

	std::cout << trackedEstimate;

	return 0;
}
