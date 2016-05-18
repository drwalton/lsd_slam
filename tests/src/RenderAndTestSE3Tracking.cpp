#include "Tracking/SE3Tracker.hpp"
#include "Tracking/TrackingReference.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.hpp"
#include "DataStructures/Frame.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"
#include "ModelLoader.hpp"
#include "Raycast.hpp"

using namespace lsd_slam;

cv::Mat depth1, depth2;

void depthMouseCallback(int event, int x, int y, int flags, void *userData)
{
	if(event == CV_EVENT_LBUTTONDOWN) {
		float depth = depth1.at<float>(y,x);
		std::cout << "CLICKED DEPTH: " << depth << std::endl;
	}
}

int main(int argc, char **argv)
{
	if(argc < 3 || argv[1] == std::string("-h")) {
		std::cout << "RenderandTestSE3Tracking: render two views of a model using"
			" a supplied camera model, and attempt to recover the transform using"
			" the SE3 tracker.\nUsage: ./RenderAndTestSE3Tracking [camModel]"
			" [3dModel]" << std::endl;
		return 0;
	}

	std::cout << "Loading scene from file: " << argv[2] << std::endl;

	ModelLoader m;
	m.loadFile(resourcesDir() + argv[2]);

	std::cout << "Vertices: \n" << m.vertices();


	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);

	WorldToCamTransform t1;
	t1.translation = vec3(-0.1f, -0.1f, 0.f);
	WorldToCamTransform t2;
	t2.translation = vec3(0.1f, 0.1f, 0.f);
	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			static_cast<uchar>(color.x() * 255.f), 
			static_cast<uchar>(color.y() * 255.f), 
			static_cast<uchar>(color.z() * 255.f)));
	}

	std::cout << "Rendering..." << std::endl;

	cv::Mat image1 = raycast(m.vertices(), m.indices(), colors, t1, *model, true, depth1);
	depth1.setTo(1.f, depth1 == -1);
	cv::Mat image2 = raycast(m.vertices(), m.indices(), colors, t2, *model, true, depth2);
	depth2.setTo(1.f, depth2 == -1);

	cv::imshow("Im1", image1);
	cv::moveWindow("Im1", 0, 30);
	cv::imshow("Im2", image2);
	cv::moveWindow("Im2", image1.cols, 30);
	cv::waitKey(1);

	std::cout << "Real transform applied:\n" << t2 << std::endl;

	cv::Mat fltImage1;
	image1.convertTo(fltImage1, CV_32FC1);
	cv::Mat fltImage2;
	image2.convertTo(fltImage2, CV_32FC1);

	lsd_slam::Frame keyframe(0, *model, 0.0, fltImage1.ptr<float>(0));
	lsd_slam::Frame newFrame(1, *model, 1.0, fltImage2.ptr<float>(0));

//	{
//		lsd_slam::DepthMapPixelHypothesis *arr = new lsd_slam::DepthMapPixelHypothesis[model->w*model->h];
//
//		for (size_t r = 0; r < model->h; ++r) {
//			for (size_t c = 0; c < model->w; ++c) {
//				arr[r*model->w + c].idepth =
//					arr[r*model->w + c].idepth_smoothed = 
//					1.f / depth1.at<float>(r, c);
//				arr[r*model->w + c].idepth_var = 
//					arr[r*model->w + c].idepth_var_smoothed = 0.01f;
//				arr[r*model->w + c].isValid = true;
//			}
//		}
//
//		keyframe.setDepth(arr);
//
//
//		delete[] arr;
//	}

	double min, max;
	cv::minMaxLoc(depth1, &min, &max);
	std::cout << "Depths: Min: " << min << " Max: " << max << std::endl;
	cv::imshow("Depth", depth1 / max);
	cv::setMouseCallback("Depth", depthMouseCallback);
	cv::moveWindow("Depth", 0, 30*2 + image1.rows);
	keyframe.setDepthFromGroundTruth(depth1.ptr<float>(0));
	newFrame.setDepthFromGroundTruth(depth2.ptr<float>(0));

//	lsd_slam::DepthMap depthMap(*model);
//	depthMap.initializeFromGTDepth(&keyframe);

	lsd_slam::TrackingReference reference;
	reference.importFrame(&keyframe);
	keyframe.depthHasBeenUpdatedFlag = false;


	SE3 initialEstimate;
	initialEstimate.translation() += Eigen::Vector3d(0.05, 0., 0.);

	SE3Tracker tracker(*model);

	std::cout << "**** TRACKING FRAME ****" << std::endl;
	SE3 trackedEstimate = tracker.trackFrame(&reference, &newFrame, initialEstimate);
	
	if(tracker.diverged) {
		std::cout << "Tracking Lost!" << std::endl;
		return -1;
	}
	std::cout << "**** FINISHED TRACKING ****" << std::endl;

	std::cout << "Value returned from tracker.trackFrame():\n";
	std::cout << trackedEstimate;
	std::cout << "\nNew frame's transform:\n"
		<< newFrame.pose->thisToParent_raw << std::endl;
	

	WorldToCamTransform estTransform;
	estTransform.rotation = trackedEstimate.rotationMatrix().cast<float>();
	estTransform.translation = trackedEstimate.translation().cast<float>();
	
	std::cout << estTransform << std::endl;
	std::cout << "newFrame.thisToOther_t: " << newFrame.thisToOther_t << std::endl;
	std::cout << "newFrame.otherToThis_t : " << newFrame.otherToThis_t << std::endl;
	std::cout << "newFrame.getScaledCamToWorld().translation() : "
		<< newFrame.getScaledCamToWorld().translation() << std::endl;
	
	cv::Mat image3 = raycast(m.vertices(), m.indices(), colors,
		t1 * estTransform.inverse(), *model);

	cv::imshow("Estimated transform visualisation (should be same as Im2)", image3);
	cv::moveWindow("Estimated transform visualisation (should be same as Im2)",
		image1.cols+image2.cols, 30);

	cv::imshow("Difference from Im2", image3 != image2);
	cv::moveWindow("Difference from Im2",image1.cols, 60 + image1.rows);
	cv::imshow("Difference from Im1", image3 != image1);
	cv::moveWindow("Difference from Im1",image1.cols+image2.cols, 60 + image1.rows);

	cv::waitKey();

	return 0;
}
