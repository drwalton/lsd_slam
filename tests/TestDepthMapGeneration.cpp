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

int main(int argc, char **argv)
{
	if(argc < 2 || argv[1] == std::string("-h")) {
		return -1;
	}
	
	std::cout << "Loading scene from file: " << argv[1] << std::endl;

	ModelLoader m;
	m.loadFile(resourcesDir() + argv[1]);

	std::cout << "Vertices: \n" << m.vertices();


	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);

	SE3 transform;
	transform.translation() = Eigen::Vector3d(0.05, 0., 0.);
	WorldToCamTransform t1;
	WorldToCamTransform t2;
	t2.translation = vec3(0.1f, 0.f, 0.f);
	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			static_cast<uchar>(color.x() * 255.f),
			static_cast<uchar>(color.y() * 255.f),
			static_cast<uchar>(color.z() * 255.f)));
	}
	cv::Mat depth1;

	std::cout << "Rendering..." << std::endl;

	cv::Mat image1 = raycast(m.vertices(), m.indices(), colors, t1, *model, true, depth1);
	cv::Mat image2 = raycast(m.vertices(), m.indices(), colors, t2, *model);

	cv::imshow("Im1", image1);
	cv::imshow("Im2", image2);
	cv::waitKey(1);

	std::cout << "Real transform applied:\n" << t2 << std::endl;

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

	std::shared_ptr<lsd_slam::Frame> newFrame(new lsd_slam::Frame(1, *model, 1.0, fltImage2.ptr<float>(0)));

	SE3 initialEstimate;
	initialEstimate.translation() += Eigen::Vector3d(.1, 0., 0.);

	SE3Tracker tracker(*model);

	SE3 trackedEstimate = tracker.trackFrame(&reference, newFrame.get(), initialEstimate);

	std::cout << trackedEstimate;

	std::deque<std::shared_ptr<Frame>> updateFrames;
	updateFrames.push_back(newFrame);

	depthMap.updateKeyframe(updateFrames);

	cv::imshow("DEBUG DEPTH", depthMap.debugImageDepth);
	cv::waitKey();

	return 0;
}