#include "Tracking/SE3Tracker.hpp"
#include "Tracking/TrackingReference.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "DataStructures/Frame.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"
#include "ModelLoader.hpp"
#include "Raycast.hpp"

using namespace lsd_slam;

int main(int argc, char **argv)
{
	std::cout << "Loading scene from file: " << "cube.ply" << std::endl;

	ModelLoader m;
	m.loadFile(resourcesDir() + "cube.ply");

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

	cv::Mat image1 = raycast(m.vertices(), m.indices(), colors, t1, *model, cv::Size(model->w, model->h));
	cv::Mat image2 = raycast(m.vertices(), m.indices(), colors, t2, *model, cv::Size(model->w, model->h));

	

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
