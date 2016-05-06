#include "DepthEstimation/DepthMap.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"

using namespace lsd_slam;

int main(int argc, char **argv)
{
	std::unique_ptr<CameraModel> cameraModel = CameraModel::loadFromFile(argv[1]);
	DepthMap depthMap(*cameraModel);

	cv::Mat im1 = cv::imread(argv[2]);
	cv::Mat im2 = cv::imread(argv[3]);

	RigidTransform displacement;
	displacement.translation.z() += 1.f;

	//TODO depthMap.updateKeyframe();

	cv::imshow("Debug DEPTH", depthMap.debugImageDepth);

	return 0;
}
