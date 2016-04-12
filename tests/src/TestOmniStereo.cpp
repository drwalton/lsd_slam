#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"

using namespace lsd_slam;

int main(int argc, char **argv)
{
	cv::Mat image = cv::imread(argv[1]);
	cv::Mat fltImage;
	image.convertTo(fltImage, CV_32FC1);

	OmniCameraModel model = OmniCameraModel::makeDefaultModel();

	RigidTransform keyframeToReference;
	keyframeToReference.translation.x() += 1.f;

	std::array<float, 5> vals = findValuesToSearchFor(keyframeToReference,
		model, image.ptr<float>(0), 300, 200, 400);

	std::cout << vals;

	return 0;
}
