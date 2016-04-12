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
	keyframeToReference.translation.y() += 1.f;

	vec3 pointDir;
	std::array<float, 5> vals = findValuesToSearchFor(keyframeToReference,
		model, fltImage.ptr<float>(0), 100, 200, 400, pointDir);

	std::cout << vals;

	return 0;
}
