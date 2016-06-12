#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <opencv2/opencv.hpp>
#include "Util/globalFuncs.hpp"
#include "Util/ModelLoader.hpp"
#include "Util/Raycast.hpp"

using namespace lsd_slam;

OmniCameraModel *omCamModel;

int main(int argc, char **argv)
{
	if(argc < 2 || argv[1] == std::string("-h")) {
		std::cout << "TestTraceEpipolarLine: test tracing lengths of an epipolar line\n"
			"\nUsage: ./TestEpipolarLine [camModel]" << std::endl;
		return 0;
	}
	
	std::unique_ptr<CameraModel> camModel = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	omCamModel = dynamic_cast<OmniCameraModel*>(camModel.get());
	if(!omCamModel) {
		std::cout << "Loaded camera model not omnidirectional!" << std::endl;
		return -2;
	}

	vec2 pixelLoc1(200, 200);
	vec2 pixelLoc2(250, 250);
	vec3 projPt1 = omCamModel->pixelToCam(pixelLoc1);
	vec3 projPt2 = omCamModel->pixelToCam(pixelLoc2);

	float a = 0.f;
	float increment = 2.f;
	a += omCamModel->getEpipolarParamIncrement(a, projPt1, projPt2, increment);
	vec3 midPt = a*(projPt1)+(1.f - a)*(projPt2);
	vec2 projMidPt = omCamModel->camToPixel(midPt);

	std::cout << "Moving " << increment << " pixels along the line from\n"
		<< pixelLoc1 << "\nto\n" << pixelLoc2 << "\ngives pixel location\n"
		<< projMidPt << "\n" << std::endl;

	projPt1 = 10.f * projPt1;
	projPt2 = 10.f * projPt2;
	a = 0.f;
	a += omCamModel->getEpipolarParamIncrement(a, projPt1, projPt2, increment);
	midPt = a*(projPt1)+(1.f - a)*(projPt2);
	projMidPt = omCamModel->camToPixel(midPt);

	std::cout << "Moving " << increment << " pixels along the line from\n"
		<< pixelLoc1 << "\nto\n" << pixelLoc2 << "\ngives pixel location\n"
		<< projMidPt << "\n" << std::endl;

	projPt1 = 2.f * projPt1;
	a = 0.f;
	a += omCamModel->getEpipolarParamIncrement(a, projPt1, projPt2, increment);
	midPt = a*(projPt1)+(1.f - a)*(projPt2);
	projMidPt = omCamModel->camToPixel(midPt);

	std::cout << "Moving " << increment << " pixels along the line from\n"
		<< pixelLoc1 << "\nto\n" << pixelLoc2 << "\ngives pixel location\n"
		<< projMidPt << "\n" << std::endl;


	a = 0.f;
	a += omCamModel->getEpipolarParamIncrement(a, projPt2, projPt1, increment);
	midPt = a*(projPt2)+(1.f - a)*(projPt1);
	projMidPt = omCamModel->camToPixel(midPt);

	std::cout << "Moving " << increment << " pixels along the line from\n"
		<< pixelLoc2 << "\nto\n" << pixelLoc1 << "\ngives pixel location\n"
		<< projMidPt << "\n" << std::endl;


	projPt1 = 1000.f * projPt1;
	a = 0.f;
	a += omCamModel->getEpipolarParamIncrement(a, projPt1, projPt2, increment);
	midPt = a*(projPt1)+(1.f - a)*(projPt2);
	projMidPt = omCamModel->camToPixel(midPt);

	std::cout << "Moving " << increment << " pixels along the line from\n"
		<< pixelLoc1 << "\nto\n" << pixelLoc2 << "\ngives pixel location\n"
		<< projMidPt << "\n" << std::endl;

	return 0;
}
