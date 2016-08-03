#include "CameraModel/ConvertDepthMap.hpp"
#include "Util/globalFuncs.hpp"
#include <opencv2/opencv.hpp>

using namespace lsd_slam;



int main(int argc, char **argv) {
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	std::cout << "Loaded camera model from file: \"" << argv[1] << "\"" << std::endl;
	std::cout << *model << std::endl;

	float *testIm = new float[model->w*model->h];
	for (size_t i = 0; i < model->w*model->h; ++i) {
		testIm[i] = 1.f;
	}

	float *convImProj = new float[model->w*model->h];
	float *convImOmni = new float[model->w*model->h];
	depthMapOmniToProj(testIm, convImProj, model.get());
	depthMapProjToOmni(convImProj, convImOmni, model.get());

	cv::Mat inputMat(int(model->h), int(model->w), CV_32FC1, testIm);
	cv::Mat projMat(int(model->h), int(model->w), CV_32FC1, convImProj);
	cv::Mat omniMat(int(model->h), int(model->w), CV_32FC1, convImOmni);
	cv::imshow("INPUT", inputMat / 2.f);
	cv::imshow("PROJ VERSION", projMat / 2.f);
	cv::imshow("OMNI VERSION", omniMat / 2.f);
	cv::waitKey();

	return 0;
}
