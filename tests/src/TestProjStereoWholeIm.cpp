#include "DepthEstimation/DepthMapLineStereo.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"
#include "ModelLoader.hpp"
#include "Raycast.hpp"

using namespace lsd_slam;

cv::Mat fltIm1, fltIm2, depth1;
cv::Mat showIm1, showIm2;
RigidTransform keyframeToReference;
ProjCameraModel *pjCamModel;
vec3 pointDir;
bool addNoise = true;

const float DEPTH_SEARCH_RANGE = 0.4f;
void showPossibleColors();


int main(int argc, char **argv)
{
	if(argc < 3 || argv[1] == std::string("-h")) {
		std::cout << "TestProjStereoWholeIm: render two views of a model using"
			" a supplied camera model, and perform projective stereo."
			"\nUsage: ./TestProjStereoWholeIm [camModel]"
			" [3dModel]" << std::endl;
		return 0;
	}
	
	std::unique_ptr<CameraModel> camModel = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	pjCamModel = dynamic_cast<ProjCameraModel*>(camModel.get());
	if(!pjCamModel) {
		std::cout << "Loaded camera model not projective!" << std::endl;
		return -2;
	}

	std::cout << "Loading scene from file: " << argv[2] << std::endl;

	ModelLoader m;
	m.loadFile(resourcesDir() + argv[2]);

	std::cout << "Vertices: \n" << m.vertices();

	SE3 transform;
	WorldToCamTransform t1;
	WorldToCamTransform t2;
	t2.translation = vec3(-0.15f, 0.f, 0.f);
	transform.translation() = t2.translation.cast<double>();
	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			static_cast<uchar>(color.x() * 255.f), 
			static_cast<uchar>(color.y() * 255.f), 
			static_cast<uchar>(color.z() * 255.f)));
	}

	std::cout << "Rendering Image 1..." << std::endl;
	cv::Mat image1 = raycast(m.vertices(), m.indices(), colors, t1, *camModel, true, depth1);
	std::cout << "Rendering Image 2..." << std::endl;
	cv::Mat image2 = raycast(m.vertices(), m.indices(), colors, t2, *camModel);
	std::cout << "Rendering complete!" << std::endl;

	cv::Mat im1gray, im2gray;
	cv::cvtColor(image1, im1gray, CV_RGB2GRAY);
	cv::cvtColor(image2, im2gray, CV_RGB2GRAY);
	im1gray.convertTo(fltIm1, CV_32FC1);
	im2gray.convertTo(fltIm2, CV_32FC1);
	if (addNoise) {
		float noiseMean = 10.f;
		cv::Mat randIm(fltIm1.size(), CV_32FC1);
		cv::randn(randIm, 0.f, noiseMean);
		fltIm1 += randIm;
		cv::randn(randIm, 0.f, noiseMean);
		fltIm2 += randIm;
	}

	keyframeToReference.translation = transform.translation().cast<float>();
	keyframeToReference.rotation = transform.rotationMatrix().cast<float>();

	cv::imshow("Im1", im1gray);
	cv::imshow("Im2", im2gray);
	cv::moveWindow("Im1", 0, 30);
	cv::moveWindow("Im2", fltIm1.cols, 30);
	
	cv::waitKey(1);

	if(!fltIm1.isContinuous() || !fltIm2.isContinuous()) {
		throw std::runtime_error("Need continuous mats!");
	}

	cv::cvtColor(fltIm1, showIm1, CV_GRAY2BGR);
	showIm1.convertTo(showIm1, CV_8UC3);
	RunningStats stats;
	float r_idepth, r_var, r_eplLength;
	float r_gradAlongLine, r_lineLen;

	try{
		for (size_t r = 0; r < fltIm1.rows; ++r) {
			for (size_t c = 0; c < fltIm1.cols; ++c) {
				vec3 a;
				//findValuesToSearchFor(keyframeToReference, *omCamModel, fltIm1.ptr<float>(0), x, y, fltIm1.cols, a, showIm1);
				vec3 dir = pjCamModel->pixelToCam(vec2(c, r));

				float depth = depth1.at<float>(r, c);

				vec3 matchDir; vec2 epDir;

				float epxn, epyn;
				makeAndCheckEPL(c, r, fltIm2.ptr<float>(0), fltIm1.ptr<float>(0),
					keyframeToReference,
					&epxn, &epyn, &stats, *pjCamModel);

				//TODO
				std::vector<Eigen::Vector4f> gradients;
				float initialTrackedResidual;

				float estDepth = doLineStereo(
					c, r, epxn, epyn,
					1.f / (depth * DEPTH_SEARCH_RANGE),
					1.f / (depth),
					1.f / (depth * (2.f - DEPTH_SEARCH_RANGE)),
					fltIm1.ptr<float>(0), fltIm2.ptr<float>(0),
					*pjCamModel, keyframeToReference,
					r_idepth, r_var, r_lineLen,
					gradients.data(),
					initialTrackedResidual,
					&stats);
				if (estDepth > 0) {
					vec3 color = 255.f * hueToRgb(estDepth / 2.f);
					showIm1.at<cv::Vec3b>(r, c) = cv::Vec3b(color.z(), color.y(), color.x());
				}
			}
		}
	}
	catch (cv::Exception &e) {
		std::cout << e.what() << std::endl;
		std::cin.get();
		return -1;
	}

	cv::imshow("MATCHES & DEPTHS", showIm1);

	int key = 0;
	while(key != 27) {
		cv::waitKey(100);
	}

	return 0;
}
