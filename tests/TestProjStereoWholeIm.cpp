#include "DepthEstimation/DepthMapLineStereo.hpp"
#include <opencv2/opencv.hpp>
#include "Util/globalFuncs.hpp"
#include "Util/ModelLoader.hpp"
#include "Util/Raycast.hpp"
#include "DataStructures/Frame.hpp"

using namespace lsd_slam;

cv::Mat fltIm1, fltIm2, depth1;
cv::Mat showIm1, showIm2, estDepths;
RigidTransform keyframeToReference;
ProjCameraModel *pjCamModel;
vec3 pointDir;
bool addNoise = true;

const float DEPTH_SEARCH_RANGE = 0.9f;
void showPossibleColors();
cv::Mat visualizeDepthError(const cv::Mat &estDepth, const cv::Mat &realDepth,
	float maxDepthError);

void matchesDepthsMouseCallback(int event, int x, int y, int flags, void *userData)
{
	if (event == CV_EVENT_LBUTTONDOWN) {
		float estDepth = estDepths.at<float>(y, x);
		float realDepth = depth1.at<float>(y, x);
		std::cout << "Est depth: " << estDepth << ", Real depth: " << realDepth
			<< std::endl;
	}
}
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
		//fltIm1 += randIm;
		cv::randn(randIm, 0.f, noiseMean);
		//fltIm2 += randIm;
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
	float r_idepth, r_var;
	float r_lineLen;
	estDepths = cv::Mat(fltIm1.size(), CV_32FC1);
	estDepths.setTo(-1.f);

	ModelLoader modelLoader;
	ModelLoader gtDepthLoader;

	std::cout << "Computing Stereo..." << std::endl;
	try{
		processImageWithProgressBar(fltIm1.size(), [&](int r, int c) {
			vec3 a;
			//findValuesToSearchFor(keyframeToReference, *omCamModel, 
			//	fltIm1.ptr<float>(0), x, y, fltIm1.cols, a, showIm1);
			float depth = depth1.at<float>(r, c);

			vec3 matchDir; vec2 epDir;

			float epxn, epyn;
			makeAndCheckEPL(c, r, fltIm2.ptr<float>(0), fltIm1.ptr<float>(0),
				keyframeToReference.inverse(),
				&epxn, &epyn, &stats, *pjCamModel);

			std::vector<Eigen::Vector4f> gradients(fltIm1.cols*fltIm1.rows);
			calculateImageGradients(fltIm1.ptr<float>(0), gradients.data(),
				fltIm1.cols, fltIm1.rows);
			float initialTrackedResidual = 1.f;

			float err = doLineStereo(
				float(c), float(r), epxn, epyn,
				1.f / (depth * (2.f - DEPTH_SEARCH_RANGE)),
				1.f / (depth),
				1.f / (depth * DEPTH_SEARCH_RANGE),
				fltIm1.ptr<float>(0), fltIm2.ptr<float>(0),
				*pjCamModel, keyframeToReference,
				r_idepth, r_var, r_lineLen,
				gradients.data(),
				initialTrackedResidual,
				&stats);
			if (err >= 0) {
				float depth = 1.f / r_idepth;
				estDepths.at<float>(r, c) = depth;
				vec3 color = 255.f * hueToRgb(depth / 2.f);
				showIm1.at<cv::Vec3b>(r, c) = cv::Vec3b(
					uchar(color.z()), uchar(color.y()), uchar(color.x()));

				modelLoader.vertices().push_back(camModel->pixelToCam(vec2(c, r), depth));
				modelLoader.vertColors().push_back(
					255.f*hueToRgb(0.8f*(r_var) /0.001f));

				gtDepthLoader.vertices().push_back(camModel->pixelToCam(vec2(r, c)) * depth1.at<float>(r, c));
			}
		});
	}
	catch (cv::Exception &e) {
		std::cout << e.what() << std::endl;
		std::cin.get();
		return -1;
	}

	modelLoader.saveFile(resourcesDir() + "testStereoCloudProj.ply");
	gtDepthLoader.saveFile(resourcesDir() + "testStereoCloudProjGt.ply");
	
	std::cout << "Stereo Done!" << std::endl;

	cv::imshow("MATCHES & DEPTHS", showIm1);
	cv::setMouseCallback("MATCHES & DEPTHS", matchesDepthsMouseCallback);

	cv::Mat depthErrorIm = visualizeDepthError(estDepths, depth1, 2.f);
	cv::imshow("Depth Error", depthErrorIm);
	cv::moveWindow("Depth Error", 60 + fltIm1.rows, showIm1.cols);


	cv::Mat depthPlusMinus(depth1.size(), CV_8UC3);
	depthPlusMinus.setTo(cv::Vec3b(255, 0, 0), estDepths < depth1);
	depthPlusMinus.setTo(cv::Vec3b(0, 0, 255), depth1 < estDepths);
	cv::imshow("+-", depthPlusMinus);

	int key = 0;
	while(key != 27) {
		cv::waitKey(100);
	}

	return 0;
}

cv::Mat visualizeDepthError(const cv::Mat &estDepth, const cv::Mat &realDepth, 
	float maxDepthError)
{
	if (estDepth.size() != realDepth.size()) {
		throw std::runtime_error("DIFFERENT SIZES IN visualizeDepthError()!");
	}
	cv::Mat visIm(estDepth.size(), CV_8UC3);
	visIm.setTo(cv::Scalar(0, 0, 0));
	for (size_t r = 0; r < size_t(visIm.rows); ++r) {
		for (size_t c = 0; c < size_t(visIm.cols); ++c) {
			if (estDepth.at<float>(r, c) <= 0.f) continue;
			float err = fabsf(realDepth.at<float>(r, c) - estDepth.at<float>(r, c));
			vec3 color = 255.f * hueToRgb(0.8f*(err) / maxDepthError);
			visIm.at<cv::Vec3b>(r, c) = vec3ToColor(color);
		}
	}

	return visIm;
}

void showPossibleColors()
{
	size_t h = 30;
	size_t w = 400;

	cv::Mat colors(cv::Size(w, h), CV_8UC3);

	for (size_t i = 0; i < w; ++i) {
		vec3 color = 255.f * hueToRgb(0.8f * float(i) / float(w));
		for (size_t j = 0; j < h; ++j) {
			colors.at<cv::Vec3b>(j, i) = cv::Vec3b(
				uchar(color.z()), uchar(color.y()), uchar(color.x()));
		}
	}

	cv::imshow("Err Scale (Min-Max)", colors);
	cv::moveWindow("Err Scale (Min-Max)", 0, 800);
}
