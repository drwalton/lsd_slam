#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <opencv2/opencv.hpp>
#include "globalFuncs.hpp"
#include "ModelLoader.hpp"
#include "Raycast.hpp"

using namespace lsd_slam;

cv::Mat fltIm1, fltIm2, depth1;
cv::Mat showIm1, showIm2;
RigidTransform keyframeToReference;
OmniCameraModel *omCamModel;
vec3 pointDir;

const float DEPTH_SEARCH_RANGE = 0.2f;

void im1MouseCallback(int event, int x, int y, int flags, void *userData) {
	if(event == CV_EVENT_LBUTTONDOWN) {
		std::cout << "Clicked (" << x << "," << y << "), dir:\n" <<
			pointDir << std::endl;
		cv::cvtColor(fltIm1, showIm1, CV_GRAY2BGR);
		showIm1.convertTo(showIm1, CV_8UC3);
		vec3 a;
		findValuesToSearchFor(keyframeToReference, *omCamModel, fltIm1.ptr<float>(0), x, y, fltIm1.cols, a, showIm1);
		//cv::circle(showIm1, cv::Point(x, y), 3, cv::Scalar(255, 0, 0));
		cv::imshow("Im1", showIm1);
		
		float depth = depth1.at<float>(y, x);
		std::cout << "Clicked depth: " << depth << std::endl;
		float minSsd;
		
		vec3 matchDir; vec2 matchPixel;
		cv::cvtColor(fltIm2, showIm2, CV_GRAY2BGR);
		showIm2.convertTo(showIm2, CV_8UC3);
		
		float r_idepth, r_var, r_eplLength;
		float r_gradAlongLine, r_lineLen;
		RunningStats s;
		float err = doOmniStereo(
			x, y, matchDir,
			1.f / (depth * DEPTH_SEARCH_RANGE), 
			1.f / (depth),
			1.f / (depth * (2.f-DEPTH_SEARCH_RANGE)),
			fltIm1.ptr<float>(0), fltIm2.ptr<float>(0),
			keyframeToReference, r_idepth, r_var, r_eplLength,
			&s, *omCamModel, fltIm1.cols,
			matchPixel, matchDir, 
			r_gradAlongLine, r_lineLen,
			showIm2);
		if(err < 0.f) {
			std::cout << "Stereo match failed! Code: " << err << std::endl;
		} else {
			std::cout << "Match found: " << matchPixel << std::endl;
			matchPixel = omCamModel->camToPixel(matchDir);
			cv::circle(showIm2, cv::Point(matchPixel.x(), matchPixel.y()), 3, cv::Scalar(0,255,0));
		}
		cv::imshow("Im2", showIm2);
		cv::waitKey(1);
	}
}

int main(int argc, char **argv)
{
	if(argc < 3 || argv[1] == std::string("-h")) {
		std::cout << "TestOmniStereo: render two views of a model using"
			" a supplied camera model, and perform omnidirectional stereo."
			"\nUsage: ./RenderAndTestSE3Tracking [camModel]"
			" [3dModel]" << std::endl;
		return 0;
	}
	
	std::unique_ptr<CameraModel> camModel = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	omCamModel = dynamic_cast<OmniCameraModel*>(camModel.get());
	if(!omCamModel) {
		std::cout << "Loaded camera model not omnidirectional!" << std::endl;
		return -2;
	}

	std::cout << "Loading scene from file: " << argv[2] << std::endl;

	ModelLoader m;
	m.loadFile(resourcesDir() + argv[2]);

	std::cout << "Vertices: \n" << m.vertices();

	SE3 transform;
	transform.translation() = Eigen::Vector3d(0.05, 0., 0.);
	WorldToCamTransform t1;
	WorldToCamTransform t2;
	t2.translation = vec3(0.15f, 0.f, 0.f);
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

	keyframeToReference.translation = transform.translation().cast<float>();
	keyframeToReference.rotation = transform.rotationMatrix().cast<float>();

	cv::imshow("Im1", im1gray);
	cv::imshow("Im2", im2gray);
	cv::moveWindow("Im1", 0, 30);
	cv::moveWindow("Im2", fltIm1.cols, 30);
	
	cv::waitKey(1);
	cv::setMouseCallback("Im1", im1MouseCallback);

	int key = 0;
	while(key != 27) {
		cv::waitKey(100);
	}

	return 0;
}
