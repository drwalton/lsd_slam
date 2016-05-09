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

const float DEPTH_SEARCH_RANGE = 0.1f;

void im1MouseCallback(int event, int x, int y, int flags, void *userData) {
	if(event == CV_EVENT_LBUTTONDOWN) {
		std::array<float, 5> vals = findValuesToSearchFor(keyframeToReference,
			*omCamModel, fltIm1.ptr<float>(0), x, y, fltIm1.cols, pointDir);
		std::cout << "Clicked (" << x << "," << y << "), dir:\n" <<
			pointDir << std::endl;
		std::cout << "Vals to search for: " << vals << std::endl;
		fltIm1.convertTo(showIm1, CV_32FC3);
		cv::circle(showIm1, cv::Point(x, y), 3, cv::Scalar(0,0,255));
		cv::imshow("Im1", showIm1);
		cv::waitKey(1);
		
		float depth = depth1.at<float>(y, x);
		float minSsd;
		
		vec3 matchDir; vec2 matchPixel;
		bool ok = omniStereo(keyframeToReference, *omCamModel,
			fltIm1.ptr<float>(0), fltIm2.ptr<float>(0),
			fltIm1.cols,
			x, y,
			depth - DEPTH_SEARCH_RANGE, depth + DEPTH_SEARCH_RANGE,
			minSsd,
			matchDir, matchPixel);
		if(!ok) {
			std::cout << "Stereo match failed!" << std::endl;
		} else {
			std::cout << "Match found: " << matchPixel << std::endl;
			fltIm2.convertTo(showIm2, CV_32FC3);
			cv::circle(showIm2, cv::Point(matchPixel.x(), matchPixel.y()), 3, cv::Scalar(0,255,0));
			cv::imshow("Im2", showIm2);
			cv::waitKey(1);
		}
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
	t2.translation = vec3(0.1f, 0.f, 0.f);
	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			static_cast<uchar>(color.x() * 255.f), 
			static_cast<uchar>(color.y() * 255.f), 
			static_cast<uchar>(color.z() * 255.f)));
	}
	cv::Mat depth1;

	std::cout << "Rendering Image 1..." << std::endl;
	cv::Mat image1 = raycast(m.vertices(), m.indices(), colors, t1, *camModel, true, depth1);
	std::cout << "Rendering Image 2..." << std::endl;
	cv::Mat image2 = raycast(m.vertices(), m.indices(), colors, t2, *camModel);
	std::cout << "Rendering complete!" << std::endl;

	
	image1.convertTo(fltIm1, CV_32FC1);
	image2.convertTo(fltIm2, CV_32FC1);

	keyframeToReference.translation = transform.translation().cast<float>();
	keyframeToReference.rotation = transform.rotationMatrix().cast<float>();

	cv::imshow("Im1", fltIm1);
	cv::imshow("Im2", fltIm2);
	cv::moveWindow("Im1", 0, 30);
	cv::moveWindow("Im2", fltIm1.cols, 30);
	
	cv::waitKey(1);
	cv::setMouseCallback("Im1", im1MouseCallback);

	int key = 0;
	while(key != 27) {
		cv::waitKey(30);
	}

	return 0;
}
