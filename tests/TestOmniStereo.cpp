#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <opencv2/opencv.hpp>
#include "Util/globalFuncs.hpp"
#include "Util/ModelLoader.hpp"
#include "Util/Raycast.hpp"

using namespace lsd_slam;

cv::Mat fltIm1, fltIm2, depth1;
cv::Mat showIm1, showIm2;
RigidTransform keyframeToReference;
OmniCameraModel *omCamModel;
vec3 pointDir;
bool addNoise = false;
int lastClickedX = -1, lastClickedY = -1;

const float DEPTH_SEARCH_RANGE = 0.4f;
void showPossibleColors();

void im1MouseCallback(int event, int x, int y, int flags, void *userData) {
	if(event == CV_EVENT_LBUTTONDOWN) {
		lastClickedX = x, lastClickedY = y;
		cv::cvtColor(fltIm1, showIm1, CV_GRAY2BGR);
		showIm1.convertTo(showIm1, CV_8UC3);
		vec3 a;
		//findValuesToSearchFor(keyframeToReference, *omCamModel, fltIm1.ptr<float>(0), x, y, fltIm1.cols, a, showIm1);
		vec3 dir = omCamModel->pixelToCam(vec2(x, y));
		std::cout << "Clicked (" << x << "," << y << "), dir:\n" <<
			dir << std::endl;
		std::array<float, 5> valsToFind;
		getValuesToFindOmni(dir, keyframeToReference.translation, 
			fltIm1.ptr<float>(0), fltIm1.cols, *omCamModel, float(x), float(y),
			valsToFind, showIm1);
		//cv::circle(showIm1, cv::Point(x, y), 3, cv::Scalar(255, 0, 0));
		cv::imshow("Im1", showIm1);
		
		float depth = depth1.at<float>(y, x);
		std::cout << "Clicked depth: " << depth << std::endl;
		
		vec3 matchDir; vec2 epDir;
		cv::cvtColor(fltIm2, showIm2, CV_GRAY2BGR);
		showIm2.convertTo(showIm2, CV_8UC3);
		
		float r_idepth;
		float r_gradAlongLine, r_lineLen;
		RunningStats s;
		if(!fltIm1.isContinuous() || !fltIm2.isContinuous()) {
			throw std::runtime_error("Need continuous mats!");
		}

		std::cout << "Depth range: " << depth * DEPTH_SEARCH_RANGE
			<< ", " << depth << ", " << depth * (2.f - DEPTH_SEARCH_RANGE)
			<< std::endl;

		vec3 keyframeMatchPt;
		float err = doOmniStereo(
			float(x), float(y), -keyframeToReference.translation,
			.4f / depth,
			1.f / depth,
			1.4f / depth,
			fltIm1.ptr<float>(0), fltIm2.ptr<float>(0),
			keyframeToReference,
			&s, *omCamModel, fltIm1.cols,
			epDir, matchDir, 
			r_gradAlongLine, r_lineLen, keyframeMatchPt,
			showIm2, true, 1);
		
		std::cout << "Grad Along Line: " << r_gradAlongLine << std::endl;
		

		if(err < 0.f) {
			std::cout << "Stereo match failed! Code: " << err << std::endl;
			switch (int(err)) {
				case -3:
					std::cout << "Error too large!" << std::endl;
					break;
				case -2:
					std::cout << "Winner not clear enough!" << std::endl;
					break;
				case -1:
					std::cout << "One of lines went outside image!" << std::endl;
					break;
			}
		} else {
//			r_idepth = findInvDepthOmni(float(x), float(y), matchDir, omCamModel, keyframeToReference.inverse(),
//				&s);
			r_idepth = 1.f / keyframeMatchPt.norm();
			vec2 matchPixel = omCamModel->camToPixel(matchDir);
			std::cout << "Match found: " << matchPixel << std::endl;
			cv::circle(showIm2, vec2Point(matchPixel), 3, cv::Scalar(0,255,0));
			std::cout << "\t* Estimated inverse depth: " << r_idepth <<
				"\n\t* Estimated depth: " << 1.f / r_idepth << 
				//"\n\t* Estimated variance: " << r_var <<
				"\n\t* Grad along line: " << r_gradAlongLine <<
				"\n\t* Line Length: " << r_lineLen << std::endl;
		}
		cv::imshow("Im2", showIm2);
		cv::waitKey(1);
	}
}

void im2MouseCallback(int event, int x, int y, int flags, void *userData) {
	if (event == CV_EVENT_LBUTTONDOWN) {
		if (lastClickedX >= 0 && lastClickedY >= 0) {
			vec3 im1ClickedDir = omCamModel->pixelToCam(vec2(lastClickedX, lastClickedY));
			vec3 im2ClickedDir = omCamModel->pixelToCam(vec2(x, y));
			im2ClickedDir = keyframeToReference.rotation.inverse() * im2ClickedDir;
			Ray im1Ray, im2Ray;
			im1Ray.origin = vec3::Zero(); im1Ray.dir = im1ClickedDir;
			im2Ray.origin = -keyframeToReference.translation; im2Ray.dir = im2ClickedDir;

			RayIntersectionResult result = computeRayIntersection(im1Ray, im2Ray);

			std::cout << "Clicked points intersection: " << result <<
				"\nDist: " << result.position.norm() << std::endl;
		}
	}
}

int main(int argc, char **argv)
{
	if(argc < 3 || argv[1] == std::string("-h")) {
		std::cout << "TestOmniStereo: render two views of a model using"
			" a supplied camera model, and perform omnidirectional stereo."
			"\nUsage: ./TestOmniStereo [camModel]"
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
	WorldToCamTransform t1;
	WorldToCamTransform t2;
	t2.translation = vec3(-0.15f, 0.f, 0.f);
	t2.rotation = Eigen::AngleAxisf(0.1f, vec3(1.f, 1.f, 0.f)).toRotationMatrix();
	transform.translation() = t2.translation.cast<double>();
	transform.setRotationMatrix(t2.rotation.cast<double>());
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

	showPossibleColors();

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
	cv::namedWindow("TO FIND");
	cv::moveWindow("TO FIND", 0, 60+fltIm1.rows);
	cv::namedWindow("SEARCHED");
	cv::moveWindow("SEARCHED", fltIm1.cols, 60+fltIm1.rows);
	cv::namedWindow("ERRS");
	cv::moveWindow("ERRS", fltIm1.cols, 100+fltIm1.rows);
	
	cv::waitKey(1);
	cv::setMouseCallback("Im1", im1MouseCallback);
	cv::setMouseCallback("Im2", im2MouseCallback);

	int key = 0;
	while(key != 27) {
		cv::waitKey(100);
	}

	return 0;
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