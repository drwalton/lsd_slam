#include <iostream>
#include "Raycast.hpp"
#include "ModelLoader.hpp"
#include "OmniCameraModel.hpp"
#include "CameraMotion.hpp"
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include "util/settings.hpp"
#include "util/ImgProc.hpp"

template<typename T>
std::ostream &operator << (std::ostream &s, std::vector<T> &t) {
	
	s << t[0];
	for (size_t i = 1; i < t.size(); ++i) {
		s << ", " << t[i];
	}
	return s;
}

using namespace lsd_slam;

const int numFrames = 1000;

enum MotionType {
	OSC, ELLIPSE
};
MotionType motionType = ELLIPSE;

int main(int argc, char **argv) {
	if (argc < 7) {
		std::cout << "Usage: MakeRaycastedVideo [camModel] [modelFilename] [camTransform] "
			" [vidFilename] [imRows] [imCols] [saveImages]" << std::endl;
		return 1;
	}
	bool saveImages = argc >= 8;

	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(lsd_slam::resourcesDir() + argv[1]);

	std::string vidFilename(lsd_slam::resourcesDir() + argv[4]);
	std::string imageFolder = vidFilename.substr(0, vidFilename.find_last_of('.'));
	std::string depthFolder = vidFilename.substr(0, vidFilename.find_last_of('.')) + "_depth";
	boost::filesystem::create_directories(boost::filesystem::path(imageFolder));
	boost::filesystem::create_directories(boost::filesystem::path(depthFolder));

	std::cout << "Loading scene from file: " << argv[1] << std::endl;

	ModelLoader m;
	m.loadFile(argv[2]);

	std::cout << "Vertices: \n" << m.vertices();

	mat4 worldToCam = loadCamTransform(lsd_slam::resourcesDir() + argv[3]);
	std::unique_ptr<CameraMotion> camMotion;



	if (motionType == ELLIPSE) {
		camMotion.reset(new EllipticalCameraMotion(
			WorldToCamTransform(worldToCam), vec3(0.5f, 0.f, 0.f), vec3(0.f, 0.25f, 0.f), 100));
	} else {
		camMotion.reset(new OscillatingCameraMotion(
			WorldToCamTransform(worldToCam), vec3(0.5f, 0.f, 0.f), 50));
	}

	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			uchar(color.x() * 255.f), 
			uchar(color.y() * 255.f), 
			uchar(color.z() * 255.f)));
	}
	
	std::cout << "Colors: \n " << colors;

	cv::Size size(atoi(argv[5]), atoi(argv[6]));

	cv::VideoWriter video;
	video.open(lsd_slam::resourcesDir() + argv[3], cv::VideoWriter::fourcc('M','J','P','G'), 30, size);

	cv::Mat image;
	cv::Mat depth;
	int percentage = 0;

	cv::namedWindow("Color");
	cv::namedWindow("Depth");
	cv::moveWindow("Color", 0, 30);
	cv::moveWindow("Depth", model->w, 30);
	
	size_t imgCounter = 0;
	for (size_t i = 0; i < numFrames; ++i) {
		//TODO also record depth, for initialising when testing tracking.
		image = raycast(m.vertices(), m.indices(), colors, camMotion->getNextTransform(), 
			*model, true, depth);
		cv::imshow("Color", image);
		double minD, maxD;
		cv::minMaxLoc(depth, &minD, &maxD);
		cv::imshow("Depth", depth / maxD);
		cv::waitKey(1);
		if (saveImages) {
			std::stringstream ifName;
			ifName << imageFolder << "/" << 
				std::setfill('0') << std::setw(5) << imgCounter << ".png";
			std::string str = ifName.str();
			cv::imwrite(str, image);
			std::stringstream dfName;
			dfName << depthFolder << "/" << 
				std::setfill('0') << std::setw(5) << imgCounter << ".tiff";
			str = dfName.str();
			imwriteFloat(str, depth);
			++imgCounter;
		}

		video << image;
		
		float percentage_f = 100.f * float(i) / float(numFrames);
		if(int(percentage_f) > percentage) {
			percentage = int(percentage_f);
			if(percentage % 10 == 0) {
				std::cout << percentage << "% complete..." << std::endl;
			}
		}
	}
	
	video.release();
	
	std::cout << "Completed!" << std::endl;
	
	return 0;
}
