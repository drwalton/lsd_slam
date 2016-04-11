#include <iostream>
#include "Raycast.hpp"
#include "ModelLoader.hpp"
#include "OmniCameraModel.hpp"
#include "CameraMotion.hpp"
#include <fstream>

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
	if (argc != 6) {
		std::cout << "Usage: MakeRaycastedVideo [modelFilename] [camTransform] [vidFilename] [imRows] [imCols]" << std::endl;
		return 1;
	}

	OmniCameraModel model = OmniCameraModel::makeDefaultModel();


	std::cout << "Loading scene from file: " << argv[1] << std::endl;

	ModelLoader m;
	m.loadFile(argv[1]);

	std::cout << "Vertices: \n" << m.vertices();

	mat4 worldToCam = loadCamTransform(argv[2]);
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
		colors.push_back(cv::Vec3b(color.x() * 255.f, color.y() * 255.f, color.z() * 255.f));
	}
	
	std::cout << "Colors: \n " << colors;

	cv::Size size(atoi(argv[4]), atoi(argv[5]));

	cv::VideoWriter video;
	video.open(argv[3], cv::VideoWriter::fourcc('M','J','P','G'), 30, size);

	cv::Mat image;
	int percentage = 0;
	
	for (size_t i = 0; i < numFrames; ++i) {
		image = raycast(m.vertices(), m.indices(), colors, camMotion->getNextTransform(), 
			model, size);
		cv::imshow("PREVIEW", image);
		cv::waitKey(1);
		video << image;
		
		float percentage_f = 100.f * float(i) / float(numFrames);
		if(int(percentage_f) > percentage) {
			percentage = int(percentage_f);
			if(percentage % 10 == 0) {
				std::cout << percentage << "\% complete..." << std::endl;
			}
		}
	}
	
	video.release();
	
	std::cout << "Completed!" << std::endl;
	
	return 0;
}
