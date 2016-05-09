#include <iostream>
#include "Raycast.hpp"
#include "ModelLoader.hpp"
#include "OmniCameraModel.hpp"
#include <fstream>
#include "globalFuncs.hpp"

using namespace lsd_slam;

int main(int argc, char **argv) {
	if(argc < 5) {
		std::cout << "Usage: MakeRaycastedImage [modelFilename] [camTransform] [imageFilename] [camModel]" << std::endl;
		return 1;
	}
	
	std::unique_ptr<CameraModel> model = OmniCameraModel::loadFromFile(
		resourcesDir() + argv[4]);
	
	
	std::cout << "Loading scene from file: " << argv[1] << std::endl;

	ModelLoader m;
	m.loadFile(argv[1]);

	std::cout << "Vertices: \n" << m.vertices();
	
	mat4 worldToCam = loadCamTransform(argv[2]);

	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(
			static_cast<uchar>(color.x() * 255.f), 
			static_cast<uchar>(color.y() * 255.f), 
			static_cast<uchar>(color.z() * 255.f)));
	}
	
	std::cout << "Colors: \n " << colors;

	cv::Mat image = raycast(m.vertices(), m.indices(), colors, worldToCam, 
		*model);
	
	cv::imwrite(argv[3], image);
	
	return 0;
}
