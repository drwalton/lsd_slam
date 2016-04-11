#include <iostream>
#include "Raycast.hpp"
#include "ModelLoader.hpp"
#include "OmniCameraModel.hpp"
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

int main(int argc, char **argv) {
	if(argc != 6) {
		std::cout << "Usage: MakeRaycastedImage [modelFilename] [camTransform] [imageFilename] [imRows] [imCols]" << std::endl;
		return 1;
	}
	
	OmniCameraModel model = OmniCameraModel::makeDefaultModel();
	
	
	std::cout << "Loading scene from file: " << argv[1] << std::endl;

	ModelLoader m;
	m.loadFile(argv[1]);

	std::cout << "Vertices: \n" << m.vertices();
	
	mat4 worldToCam = loadCamTransform(argv[2]);

	std::vector<cv::Vec3b> colors;
	for (auto & color : m.vertColors()) {
		colors.push_back(cv::Vec3b(color.x() * 255.f, color.y() * 255.f, color.z() * 255.f));
	}
	
	std::cout << "Colors: \n " << colors;

	cv::Mat image = raycast(m.vertices(), m.indices(), colors, worldToCam, 
		model, cv::Size(atoi(argv[4]), atoi(argv[5])));
	
	cv::imwrite(argv[3], image);
	
	return 0;
}
