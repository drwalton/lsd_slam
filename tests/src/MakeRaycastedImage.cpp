#include <iostream>
#include "Raycast.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "OmniCameraModel.hpp"
#include <fstream>

using namespace lsd_slam;

int main(int argc, char **argv) {
	if(argc != 6) {
		std::cout << "Usage: MakeRaycastedImage [modelFilename] [camTransform] [imageFilename] [imRows] [imCols]" << std::endl;
		return 1;
	}
	
	OmniCameraModel model = OmniCameraModel::makeDefaultModel();
	
	
	std::cout << "Loading scene from file: " << argv[1] << std::endl;
	
	std::ifstream file(argv[1]);
	
	size_t nVerts, nIndices;
	file >> nVerts;
	file >> nIndices;
	
	std::vector<vec3> vertices;
	std::vector<cv::Vec3b> colors;
	std::vector<unsigned int> indices;
	
	
	for(size_t i = 0; i < nVerts; ++i) {
		vec3 v;
		file >> v.x() >> v.y() >> v.z();
		vertices.push_back(v);
	}
	for(size_t i = 0; i < nVerts; ++i) {
		int a, b, c;
		file >> a >> b >> c;
		colors.push_back(cv::Vec3b(a, b, c));
	}
	for(size_t i = 0; i < nIndices; ++i) {
		size_t s;
		file >> s;
		indices.push_back(s);
	}
	
	std::cout << "Loaded " << vertices.size() << " vertices, " << indices.size() << " indices, " << colors.size() << " colors" << std::endl;
	std::cout << "verts";
	for(auto &vert : vertices) std::cout << vert << std::endl;
	std::cout << "Indices";
	for(auto &vert : indices) std::cout << vert << std::endl;
	std::cout << "Colors";
	for(auto &vert : colors) std::cout << vert << std::endl;
	
	
	mat4 worldToCam = loadCamTransform(argv[2]);
	
	cv::Mat image = raycast(vertices, indices, colors, worldToCam, model, cv::Size(atoi(argv[4]), atoi(argv[5])));
	
	cv::imwrite(argv[3], image);
	
	return 0;
}
