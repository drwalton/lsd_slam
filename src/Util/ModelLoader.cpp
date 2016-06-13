#include "Util/ModelLoader.hpp"
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <iostream>
#include <fstream>

namespace lsd_slam {

void saveCloudToPlyFile(const std::string &filename,
	const std::vector<vec3> &positions,
	const std::vector<vec3> &colors,
	const std::vector<vec3> &normals,
	const std::vector<float> &radii,
	bool binary);
void saveMeshToPlyFile(const std::string &filename,
	const std::vector<vec3> &positions,
	const std::vector<vec3> &colors,
	const std::vector<vec3> &normals,
	const std::vector<float> &radii,
	const std::vector<GLuint> &indices,
	bool binary);


struct ModelLoader::Impl
{
	std::vector<vec3> verts;
	std::vector<GLuint> indices;
	std::vector<vec3> normals;
	std::vector<vec2> texCoords;
	std::vector<vec3> vertColors;

	void loadFile(const std::string &filename);
	void saveFile(const std::string &filename);
	bool plyIsPointCloud(const std::string &filename);
	void loadFileAssimp(const std::string &filename);
	void saveFileAssimp(const std::string &filename);
	void loadPlyFile(const std::string &filename);
	void centerMesh();
};


void ModelLoader::Impl::loadFile(const std::string &filename)
{
	//if(filename.substr(filename.length() - 3, filename.length()) == "ply") {
	//	loadPlyFile(filename);
	//} else {
		loadFileAssimp(filename);
	//}
}

void ModelLoader::Impl::saveFile(const std::string &filename)
{
	saveFileAssimp(filename);
}

void ModelLoader::Impl::loadFileAssimp(const std::string &filename)
{
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(filename.c_str(),
		aiProcess_GenSmoothNormals |
		aiProcess_Triangulate |
		aiProcess_FlipWindingOrder);

	if (scene == nullptr)
	{
		std::string assimpError(importer.GetErrorString());
		throw std::runtime_error(("File \"" + filename +
			"\" could not be loaded: " + assimpError).c_str());
	}

	if (scene->mNumMeshes == 0)
	{
		throw std::runtime_error(("File \"" + filename + "\" loaded, but does not "
			"appear to contain any meshes").c_str());
	}

	const aiMesh *mesh = scene->mMeshes[0];

	size_t maxIndex = 0;

	//NOTE: Here we assume all faces are triangles.
	size_t nIndices = mesh->mNumFaces * 3;

	indices.resize(nIndices);
	for (size_t f = 0; f < mesh->mNumFaces; ++f)
	{
		indices[f * 3 + 0] = mesh->mFaces[f].mIndices[0];
		indices[f * 3 + 1] = mesh->mFaces[f].mIndices[2];
		indices[f * 3 + 2] = mesh->mFaces[f].mIndices[1];
	}

	for (size_t i = 0; i < mesh->mNumFaces * 3; ++i)
	{
		if (indices[i] > maxIndex) maxIndex = indices[i];
	}

	if (mesh->HasPositions())
	{
		verts.resize(maxIndex+1);
		std::copy(mesh->mVertices,
			mesh->mVertices + maxIndex+1, (aiVector3D*)verts.data());
	}

	if (mesh->HasNormals())
	{

		normals.resize(maxIndex+1);
		std::copy(mesh->mNormals,
			mesh->mNormals + maxIndex+1, (aiVector3D*)normals.data());
	}

	if (mesh->HasTextureCoords(0))
	{
		texCoords.resize(maxIndex+1);
		for (size_t i = 0; i <= maxIndex; ++i)
		{
			texCoords[i](0) = mesh->mTextureCoords[0][i].x;
			texCoords[i](1) = mesh->mTextureCoords[0][i].y;
		}
	}

	if (mesh->HasVertexColors(0)) {
		vertColors.resize(maxIndex+1);
		float s = 1.f;// / 255.f;
		for (size_t i = 0; i <= maxIndex; ++i) {
			vertColors[i](0) = s*mesh->mColors[0][i][0];
			vertColors[i](1) = s*mesh->mColors[0][i][1];
			vertColors[i](2) = s*mesh->mColors[0][i][2];
		}
	}

}

void ModelLoader::Impl::saveFileAssimp(const std::string &filename)
{
	std::vector<float> radii;
	if (indices.size() == 0) {
		saveCloudToPlyFile(filename, verts, vertColors, normals, radii, false);
	} else {
		saveMeshToPlyFile(filename, verts, vertColors, normals, radii, indices, false);
	//	Assimp::Exporter exporter;

	//	aiScene scene;
	//	scene.mRootNode = new aiNode; 
	//	aiMesh *p;
	//	unsigned int mMeshes[] = { 0 };
	//	scene.mRootNode->mMeshes = mMeshes;
	//	scene.mRootNode->mName = "";
	//	scene.mRootNode->mNumChildren = 0;
	//	scene.mRootNode->mNumMeshes = 1;
	//	scene.mRootNode->mParent = nullptr;
	//	aiMatrix4x4 transformation = {
	//		1.f, 0.f, 0.f, 0.f,
	//		0.f, 1.f, 0.f, 0.f,
	//		0.f, 0.f, 1.f, 0.f,
	//		0.f, 0.f, 0.f, 1.f
	//	};
	//	scene.mRootNode->mTransformation = transformation;

	//	scene.mMeshes = &p;
	//	scene.mMeshes[0] = new aiMesh;
	//	scene.mNumMeshes = 1;
	//	aiMesh *mesh = scene.mMeshes[0];
	//	mesh->mVertices = reinterpret_cast<aiVector3D*>(verts.data());
	//	mesh->mNumVertices = verts.size();

	//	if (indices.size() > 0)
	//	{
	//		mesh->mNumFaces = (indices.size() / 3);
	//		mesh->mFaces = new aiFace[indices.size() / 3];
	//		for (size_t f = 0; f < mesh->mNumFaces; ++f)
	//		{
	//			aiFace &face = mesh->mFaces[f];
	//			face.mNumIndices = 3;
	//			face.mIndices = new unsigned int[3];
	//			for (size_t i = 1; i < 3; ++i)
	//			{
	//				face.mIndices[i] = indices[3 * f + i];
	//			}
	//		}
	//	}

	//	if (normals.size() == verts.size())
	//	{
	//		mesh->mNormals = reinterpret_cast<aiVector3D*>(verts.data());
	//	}

	//	if (vertColors.size() == verts.size())
	//	{
	//		//TODO implement vertex colors
	//		std::cout << "Warning: not saving vertex colors!" << std::endl;
	//	}

	//	if (texCoords.size() == verts.size())
	//	{
	//		mesh->mNumUVComponents[0] = 2;
	//		mesh->mTextureCoords[0] = new aiVector3D[verts.size()];

	//		for (size_t i = 0; i < verts.size(); ++i)
	//		{
	//			aiVector3D &t = mesh->mTextureCoords[0][i];
	//			t.x = texCoords[i](0);
	//			t.y = texCoords[i](1);
	//			t.z = 0.0f;
	//		}
	//	}

	//	exporter.Export(&scene,
	//		filename.substr(filename.find_last_of(".") + 1), filename);
	}
}

ModelLoader::ModelLoader()
	:pimpl_(new Impl)
{}

ModelLoader::ModelLoader(const std::string &filename)
	: pimpl_(new Impl)
{
	loadFile(filename);
}

ModelLoader::~ModelLoader() throw()
{
}

void ModelLoader::loadFile(const std::string & filename)
{
	pimpl_->loadFile(filename);
}

void ModelLoader::saveFile(const std::string &filename)
{
	pimpl_->saveFile(filename);
}

void ModelLoader::center()
{
	if (pimpl_->verts.size() == 0)
	{
		throw std::runtime_error("ModelLoader::center() called on empty mesh!");
	}

	//KahanVal<vec3> centroidSum(vec3::Zero());
	vec3 centroidSum(vec3::Zero());

	for (auto &vert : pimpl_->verts)
	{
		centroidSum += vert;
	}

	vec3 centroid = centroidSum /
		static_cast<float>(pimpl_->verts.size());

	for (auto &vert : pimpl_->verts)
	{
		vert -= centroid;
	}
}

void ModelLoader::uniformScale(float dim)
{
	//float dist = dim * 0.5f;
	for (vec3 & v : pimpl_->verts) {
		v *= dim;
	}
}

std::vector<vec3> &ModelLoader::vertices()
{
	return pimpl_->verts;
}

std::vector<GLuint> &ModelLoader::indices()
{
	return pimpl_->indices;
}

bool ModelLoader::hasNormals() const
{
	return pimpl_->normals.size() == pimpl_->verts.size();
}

std::vector<vec3> &ModelLoader::normals()
{
	return pimpl_->normals;
}

bool ModelLoader::hasTexCoords() const
{
	return pimpl_->texCoords.size() == pimpl_->verts.size();
}

std::vector<vec2> &ModelLoader::texCoords()
{
	return pimpl_->texCoords;
}

bool ModelLoader::hasVertColors() const
{
	return pimpl_->vertColors.size() == pimpl_->verts.size();
}

std::vector<vec3> &ModelLoader::vertColors()
{
	return pimpl_->vertColors;
}


void ModelLoader::Impl::loadPlyFile(const std::string &filename)
{
	std::ifstream f(filename);
	std::string line;
	std::getline(f, line);
	std::getline(f, line);
	if(line != "ply") {
		throw std::runtime_error("Bad PLY format!");
	}
	
	std::string formatLine;
	std::getline(f, formatLine);
	if(formatLine == "format ascii 1.0") {
		//ASCII
	} else if(formatLine == "format binary 1.0") {
		//BINARY
		throw std::runtime_error("importing binary ply files not implemented yet!");
	} else {
		throw std::runtime_error("bad format line!");
	}
	
	size_t nVerts = 0, nElems = 0;
	bool hasNorms = false, hasColors = false;
	std::getline(f, line);
	while(line != "end_header") {
		if(line.substr(0, 14) == "element vertex") {
			nVerts = atoi((line.substr(14, line.size())).c_str());
		} else if(line.substr(0, 12) == "element face") {
			nElems = 3 * atoi((line.substr(12, line.size())).c_str());
		} else if(line == "property float nx") {
			hasNorms = true;
		} else if(line == "property uchar red") {
			hasColors = true;
		}
		
		std::getline(f, line);
	}
	
	//Get Verts
	verts.resize(nVerts);
	if(hasNorms) normals.resize(nVerts);
	if(hasColors) vertColors.resize(nVerts);
	for(size_t v = 0; v < nVerts; ++v) {
		f >> verts[v].x();
		f >> verts[v].y();
		f >> verts[v].z();
		if(hasNorms) {
    		f >> normals[v].x();
    		f >> normals[v].y();
    		f >> normals[v].z();
		}
		if(hasColors) {
			f >> vertColors[v].x();
			f >> vertColors[v].y();
			f >> vertColors[v].z();
		}
	}
	
	//Get Elems
	indices.resize(nElems);
	size_t nIndicesInFace;
	for(size_t e = 0; e < nElems / 3; ++e) {
		f >> nIndicesInFace;
		if(nIndicesInFace != 4) throw std::runtime_error("Can't deal with faces of more than 3 indices!");
		f >> indices[e*3];
		f >> indices[e*3 + 1];
		f >> indices[e*3 + 2];
	}
}

void saveCloudToPlyFile(const std::string &filename,
	const std::vector<vec3> &positions,
	const std::vector<vec3> &colors,
	const std::vector<vec3> &normals,
	const std::vector<float> &radii,
	bool binary)
{

	std::ios::openmode flags = std::ios::trunc;
	if (binary) flags = std::ios::binary | std::ios::trunc;
	std::ofstream file(filename, flags);
	if (file.fail()) throw std::runtime_error("Could not open file \"" +
		filename + "\" to write PLY file.");

	//Write header.
	file << "ply\n";
	if (binary) {
		file << "format binary_little_endian 1.0\n";
	}
	else {
		file << "format ascii 1.0\n";
	}
	file
		<< "comment author: David R. Walton\n"
		<< "comment object: Saved Reconstruction\n"
		<< "element vertex " << positions.size() << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n";

	if (colors.size() == positions.size()) {
		file
			<< "property uchar red\n"
			<< "property uchar green\n"
			<< "property uchar blue\n";
	}
	if (normals.size() == positions.size()) {
		file
			<< "property float nx\n"
			<< "property float ny\n"
			<< "property float nz\n";
	}
	if (radii.size() == positions.size()) {
		file
			<< "property float radius\n";
	}
	file
		<< "end_header\n";

	if (binary) {
		//Write content.
		for (size_t i = 0; i < positions.size(); ++i) {
			file.write((const char*)(&(positions[i].x())), sizeof(float))
				.write((const char*)(&(positions[i].y())), sizeof(float))
				.write((const char*)(&(positions[i].z())), sizeof(float));
			if (colors.size() == positions.size()) {
				cv::Vec3b c(colors[i].x(), colors[i].y(), colors[i].z());
				file.write((const char*)(&(colors[i][0])), sizeof(char))
					.write((const char*)(&(colors[i][1])), sizeof(char))
					.write((const char*)(&(colors[i][2])), sizeof(char));
			}
			if (normals.size() == positions.size()) {
				file.write((const char*)(&(normals[i].x())), sizeof(float))
					.write((const char*)(&(normals[i].y())), sizeof(float))
					.write((const char*)(&(normals[i].z())), sizeof(float));
			}
			if (radii.size() == positions.size()) {
				file.write((const char*)(&(radii[i])), sizeof(float));
			}
		}
	}
	else {
		file << std::fixed;

		//Write content.
		for (size_t i = 0; i < positions.size(); ++i) {
			file
				<< positions[i].x() << " "
				<< positions[i].y() << " "
				<< positions[i].z() << " ";
			if (colors.size() == positions.size()) {
				file
					<< (unsigned int)(colors[i][0]) << " "
					<< (unsigned int)(colors[i][1]) << " "
					<< (unsigned int)(colors[i][2]) << " ";
			}
			if (normals.size() == positions.size()) {
				file
					<< normals[i].x() << " "
					<< normals[i].y() << " "
					<< normals[i].z() << " ";
			}
			if (radii.size() == positions.size()) {
				file
					<< radii[i];
			} 
			file << "\n";

		}

		file << "\n";
	}
}

void saveMeshToPlyFile(const std::string &filename,
	const std::vector<vec3> &positions,
	const std::vector<vec3> &colors,
	const std::vector<vec3> &normals,
	const std::vector<float> &radii,
	const std::vector<GLuint> &indices,
	bool binary)
{

	std::ios::openmode flags = std::ios::trunc;
	if (binary) flags = std::ios::binary | std::ios::trunc;
	std::ofstream file(filename, flags);
	if (file.fail()) throw std::runtime_error("Could not open file \"" +
		filename + "\" to write PLY file.");

	//Write header.
	file << "ply\n";
	if (binary) {
		file << "format binary_little_endian 1.0\n";
	}
	else {
		file << "format ascii 1.0\n";
	}
	file
		<< "comment author: David R. Walton\n"
		<< "comment object: Saved Reconstruction\n"
		<< "element vertex " << positions.size() << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n";

	if (colors.size() == positions.size()) {
		file
			<< "property uchar red\n"
			<< "property uchar green\n"
			<< "property uchar blue\n";
	}
	if (normals.size() == positions.size()) {
		file
			<< "property float nx\n"
			<< "property float ny\n"
			<< "property float nz\n";
	}
	if (radii.size() == positions.size()) {
		file
			<< "property float radius\n";
	}
	file << "element face " << indices.size() / 3;
	file << "\nproperty list uchar uint vertex_indices\n";
	file << "end_header\n";

	if (binary) {
		//Write content.
		for (size_t i = 0; i < positions.size(); ++i) {
			file.write((const char*)(&(positions[i].x())), sizeof(float))
				.write((const char*)(&(positions[i].y())), sizeof(float))
				.write((const char*)(&(positions[i].z())), sizeof(float));
			if (colors.size() == positions.size()) {
				cv::Vec3b c(colors[i].x(), colors[i].y(), colors[i].z());
				file.write((const char*)(&(colors[i][0])), sizeof(char))
					.write((const char*)(&(colors[i][1])), sizeof(char))
					.write((const char*)(&(colors[i][2])), sizeof(char));
			}
			if (normals.size() == positions.size()) {
				file.write((const char*)(&(normals[i].x())), sizeof(float))
					.write((const char*)(&(normals[i].y())), sizeof(float))
					.write((const char*)(&(normals[i].z())), sizeof(float));
			}
			if (radii.size() == positions.size()) {
				file.write((const char*)(&(radii[i])), sizeof(float));
			}
		}
		for (size_t i = 0; i < indices.size(); i++) {
			static const uchar three = 3;
			file.write((const char*)(&(three)), sizeof(uchar));
			file.write((const char*)(&(indices[i])), sizeof(GLuint));
		}
	}
	else {
		file << std::fixed;

		//Write content.
		for (size_t i = 0; i < positions.size(); ++i) {
			file
				<< positions[i].x() << " "
				<< positions[i].y() << " "
				<< positions[i].z() << " ";
			if (colors.size() == positions.size()) {
				file
					<< (unsigned int)(colors[i][0]) << " "
					<< (unsigned int)(colors[i][1]) << " "
					<< (unsigned int)(colors[i][2]) << " ";
			}
			if (normals.size() == positions.size()) {
				file
					<< normals[i].x() << " "
					<< normals[i].y() << " "
					<< normals[i].z() << " ";
			}
			if (radii.size() == positions.size()) {
				file
					<< radii[i];
			} 
			file << "\n";

		}

		for (size_t i = 0; i < indices.size(); i += 3) {
			file <<  "3 " 
				<< indices[i    ] << " " 
				<< indices[i + 1] << " "
				<< indices[i + 2] << "\n";
		}

		file << "\n";
	}
}

}
