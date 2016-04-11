#include "ModelLoader.hpp"
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <iostream>

namespace lsd_slam {

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
	void centerMesh();
};


void ModelLoader::Impl::loadFile(const std::string &filename)
{
	loadFileAssimp(filename);
}

void ModelLoader::Impl::saveFile(const std::string &filename)
{
	saveFileAssimp(filename);
}

void ModelLoader::Impl::loadFileAssimp(const std::string &filename)
{
	Assimp::Importer importer;
	importer.ReadFile(filename.c_str(),
		aiProcess_GenSmoothNormals |
		aiProcess_Triangulate |
		aiProcess_FlipWindingOrder);

	const aiScene *scene = importer.GetScene();

	if (scene == nullptr)
	{
		throw std::runtime_error(("File \"" + filename +
			"\" could not be loaded").c_str());
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
	/*
	Assimp::Exporter exporter;

	aiScene scene;

	scene.mMeshes[0] = new aiMesh;
	scene.mNumMeshes = 1;
	aiMesh *mesh = scene.mMeshes[0];

	mesh->mVertices = reinterpret_cast<aiVector3D*>(verts.data());
	mesh->mNumVertices = verts.size();

	if (indices.size() > 0)
	{
		mesh->mNumFaces = (indices.size() / 3);
		mesh->mFaces = new aiFace[indices.size() / 3];
		for (size_t f = 0; f < mesh->mNumFaces; ++f)
		{
			aiFace &face = mesh->mFaces[f];
			face.mNumIndices = 3;
			face.mIndices = new unsigned int[3];
			for (size_t i = 1; i < 3; ++i)
			{
				face.mIndices[i] = indices[3 * f + i];
			}
		}
	}

	if (normals.size() == verts.size())
	{
		mesh->mNormals = reinterpret_cast<aiVector3D*>(verts.data());
	}

	if (vertColors.size() == verts.size())
	{
		//TODO implement vertex colors
		std::cout << "Warning: not saving vertex colors!" << std::endl;
	}

	if (texCoords.size() == verts.size())
	{
		mesh->mNumUVComponents[0] = 2;
		mesh->mTextureCoords[0] = new aiVector3D[verts.size()];

		for (size_t i = 0; i < verts.size(); ++i)
		{
			aiVector3D &t = mesh->mTextureCoords[0][i];
			t.x = texCoords[i](0);
			t.y = texCoords[i](1);
			t.z = 0.0f;
		}
	}

	exporter.Export(&scene,
		filename.substr(filename.find_last_of(".") + 1), filename);
		*/
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

void ModelLoader::scale(float dim)
{
	//float dist = dim * 0.5f;
	//TODO
	throw std::runtime_error("ModelLoader::scale NOT IMPLEMENTED");
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

}
