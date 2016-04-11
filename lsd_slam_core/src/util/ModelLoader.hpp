#ifndef LSD_SLAM_MODELLOADER_HPP_INCLUDED
#define LSD_SLAM_MODELLOADER_HPP_INCLUDED

#include "VectorTypes.hpp"
#include <vector>
#include <memory>
#include <GL/glew.h>

namespace lsd_slam {

	class ModelLoader final
	{
	public:
		explicit ModelLoader();
		explicit ModelLoader(const std::string &filename);
		~ModelLoader() throw();

		/// \brief Load a file.
		/// \note Deletes any content already present in the ModelLoader object.
		void loadFile(const std::string &filename);

		/// \brief Save data in ModelLoader object to a file.
		void saveFile(const std::string &filename);

		///\brief Move the centroid of the supplied mesh to the model-space origin.
		void center();

		///\brief Apply uniform scaling to a mesh so that all its points lie within
		/// an axis-aligned cube centred at the origin.
		///\param dim The length of a side of the cube.
		void scale(float dim = 2.0f);

		std::vector<Eigen::Vector3f> &vertices();
		std::vector<GLuint> &indices();

		bool hasNormals() const;
		std::vector<Eigen::Vector3f> &normals();

		bool hasTexCoords() const;
		std::vector<Eigen::Vector2f> &texCoords();

		bool hasVertColors() const;
		std::vector<Eigen::Vector3f> &vertColors();
	private:

		struct Impl;
		std::shared_ptr<Impl> pimpl_;
	};

}

#endif

