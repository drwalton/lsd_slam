#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include "CameraModel/OmniCameraModel.hpp"
#include "CameraModel/ProjCameraModel.hpp"

namespace lsd_slam {

struct keyframeMsg
{
	int id;
	double time;
	bool isKeyframe;

	std::array<float, 7> camToWorld;

	float fx;
	float fy;
	float cx;
	float cy;
	uint32_t height;
	uint32_t width;
	float e;

	CameraModelType modelType;

	std::vector<char> pointcloud;
};

}
