#ifndef KEYFRAMEMSG_HPP_INCLUDED
#define KEYFRAMEMSG_HPP_INCLUDED

#include <array>
#include <vector>
#include <cstdint>

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
	
	std::vector<char> pointcloud;
};

#endif
