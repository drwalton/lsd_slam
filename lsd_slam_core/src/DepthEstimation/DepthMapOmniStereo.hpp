#ifndef DEPTHESTIMATION_DEPTHMAPOMNISTEREO_HPP_INCLUDED
#define DEPTHESTIMATION_DEPTHMAPOMNISTEREO_HPP_INCLUDED

#include "OmniCameraModel.hpp"
#include <array>

namespace lsd_slam {

std::array<float, 5> findValuesToSearchFor(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe,
	int x, int y,
	int width,
	vec3 &pointDir);

void findReferenceFrameLineEndpoints(
	vec3 &pStart, vec3 &pEnd,
	int u, int v,
	float depth, float depthStd);

bool findLowestSSDMatch(
	const RigidTransform &keyframeToReference,
	const OmniCameraModel &model,
	const float* keyframe
	);

}

#endif
