#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <iostream>

using namespace lsd_slam;

int main(int argc, char **argv)
{
	std::vector<Ray> rays = {
		Ray{vec3(0.f, 0.f, 0.f), vec3(1.f, 0.f, 0.f)},
		Ray{vec3(0.f, 0.f, 0.f), vec3(1.f, 0.f, 0.f)},

		Ray{vec3(0.f, 0.f, 0.f), vec3(1.f, 1.f, 0.f).normalized()},
		Ray{vec3(0.f, 1.f, 0.f), vec3(1.f, 0.f, 0.f)}
	};

	std::vector<RayIntersectionResult> expectedResults = {
		RayIntersectionResult{
			RayIntersectionResult::PARALLEL, 0.f, vec3(0.f, 0.f, 0.f)},
		RayIntersectionResult{
			RayIntersectionResult::VALID, 0.f, vec3(1.f, 1.f, 0.f)}
	};

	for (size_t i = 0; i < expectedResults.size(); ++i) {
		RayIntersectionResult result =
			computeRayIntersection(rays[2 * i], rays[2 * i + 1]);

		std::cout << "Test " << i 
			<< "\n\tRay 0: " << rays[2*i] 
			<< "\n\tRay 1: " << rays[2*i+1]
			<< "\n\tExpected:\n" << expectedResults[i] <<
			"\n\tActual:\n" << result << std::endl;
	}

	return 0;
}
