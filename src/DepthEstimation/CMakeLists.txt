add_library(lsd_slam_DepthEstimation
	DepthMap.cpp
	DepthMap.hpp
	DepthMapDebugImages.cpp
	DepthMapDebugImages.hpp
	DepthMapDebugDefines.hpp
	DepthMapInitMode.hpp
	DepthMapOmni.cpp
	DepthMapOmni.hpp
	DepthMapPixelHypothesis.cpp
	DepthMapPixelHypothesis.hpp
)

set_target_properties(lsd_slam_DepthEstimation PROPERTIES FOLDER lsd_slam)

