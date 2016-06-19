add_library(lsd_slam_GlobalMapping
	FabMap.cpp
	FabMap.hpp
	g2oTypeSim3Sophus.cpp
	g2oTypeSim3Sophus.hpp
	KeyframeGraph.cpp
	KeyframeGraph.hpp
	TrackableKeyframeSearch.cpp
	TrackableKeyframeSearch.hpp
)

set_target_properties(lsd_slam_GlobalMapping PROPERTIES FOLDER lsd_slam)

