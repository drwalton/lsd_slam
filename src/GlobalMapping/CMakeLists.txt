add_library(lsd_slam_GlobalMapping
	FabMap.cpp
	FabMap.hpp
	g2oTypeSim3Sophus.cpp
	g2oTypeSim3Sophus.hpp
	KeyFrameGraph.cpp
	KeyFrameGraph.hpp
	TrackableKeyFrameSearch.cpp
	TrackableKeyFrameSearch.hpp
)

set_target_properties(lsd_slam_GlobalMapping PROPERTIES FOLDER lsd_slam)

