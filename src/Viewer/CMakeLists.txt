add_library(lsd_slam_Viewer
	KeyFrameDisplay.cpp
	KeyFrameDisplay.hpp
	KeyFrameGraphDisplay.cpp
	KeyFrameGraphDisplay.hpp
	keyframeMsg.cpp
	keyframeMsg.hpp
	PointCloudViewer.cpp
	PointCloudViewer.hpp
	settings.cpp
	settings.hpp
)

set_target_properties(lsd_slam_Viewer PROPERTIES FOLDER lsd_slam)

