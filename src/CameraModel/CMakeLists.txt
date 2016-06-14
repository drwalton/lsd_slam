add_library(lsd_slam_CameraModel
	CameraModel.cpp
	CameraModel.hpp
	ConvertDepthMap.cpp
	ConvertDepthMap.hpp
	Constants.hpp
	OmniCameraModel.cpp
	OmniCameraModel.hpp
	ProjCameraModel.cpp
	ProjCameraModel.hpp
)

set_target_properties(lsd_slam_CameraModel PROPERTIES FOLDER lsd_slam)

