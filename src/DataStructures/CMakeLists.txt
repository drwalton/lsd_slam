add_library(lsd_slam_DataStructures
	Frame.cpp
	Frame.hpp
	FrameMemory.cpp
	FrameMemory.hpp
	FramePoseStruct.cpp
	FramePoseStruct.hpp
)

set_target_properties(lsd_slam_DataStructures PROPERTIES FOLDER lsd_slam)

