add_subdirectory(DataStructures)
add_subdirectory(DepthEstimation)
add_subdirectory(GlobalMapping)
add_subdirectory(IOWrapper)
add_subdirectory(Tracking)
add_subdirectory(util)

add_library(lsd_slam_System
	LiveSLAMWrapper.cpp
	LiveSLAMWrapper.h
	SlamSystem.cpp
	SlamSystem.h
)

