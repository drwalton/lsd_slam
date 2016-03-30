add_subdirectory(DataStructures)
add_subdirectory(DepthEstimation)
add_subdirectory(GlobalMapping)
add_subdirectory(IOWrapper)
add_subdirectory(Tracking)
add_subdirectory(util)

set(COMPATIBILITY_SOURCES 
	Win32Compatibility.hpp
	Win32Compatibility.cpp
	)

add_library(lsd_slam_System
	${COMPATIBILITY_SOURCES}
	LiveSLAMWrapper.cpp
	LiveSLAMWrapper.hpp
	SlamSystem.cpp
	SlamSystem.hpp
)

