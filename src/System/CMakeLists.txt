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

set_target_properties(lsd_slam_System PROPERTIES FOLDER lsd_slam)

