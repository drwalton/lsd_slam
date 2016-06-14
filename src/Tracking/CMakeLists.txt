add_library(lsd_slam_Tracking
	LGSX.hpp
	Relocalizer.cpp
	Relocalizer.hpp
	SE3Tracker.cpp
	SE3Tracker.hpp
	Sim3Tracker.cpp
	Sim3Tracker.hpp
	TrackingReference.cpp
	TrackingReference.hpp
)

set_target_properties(lsd_slam_Tracking PROPERTIES FOLDER lsd_slam)

