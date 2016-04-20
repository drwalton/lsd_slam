add_executable(MakeRaycastedImage
	MakeRaycastedImage.cpp
)

target_link_libraries(MakeRaycastedImage
	lsd_slam_util
	lsd_slam_omni
	optimized ${OPENCV_LIBRARIES} debug ${OPENCV_DEBUG_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
)

add_executable(MakeRaycastedVideo
	MakeRaycastedVideo.cpp
)

target_link_libraries(MakeRaycastedVideo
	lsd_slam_util
	lsd_slam_omni
	optimized ${OPENCV_LIBRARIES} debug ${OPENCV_DEBUG_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
)

add_executable(TestOmniStereo
	TestOmniStereo.cpp
)

target_link_libraries(TestOmniStereo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
)

add_executable(TestSE3Tracking
	TestSE3Tracking.cpp
)

target_link_libraries(TestSE3Tracking
	${LSD_SLAM_LIBS}
	lsd_slam_DepthEstimation
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
)
link_libs_exec_debug(TestSE3Tracking G2O_LIBRARIES G2O_LIBRARIES_DEBUG)

add_executable(RenderAndTestSE3Tracking
	RenderAndTestSE3Tracking.cpp
)

target_link_libraries(RenderAndTestSE3Tracking
	${LSD_SLAM_LIBS}
	lsd_slam_DepthEstimation
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
)
link_libs_exec_debug(RenderAndTestSE3Tracking G2O_LIBRARIES G2O_LIBRARIES_DEBUG)

