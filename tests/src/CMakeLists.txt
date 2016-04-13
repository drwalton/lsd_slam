
add_executable(MakeRaycastedImage
	MakeRaycastedImage.cpp
)

target_link_libraries(MakeRaycastedImage
	lsd_slam_util
	lsd_slam_omni
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
)

add_executable(MakeRaycastedVideo
	MakeRaycastedVideo.cpp
)

target_link_libraries(MakeRaycastedVideo
	lsd_slam_util
	lsd_slam_omni
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
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

