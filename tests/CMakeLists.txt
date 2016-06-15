add_executable(TestCamModels
	TestCamModels.cpp)

set_target_properties(TestCamModels PROPERTIES FOLDER Tests)

target_link_libraries(TestCamModels
	${LSD_SLAM_LIBS}
)

add_executable(TestConvertDepthMaps
	TestConvertDepthMaps.cpp)

set_target_properties(TestConvertDepthMaps PROPERTIES FOLDER Tests)

target_link_libraries(TestConvertDepthMaps
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
)

add_executable(TestStereoMatching
	TestStereoMatching.cpp)

set_target_properties(TestStereoMatching PROPERTIES FOLDER Tests)

target_link_libraries(TestStereoMatching
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
)

