add_executable(TestCamModels
	TestCamModels.cpp)

set_target_properties(TestCamModels PROPERTIES FOLDER Tests)

target_link_libraries(TestCamModels
	${LSD_SLAM_LIBS}
)
link_libs_exec_debug(TestCamModels OPENCV_LIBRARIES OPENCV_DEBUG_LIBRARIES)

add_executable(TestConvertDepthMaps
	TestConvertDepthMaps.cpp)

set_target_properties(TestConvertDepthMaps PROPERTIES FOLDER Tests)

target_link_libraries(TestConvertDepthMaps
	${LSD_SLAM_LIBS}
)
link_libs_exec_debug(TestConvertDepthMaps OPENCV_LIBRARIES OPENCV_DEBUG_LIBRARIES)

add_executable(TestStereoMatching
	TestStereoMatching.cpp)

set_target_properties(TestStereoMatching PROPERTIES FOLDER Tests)

target_link_libraries(TestStereoMatching
	${LSD_SLAM_LIBS}
	${Assimp_LIBRARY}
)
link_libs_exec_debug(TestStereoMatching G2O_LIBRARIES G2O_LIBRARIES_DEBUG)
link_libs_exec_debug(TestStereoMatching OPENCV_LIBRARIES OPENCV_DEBUG_LIBRARIES)

