add_executable(TestCamModels
	TestCamModels.cpp)

set_target_properties(TestCamModels PROPERTIES FOLDER Tests)

target_link_libraries(TestCamModels
	${LSD_SLAM_LIBS}
)

