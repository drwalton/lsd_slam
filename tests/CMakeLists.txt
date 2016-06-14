add_executable(TestCamModels
	TestCamModels.cpp)

target_link_libraries(TestCamModels
	${LSD_SLAM_LIBS}
)

