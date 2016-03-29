add_executable(ReconstructionDemo ReconstructionDemo.cpp)
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
)

