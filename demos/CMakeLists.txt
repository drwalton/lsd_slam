add_executable(ReconstructionDemo ReconstructionDemo.cpp)
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
)
link_libs_exec_debug(ReconstructionDemo G2O_LIBRARIES G2O_LIBRARIES_DEBUG)
link_libs_exec_debug(ReconstructionDemo FLTK_LIBRARIES FLTK_LIBRARIES_DEBUG)

