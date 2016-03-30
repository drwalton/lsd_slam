add_executable(ReconstructionDemo ReconstructionDemo.cpp)

link_libs_exec_debug(ReconstructionDemo G2O_LIBRARIES G2O_LIBRARIES_DEBUG)
if(NOT WIN32)
	target_link_libraries(ReconstructionDemo
		${CHOLMOD_LIBRARY} ${CSPARSE_LIBRARY})
endif(NOT WIN32)
link_libs_exec_debug(ReconstructionDemo FLTK_LIBRARIES FLTK_LIBRARIES_DEBUG)
message("Boost: ${Boost_LIBRARIES}")
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Boost_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)

