add_executable(ReconstructionDemo ReconstructionDemo.cpp)
set_target_properties(ReconstructionDemo PROPERTIES FOLDER Demos)
set(QT_LIBRARIES_RELEASE
	${Qt5Core_LIBRARIES}
	${Qt5Widgets_LIBRARIES}
	${Qt5OpenGL_LIBRARIES}
	)
set(QT_LIBRARIES_DEBUG
	${Qt5Core_LIBRARIES}
	${Qt5Widgets_LIBRARIES}
	${Qt5OpenGL_LIBRARIES}
	)

message("QT RELEASE: ${QT_LIBRARIES_RELEASE}")

link_libs_exec_debug(ReconstructionDemo G2O_LIBRARIES G2O_LIBRARIES_DEBUG)
if(NOT WIN32)
	target_link_libraries(ReconstructionDemo
		${CHOLMOD_LIBRARY} ${CSPARSE_LIBRARY})
endif(NOT WIN32)
link_libs_exec_debug(ReconstructionDemo QGLVIEWER_LIBRARY_RELEASE QGLVIEWER_LIBRARY_DEBUG)
link_libs_exec_debug(ReconstructionDemo QT_LIBRARIES_RELEASE QT_LIBRARIES_DEBUG)
link_libs_exec_debug(ReconstructionDemo OPENCV_LIBRARIES OPENCV_DEBUG_LIBRARIES)
message("Boost: ${Boost_LIBRARIES}")
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${Boost_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${libtiff_LIBRARY}
	${Assimp_LIBRARY}
)
if(WIN32)
	target_link_libraries(ReconstructionDemo
		${GLEW_LIBRARIES})
endif(WIN32)

