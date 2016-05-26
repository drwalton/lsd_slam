add_executable(ReconstructionDemo ReconstructionDemo.cpp)

set(QT_LIBRARIES_RELEASE
	${QT_QTCORE_LIBRARY_RELEASE}
	${QT_QTGUI_LIBRARY_RELEASE}
	${QT_QTOPENGL_LIBRARY_RELEASE}
	)
set(QT_LIBRARIES_DEBUG
	${QT_QTCORE_LIBRARY_DEBUG}
	${QT_QTGUI_LIBRARY_DEBUG}
	${QT_QTOPENGL_LIBRARY_DEBUG}
	)

link_libs_exec_debug(ReconstructionDemo G2O_LIBRARIES G2O_LIBRARIES_DEBUG)
if(NOT WIN32)
	target_link_libraries(ReconstructionDemo
		${CHOLMOD_LIBRARY} ${CSPARSE_LIBRARY})
endif(NOT WIN32)
link_libs_exec_debug(ReconstructionDemo FLTK_LIBRARIES FLTK_LIBRARIES_DEBUG)
link_libs_exec_debug(ReconstructionDemo QT_LIBRARIES_RELEASE QT_LIBRARIES_DEBUG)
link_libs_exec_debug(ReconstructionDemo OPENCV_LIBRARIES OPENCV_DEBUG_LIBRARIES)
target_link_libraries(ReconstructionDemo optimized ${QGLVIEWER_LIBRARY_RELEASE} debug ${QGLVIEWER_LIBRARY_DEBUG})
message("QGLVIEWER ${QGLVIEWER_LIBRARY}")
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${Boost_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${libtiff_LIBRARY}
)
if(WIN32)
	target_link_libraries(ReconstructionDemo
		${GLEW_LIBRARIES})
endif(WIN32)

