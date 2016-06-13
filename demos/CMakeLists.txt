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
link_libs_exec_debug(ReconstructionDemo QGLVIEWER_LIBRARY_RELEASE QGLVIEWER_LIBRARY_DEBUG)
link_libs_exec_debug(ReconstructionDemo QT_LIBRARIES_RELEASE QT_LIBRARIES_DEBUG)
message("Boost: ${Boost_LIBRARIES}")
target_link_libraries(ReconstructionDemo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Boost_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${libtiff_LIBRARY}
	${Assimp_LIBRARY}
)
if(WIN32)
	target_link_libraries(ReconstructionDemo
		${GLEW_LIBRARIES})
endif(WIN32)

