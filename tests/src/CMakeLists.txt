add_executable(MakeRaycastedImage
	MakeRaycastedImage.cpp
)

target_link_libraries(MakeRaycastedImage
	lsd_slam_util
	lsd_slam_omni
	optimized ${OPENCV_LIBRARIES} debug ${OPENCV_DEBUG_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
)

add_executable(MakeRaycastedVideo
	MakeRaycastedVideo.cpp
)

target_link_libraries(MakeRaycastedVideo
	lsd_slam_util
	lsd_slam_omni
	optimized ${OPENCV_LIBRARIES} debug ${OPENCV_DEBUG_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
)

add_executable(TestOmniStereoWholeIm
	TestOmniStereoWholeIm.cpp
)

target_link_libraries(TestOmniStereoWholeIm
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)
link_libs_exec_debug(TestOmniStereoWholeIm FLTK_LIB_RELEASE FLTK_LIB_DEBUG)

add_executable(TestProjStereoWholeIm
	TestProjStereoWholeIm.cpp
)

target_link_libraries(TestProjStereoWholeIm
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)
link_libs_exec_debug(TestProjStereoWholeIm FLTK_LIB_RELEASE FLTK_LIB_DEBUG)

add_executable(TestOmniStereo
	TestOmniStereo.cpp
)

target_link_libraries(TestOmniStereo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)

add_executable(TestProjStereo
	TestProjStereo.cpp
)

target_link_libraries(TestProjStereo
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)

add_executable(TestRayIntersection
	TestRayIntersection.cpp
)

target_link_libraries(TestRayIntersection
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)

add_executable(TestDepthMapGeneration
	TestDepthMapGeneration.cpp
)

target_link_libraries(TestDepthMapGeneration
	${LSD_SLAM_LIBS}
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARY}
)
link_libs_exec_debug(TestDepthMapGeneration G2O_LIBRARIES G2O_LIBRARIES_DEBUG)

add_executable(TestSE3Tracking
	TestSE3Tracking.cpp
)

target_link_libraries(TestSE3Tracking
	${LSD_SLAM_LIBS}
	lsd_slam_DepthEstimation
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
)
link_libs_exec_debug(TestSE3Tracking G2O_LIBRARIES G2O_LIBRARIES_DEBUG)

add_executable(RenderAndTestSE3Tracking
	RenderAndTestSE3Tracking.cpp
)

target_link_libraries(RenderAndTestSE3Tracking
	${LSD_SLAM_LIBS}
	lsd_slam_DepthEstimation
	${OPENCV_LIBRARIES}
	${Assimp_LIBRARY}
	${Boost_LIBRARIES}
	${libtiff_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)
link_libs_exec_debug(RenderAndTestSE3Tracking FLTK_LIB_RELEASE FLTK_LIB_DEBUG)
link_libs_exec_debug(RenderAndTestSE3Tracking G2O_LIBRARIES G2O_LIBRARIES_DEBUG)

