cmake_minimum_required(VERSION 3.0)

project(lsd_slam)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
if(NOT WIN32)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++14)
endif(NOT WIN32)

find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost COMPONENTS System Thread REQUIRED)
find_package(libtiff REQUIRED)
find_package(Assimp REQUIRED)

if(WIN32)
	link_directories("${Boost_INCLUDE_DIR}/lib32-msvc-14.0")
endif(WIN32)

if(NOT WIN32)
find_package(SuiteParse REQUIRED)
else(NOT WIN32)
	set(CSPARSE_INCLUDE_DIR ${G2O_INCLUDE_DIR}/EXTERNAL/csparse)
	set(CHOLMOD_INCLUDE_DIR "")
	include_directories("C:/local/zlib/zlib-1.2.3-lib/include")
endif(NOT WIN32)

find_package(X11 REQUIRED)
find_package(QGLViewer REQUIRED)
message("QOPENGL: ${QT_QTOPENGL_INCLUDE_DIR}")
include_directories("${QT_QTOPENGL_INCLUDE_DIR}/..")

set(CMAKE_AUTOMOC ON)
include_directories(
	${GLEW_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/lsd_slam_core/src
	${CMAKE_SOURCE_DIR}/lsd_slam_viewer/src
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/thirdparty/Sophus
	${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
	${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
	${OPENCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
	${QGLVIEWER_INCLUDE_DIR}
	${G2O_INCLUDE_DIR}
	${ASSIMP_INCLUDE_DIRS}
	${libtiff_INCLUDE_DIRS}
)

include_directories(src)
add_subdirectory(src)

set(LSD_SLAM_LIBS
	lsd_slam_CameraModel
	lsd_slam_System
	lsd_slam_DataStructures
	lsd_slam_DepthEstimation
	lsd_slam_GlobalMapping
	lsd_slam_IOWrapper
	lsd_slam_Tracking
	lsd_slam_Util
	lsd_slam_Viewer
)

function(link_libs_exec_debug TARGET EXEC_LIBS DEBUG_LIBS)
	foreach(LIB IN LISTS ${EXEC_LIBS})
		target_link_libraries(${TARGET} optimized ${LIB})
	endforeach()
	foreach(LIB IN LISTS ${DEBUG_LIBS})
		target_link_libraries(${TARGET} debug ${LIB})
	endforeach()
endfunction()

add_subdirectory(demos)

