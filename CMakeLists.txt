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
find_package(Boost REQUIRED)

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
)

add_subdirectory(lsd_slam_core)
add_subdirectory(lsd_slam_viewer)

set(LSD_SLAM_LIBS
	lsd_slam_System
	lsd_slam_DataStructures
	lsd_slam_DepthEstimation
	lsd_slam_GlobalMapping
	lsd_slam_IOWrapper
	lsd_slam_Tracking
	lsd_slam_util
	lsd_slam_viewer
)

add_subdirectory(demos)

