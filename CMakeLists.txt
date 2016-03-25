cmake_minimum_required(VERSION 3.0)

project(lsd_slam)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++14)

find_package(Eigen3 REQUIRED)
find_package(SuiteParse REQUIRED)
find_package(X11 REQUIRED)
find_package(QGLViewer REQUIRED)

include_directories(
	${CMAKE_SOURCE_DIR}/lsd_slam_core/src
	${CMAKE_SOURCE_DIR}/lsd_slam_viewer/src
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/thirdparty/Sophus
	${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
	${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
)

add_subdirectory(lsd_slam_core)
add_subdirectory(lsd_slam_viewer)

add_subdirectory(demos)

