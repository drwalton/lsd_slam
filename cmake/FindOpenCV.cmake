find_path(OPENCV_INCLUDE_DIRS
	NAMES
		opencv2/opencv.hpp
	PATHS
		/usr/local/Cellar/opencv3/3.1.0_1/include
		"C:/local/opencv/opencv-3.0.0/build/include"
	NO_DEFAULT_PATH
)

if(NOT ${OPENCV_INCLUDE_DIRS} EQUAL OPENCV_INCLUDE_DIRS-NOTFOUND)
	set(OPENCV_FOUND TRUE)
endif(NOT ${OPENCV_INCLUDE_DIRS} EQUAL OPENCV_INCLUDE_DIRS-NOTFOUND)

set(LIB_PATHS
		/usr/local/Cellar/opencv3/3.1.0_1/lib
		"C:/local/opencv/opencv-3.0.0/build/x86/vc12/lib"
)

if(WIN32)
find_library(OPENCV_WORLD
	NAMES
		opencv_world300
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_WORLDD
	NAMES
		opencv_world300d
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)

set(OPENCV_LIBRARIES
	${OPENCV_WORLD})
set(OPENCV_DEBUG_LIBRARIES
	${OPENCV_WORLDD})
else(WIN32)
	
find_library(OPENCV_CALIB3D_LIBRARY
	NAMES
		opencv_calib3d
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_CORE_LIBRARY
	NAMES
		opencv_core
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_FEATURES2D_LIBRARY
	NAMES
		opencv_features2d
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_FLANN_LIBRARY
	NAMES
		opencv_flann
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_HIGHGUI_LIBRARY
	NAMES
		opencv_highgui
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_IMGPROC_LIBRARY
	NAMES
		opencv_imgproc
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_VIDEOIO_LIBRARY
	NAMES
		opencv_videoio
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)
find_library(OPENCV_IMGCODECS_LIBRARY
	NAMES
		opencv_imgcodecs
	PATHS
		${LIB_PATHS}
	NO_DEFAULT_PATH
)

set(OPENCV_LIBRARIES
	${OPENCV_CALIB3D_LIBRARY}
	${OPENCV_CORE_LIBRARY}
	${OPENCV_FEATURES2D_LIBRARY}
	${OPENCV_FLANN_LIBRARY}
	${OPENCV_HIGHGUI_LIBRARY}
	${OPENCV_IMGPROC_LIBRARY}
	${OPENCV_VIDEOIO_LIBRARY}
	${OPENCV_IMGCODECS_LIBRARY}
)
set(OPENCV_DEBUG_LIBRARIES
	${OPENCV_LIBRARIES})
endif(WIN32)
