add_library(lsd_slam_IOWrapper
	DirectoryImageStream.cpp
	DirectoryImageStream.hpp
	ImageDisplay.hpp
	InputImageStream.hpp
	InputImageStream.cpp
	NotifyBuffer.hpp
	OpenCV/ImageDisplay_OpenCV.cpp
	Output3DWrapper.hpp
	ViewerOutput3DWrapper.hpp
	ViewerOutput3DWrapper.cpp
	Timestamp.cpp
	Timestamp.hpp
	TimestampedObject.hpp
	OpenCVImageStream.hpp
	OpenCVImageStream.cpp
)

set_target_properties(lsd_slam_IOWrapper PROPERTIES FOLDER lsd_slam)

