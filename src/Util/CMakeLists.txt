add_library(lsd_slam_Util
	Constants.cpp
	Constants.hpp
	EigenCoreInclude.hpp
	globalFuncs.cpp
	globalFuncs.hpp
	ImgProc.cpp
	ImgProc.hpp
	IndexThreadReduce.hpp
	KahanVal.cpp
	KahanVal.hpp
	ModelLoader.cpp
	ModelLoader.hpp
	PolylineModelMaker.cpp
	PolylineModelMaker.hpp
	settings.cpp
	settings.hpp
	SophusUtil.cpp
	SophusUtil.hpp
	Undistorter.cpp
	Undistorter.hpp
	VectorTypes.cpp
	VectorTypes.hpp
)

set_target_properties(lsd_slam_Util PROPERTIES FOLDER lsd_slam)

