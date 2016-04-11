add_library(lsd_slam_util
	EigenCoreInclude.hpp
	globalFuncs.cpp
	globalFuncs.hpp
	IndexThreadReduce.hpp
	settings.cpp
	settings.hpp
	SophusUtil.cpp
	SophusUtil.hpp
	Undistorter.cpp
	Undistorter.hpp
	Raycast.cpp
	Raycast.hpp
	ModelLoader.cpp
	ModelLoader.hpp
	ply.h
	plyfile.c
)

target_link_libraries(lsd_slam_util
	${ASSIMP_LIBRARY}
)

