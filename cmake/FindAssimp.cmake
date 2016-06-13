if(CMAKE_SIZEOF_VOID_P EQUAL 4)
#32-bit, assume x86
	find_library(Assimp_LIBRARY
		NAMES
			assimp
			assimp-vc120-mt
		PATHS
			"C:/local/Assimp/assimp-3.2/lib32/Release/"
			"/opt/local/lib"
			"/usr/local/lib")
	find_library(Assimp_LIBRARY_DEBUG
		NAMES
			assimp
			assimp-vc120-mtd
		PATHS
			"C:/local/Assimp/assimp-3.2/lib32/Debug/"
			"/opt/local/lib"
			"/usr/local/lib")
	set(Assimp_LIBRARY_RELEASE ${Assimp_LIBRARY})
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
#64-bit, assume x86_64
	find_library(Assimp_LIBRARY
		NAMES
			assimp
		PATHS
			"C:/local/Assimp/assimp-3.0.0/lib64"
			"C:/Program Files/Assimp/lib/x64"
			"/opt/local/lib"
			"/usr/local/lib")
endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

find_path(Assimp_INCLUDE_DIR
	NAMES
		assimp/Importer.hpp
		assimp/Exporter.hpp
	PATHS
		"C:/local/Assimp/assimp-3.0.0/include"
		"C:/Program Files/Assimp/include"
		"/opt/local/include"
		"/usr/local/include"
	)

set(ASSIMP_INCLUDE_DIR ${Assimp_INCLUDE_DIR})
set(ASSIMP_INCLUDE_DIRS ${Assimp_INCLUDE_DIR})
set(Assimp_INCLUDE_DIRS ${Assimp_INCLUDE_DIR})
set(ASSIMP_LIBRARY ${Assimp_LIBRARY})
set(Assimp_LIBRARIES ${Assimp_LIBRARY})
set(ASSIMP_LIBRARIES ${Assimp_LIBRARY})
