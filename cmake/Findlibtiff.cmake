find_library(libtiff_LIBRARY
	NAMES
		libtiff
	PATHS
		"C:/local/libtiff/libtiff2.8.2/lib"
		"/usr/local/lib"
		"/opt/local/lib"
)

set(libtiff_LIBRARIES ${libtiff_LIBRARY})

find_path(libtiff_INCLUDE_DIRS
	NAMES
		tiff.h
	PATHS
		"C:/local/libtiff/libtiff2.8.2/include"
		"/usr/local/include"
		"/opt/local/include"
)

set(libtiff_INCLUDE_DIR ${libtiff_INCLUDE_DIRS})

