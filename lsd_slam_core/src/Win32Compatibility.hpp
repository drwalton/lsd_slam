#ifndef WIN32_COMPATIBILITY_HPP_INCLUDED
#define WIN32_COMPATIBILITY_HPP_INCLUDED

#include <string>

std::string pathToForwardSlashes(const std::string &p);

#ifdef _WIN32
#define NOMINMAX
#include <GL/glew.h>

#include <g2o/stuff/timeutil.h>

#define snprintf _snprintf_s

void usleep(__int64 usec);

#endif

#endif
