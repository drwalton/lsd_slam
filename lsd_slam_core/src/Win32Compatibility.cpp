#include "Win32Compatibility.hpp"

#ifdef _WIN32
#include <windows.h>

void usleep(__int64 usec)
{
	HANDLE timer;
	LARGE_INTEGER ft;

	ft.QuadPart = -(10 * usec); // Convert to 100 nanosecond interval, negative value indicates relative time

	timer = CreateWaitableTimer(NULL, TRUE, NULL);
	SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
	WaitForSingleObject(timer, INFINITE);
	CloseHandle(timer);
}

std::string pathToForwardSlashes(const std::string &p) 
{
	std::string p2 = p;
	for (char &c : p2) {
		if (c == '\\') {
			c = '/';
		}
	}
	return p2;
}
#else
std::string pathToForwardSlashes(const std::string &p) 
{
	return p;
}
#endif
