#include "clrun.h"
#include "dynamiclib.h"

#ifdef _WIN32
#include <windows.h>
#endif

// 0 means no opencl, 1 means opencl
int isCL = 0;

int clrInit() {
	int ret = 0;

#ifdef _WIN32
	const char *libname = "OpenCL.dll";
#else
	const char *libname = "libOpenCL.so";
#endif

	if(isCL) 
		return ret;

	if((ret = loadLib(libname))) {
		if(ret == -3) // No OpenCL
			return 0;
		else
			return ret;
	}

	isCL = 1;

	return 0;
}

int clrHasOpenCL() {
	return isCL;
}


// Windows-specific DLL code
#ifdef _WIN32
HINSTANCE g_hInstance;

BOOL APIENTRY DllMain(HINSTANCE hinstDLL,	// DLL module handle
					  DWORD dwReason,		// reason called
					  LPVOID lpvReserved)	// reserved
{
	switch (dwReason)
	{
	case DLL_PROCESS_ATTACH:
		break; 

	case DLL_PROCESS_DETACH:
		break;
	default:
		break;
	}

	g_hInstance = hinstDLL;
	return TRUE;
}
#endif
