#ifndef __BACKEND_INTERFACE_H
#define __BACKEND_INTERFACE_H

#include "Common.h"

// Forward declarations
class BackendComm;
class LocalBackend;

#ifdef _WIN32
// Since std::string is a C++ type and we are exporting C-type declarations,
// we disable the "C++ type in C declaration" warning (so, so hacky)
#pragma warning(push)
#pragma warning(disable: 4190)

#ifdef __cplusplus    // If used by C++ code, 
extern "C" {          // we need to export the C interface
#endif
#endif

	EXPORTED_BE BackendComm *CreateBackendComm();
	EXPORTED_BE LocalBackend *CreateLocalBackend();

#ifdef _WIN32
#ifdef __cplusplus
}
#endif

#pragma warning(pop)
#endif

#ifndef NO_GPU_BACKEND
#ifdef _WIN32
#include <windows.h> // For LoadLibrary
#pragma comment(lib, "user32.lib")
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym 
#endif
#endif

/*
inline void load_gpu_backend(void *&ptr)
{
#ifndef NO_GPU_BACKEND
#ifdef _WIN32
	ptr = LoadLibraryA("GPUBackend.DLL");
#else
	// Security breach? Others may change this environment variable and load other stuff
	if(getenv("GPUBACKEND"))
		ptr = dlopen(getenv("GPUBACKEND"), RTLD_LAZY);
	else
		ptr = dlopen("./Backend/libgpubackend.so", RTLD_LAZY);
#endif
#else
	ptr = NULL;
#endif
}


inline void load_gpu_backendW(void *&ptr)
{
#ifndef NO_GPU_BACKEND
#ifdef _WIN32
	ptr = LoadLibraryW(L"GPUBackend.DLL");
#else
	// Security breach? Others may change this environment variable and load other stuff
	if(getenv("GPUBACKEND"))
		ptr = dlopen(getenv("GPUBACKEND"), RTLD_LAZY);
	else
		ptr = dlopen("./Backend/libgpubackend.so", RTLD_LAZY);
#endif
#else
	ptr = NULL;
#endif
}
*/

#endif
