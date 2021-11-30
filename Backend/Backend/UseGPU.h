#pragma once
#ifndef _USEGPU
#define _USEGPU

#include <cuda_runtime.h>

extern bool g_useGPU;
extern bool g_useGPUAndAvailable;

/** We add this function in order to add functionality that allow the user to choose if the system should use GPU or CPU 
the default is to use GPU if it available (g_useGPU = true)
*/
extern cudaError_t UseGPUDevice(int * devCount);

#endif
