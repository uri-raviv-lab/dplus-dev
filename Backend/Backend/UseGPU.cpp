#include "UseGPU.h"

bool g_useGPU = true;
bool g_useGPUAndAvailable = false;
cudaError_t UseGPUDevice(int * devCount)
{
	if (g_useGPU) {
		cudaError_t res = cudaGetDeviceCount(devCount);
		if (res != cudaSuccess) *devCount = 0;
		g_useGPUAndAvailable = *devCount > 0 && res == cudaSuccess;
		return res;
	}
	*devCount = 0;
	g_useGPUAndAvailable = false;
	return cudaSuccess;
}
