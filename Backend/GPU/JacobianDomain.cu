#include <cstdio>
#include <cstdlib>

#include "JacobianDomain.cuh"

#include <cuda_runtime.h>

#include <vector_functions.h>

// For integration
#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0 ) // Turn off all warnings (bypasses #pragma warning(default : X )
#endif
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif


bool GPUJacobainDomainCalculator::Initialize(int gpuID, const std::vector<float>& qPoints,
	long long totalSize, int thetaDivisions, int phiDivisions, int qLayers,
		double qMax, double stepSize, GridWorkspace& res)
{
	printf("GPUJacobainDomainCalculator::Initialize not yet implemented\n");
	return false;
}

bool GPUJacobainDomainCalculator::FreeWorkspace(GridWorkspace& workspace)
{
	printf("GPUJacobainDomainCalculator::FreeWorkspace not yet implemented\n");
	return false;
}

int GPUJacobainDomainCalculator::AssembleAmplitudeGrid(GridWorkspace& workspace, double **subAmp,
		float **subInt, double **transRot, int numSubAmps)
{
	printf("GPUJacobainDomainCalculator::AssembleAmplitudeGrid not yet implemented\n");
	return -1;
}

int GPUJacobainDomainCalculator::OrientationAverageMC(GridWorkspace& workspace, long long maxIters,
						double convergence,  double *qVals, double *iValsOut)
{
	printf("GPUJacobainDomainCalculator::OrientationAverageMC not yet implemented\n");
	return -1;
}
