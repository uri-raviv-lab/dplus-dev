#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "GPUHybridCalc.cuh"
#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"

#include <cuda_runtime.h>

#include <time.h>


bool GPUHybrid_AmpGridAmplitude(GridWorkspace& work, double* amp)
{
	cudaError err = cudaSuccess;
	long long voxels = work.totalsz/2;
	int outerLayer;
	float dummyFloat, q;

	clock_t beg, endAll;

	beg = clock();

	CHKERR(cudaMalloc(&work.d_amp, sizeof(double2) * voxels));
	CHKERR(cudaMemcpy(work.d_amp, amp, sizeof(double2) * voxels, cudaMemcpyHostToDevice));
	CHKERR(cudaMalloc(&work.d_int, sizeof(double) * work.totalsz));

	endAll = clock();

	GetQVectorFromIndex(voxels, work.thetaDivs, work.phiDivs, work.stepSize, 
					&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);

	CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
		(work.thetaDivs, work.phiDivs, outerLayer - 1, (double2*)work.d_amp, (double2*)work.d_int /*,
		*(cudaStream_t*)work.memoryStream, *(cudaStream_t*)work.computeStream*/);

	printf("Hybrid ampGrid CUDA timing:\n\tKernel: %f seconds\n", double(endAll - beg)/CLOCKS_PER_SEC);

	return err == cudaSuccess;
}
