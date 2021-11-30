#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "GPUHybridCalc.cuh"
#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"

#include <cuda_runtime.h>

#include <time.h>

#define THREADXDIM 256

// TODO::OPTIMIZATION calculate the sincosf in a separate kernel and store in constant/shared memory
// TODO::OPTIMIZATION load params to shared memory
template <typename resFType>
__global__ void UniformSphereJacobAmplitudeKernel(const float stepSize, const int thetaDivs, const int phiDivs,
												  resFType *data, float2* params, int nLayers,
												  float2 *extras /* of length 1*/, const long long voxels)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x);

	float q;
	float sn, cs;
	int qlayer;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &sn, &cs, &sn, &qlayer);

	double res = 0.0;
	
	float2 curLPar = params[0];	// fparams[i] = make_float2(r[i], ED[i]);
	for(int i = 0; i < nLayers - 1; i++) {
		float2 lPar = params[i+1];
		sincosf(q * curLPar.x, &sn, &cs);
		res -= (curLPar.y - lPar.y) * (cs * q * curLPar.x - sn);
		curLPar = lPar;
	}
	curLPar = params[nLayers - 1];
	sincosf(q * curLPar.x, &sn, &cs);
	res -= (curLPar.y) * (cs * q * curLPar.x - sn);

	res *= 4.0 * M_PI / (q*q*q);
	
	// Scale and background
	curLPar = extras[0];
	res *= curLPar.x;	// Multiply by scale
	res += curLPar.y;	// Add background

	if(id >= voxels || id == 0)
		return;

	data[id].x = res;

}


bool GPUHybrid_USphereAmplitude(GridWorkspace& work)
{
		
	cudaError err = cudaSuccess;
	
	cudaStream_t calcstream = (cudaStream_t)work.computeStream;

	int voxels = int(work.totalsz/2);

	int outerLayer;
	float dummyFloat, q;
	const int maxThreadsPerBlock = THREADXDIM/*devProp.maxThreadsPerBlock / 4*/;
	const int N = (voxels / maxThreadsPerBlock) + 1;

	GetQVectorFromIndex(voxels, work.thetaDivs, work.phiDivs, work.stepSize, 
						&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);

	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);

	TRACE_KERNEL("UniformSphereJacobAmplitudeKernel");
	UniformSphereJacobAmplitudeKernel<double2>
		<<<grid, threads, 0, calcstream>>>
		(work.stepSize, work.thetaDivs, work.phiDivs, (double2*)work.d_amp,
		(float2*)work.d_params, work.nLayers, (float2*)work.d_extraPrm, voxels);

	// Allocate the memory for the interpolants
	CHKERR(cudaMalloc(&work.d_int, sizeof(double) * work.totalsz));

	CHKERR(cudaStreamSynchronize(calcstream));

	CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
		(work.thetaDivs, work.phiDivs, outerLayer - 1, (double2*)work.d_amp, (double2*)work.d_int /*,
		*(cudaStream_t*)work.memoryStream, *(cudaStream_t*)calcstream*/);

	return err == cudaSuccess;
}

bool GPUHybrid_SetUSphere(GridWorkspace& work, float2 *params, int numLayers, float* extras, int nExtras)
{
    cudaError_t err = cudaSuccess;

	cudaStream_t stream = (cudaStream_t)work.memoryStream;

	CHKERR(cudaGetLastError());

	work.nLayers = numLayers;
	work.nParams = 2 * numLayers;
	work.nExtras = nExtras;

	CHKERR(cudaSetDevice(work.gpuID));

	CHKERR(cudaMalloc(&work.d_amp,			sizeof(double) * work.totalsz));
	CHKERR(cudaMemset(work.d_amp,		 0, sizeof(double) * work.totalsz));

	CHKERR(cudaMalloc(&work.d_params,	sizeof(float2) * numLayers));
	CHKERR(cudaMalloc(&work.d_extraPrm,	sizeof(float) * nExtras));

	CHKERR(cudaMemcpyAsync(work.d_params,	params, sizeof(float2) * numLayers,	cudaMemcpyHostToDevice, stream));
	CHKERR(cudaMemcpyAsync(work.d_extraPrm,	extras, sizeof(float ) * nExtras,	cudaMemcpyHostToDevice, stream));

	double electrons = 0.0;
	for(int i = 1; i < numLayers; i++) {
		electrons += (params[i].y /*Already delta*/) * (4.0 / 3.0) * M_PI * 
			(params[i].x * params[i].x * params[i].x - params[i-1].x * params[i-1].x * params[i-1].x);
/*
		electrons += (ED(i) - ED(i-1)) * (4.0 / 3.0) * M_PI * 
			(r(i) * r(i) * r(i) - r(i-1) * r(i-1) * r(i-1));
*/
	}

	double2 origin = make_double2(electrons * extras[0] + extras[1], 0.0);
	
	CHKERR(cudaMemcpyAsync(work.d_amp, &origin, sizeof(double2), cudaMemcpyHostToDevice, stream));

	// Ensure that origin isn't removed before being used
	CHKERR(cudaStreamSynchronize(stream));

	return err == cudaSuccess;
}


