#include <cstdio>

#include <cuda_runtime.h>

#include <mathfuncs.h>

#include "CommonCUDA.cuh"
#include "GPUInterface.h"
#include "GPUDirectCalc.cuh"

// Takes 11 microseconds
__global__ void SphereAmplitudeKernel(const float2 *layers, unsigned int nLayers, 
									  const float *qPoints,
									  unsigned int numQ,
									  double2 *outAmp)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 

	if(x >= numQ)
		return;

	double q = qPoints[x];

	double res = 0.0f;
	float2 layer;
	float2 tempLayer;

	// Count the number of electrons (lim(sinc(x -> 0)))
	if(q < 1e-10) 
	{
		float previousRad3 = 0.0f;
		float currentRad3 = 0.0f;
		for(int i = 0; i < nLayers; i++) 
		{
			layer = layers[i];
			currentRad3 = (layer.x * layer.x * layer.x);
			res += layer.y * (4.0f / 3.0f) * PI * (currentRad3 - previousRad3);
			previousRad3 = currentRad3;
		}
	}
	else
	{
		tempLayer = layers[0];

		for(int i = 1; i < nLayers; i++)
		{
			layer = layers[i];
			res += layer.y * (-cos(q * layer.x) * layer.x + cos(q * tempLayer.x) * tempLayer.x + (sin(q * layer.x) / q)  - (sin(q * tempLayer.x) / q));
			tempLayer = layer;
		}

		res *= 4.0 * PI / (q * q);
	}

	outAmp[x] = make_double2(res, 0.0);
}

// Copies the vector to the workspace as a matrix
__global__ void SphereWorkspaceTransform(const double2 *tmpWork, double2 *work, 
										 unsigned int numAngles, unsigned int numQ, 
										 unsigned int pitch)
{
	int x =  blockIdx.x * blockDim.x + threadIdx.x; 
	int y =  blockIdx.y * blockDim.y + threadIdx.y; 
	
	if(x >= numAngles || y >= numQ)
		return;

	double2* pElement = (double2*)((char*)work + y * pitch) + x;
	
	*pElement = tmpWork[y];
}

bool GPUDirect_SetSphereParams(Workspace& work, const float2 *params, int numLayers)
{
	cudaError_t err = cudaSuccess;

	// Set device
	CHKERR(cudaSetDevice(work.gpuID));

	// TODO: Later move to workspace
	float2 *d_paramvec;
	double2 *d_tmpWork;
	CHKERR(cudaMalloc(&d_paramvec, sizeof(float2) * numLayers));
	CHKERR(cudaMalloc(&d_tmpWork, sizeof(double2) * work.numQ));

	// R[0] is the solvent, R[i] is the cumulative sum of R
	// ED[i] is the actual rho (not delta rho)
	// TODO UGLY FIX DO NOT KEEP
	float2 *NCparams = (float2 *)params;
	float solventED = params[0].y;
	for(int i = 0; i < numLayers; i++)
		NCparams[i].y -= solventED;


	CHKERR(cudaMemcpyAsync(d_paramvec, params, sizeof(float2) * numLayers, cudaMemcpyHostToDevice));

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = 1;
	dimGrid.x = ((work.numQ % BLOCK_WIDTH == 0) ? (work.numQ / BLOCK_WIDTH) : 
					(work.numQ / BLOCK_WIDTH + 1));
	dimGrid.y = 1;
	
	TRACE_KERNEL("SphereAmplitudeKernel");
	SphereAmplitudeKernel<<<dimGrid, dimBlock>>>(d_paramvec, numLayers, work.d_qPoints, work.numQ,
												 d_tmpWork);
	
	dim3 dimTGrid, dimTBlock;
	dimTBlock.x = BLOCK_WIDTH;
	dimTBlock.y = BLOCK_HEIGHT;
	dimTGrid.x = ((work.numAngles % BLOCK_WIDTH == 0) ? (work.numAngles / BLOCK_WIDTH) : 
					(work.numAngles / BLOCK_WIDTH + 1));
	dimTGrid.y = ((work.numQ % BLOCK_HEIGHT == 0) ? (work.numQ / BLOCK_HEIGHT) : 
					(work.numQ / BLOCK_HEIGHT + 1));

	TRACE_KERNEL("SphereWorkspaceTransform");
	SphereWorkspaceTransform<<<dimTGrid, dimTBlock>>>(d_tmpWork, work.d_work, 
													  work.numAngles, work.numQ, work.workPitch);
													  

	CHKERR(cudaDeviceSynchronize());

	CHKERR(cudaFree(d_tmpWork));
	CHKERR(cudaFree(d_paramvec));

	return (err == cudaSuccess);
}

