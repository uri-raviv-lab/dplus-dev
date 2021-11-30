////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */


// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <assert.h>

#include "Common.h"

// CUDA runtime
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

__device__ bool closeToZero2(double x) {
	return (fabs(x) < 1.0e-9);
}

__global__ void CalcSpaceFillingKernelSphr(u64 voxels, unsigned short dimx,
											  double stepSize, double2 *data, const ulonglong2 *idx,
											  const double av0, const double av1, const double av2, 
											  const double bv0, const double bv1, const double bv2, 
											  const double cv0, const double cv1, const double cv2,
											  const double Na, const double Nb, const double Nc ) {
	int tid = threadIdx.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid);
	if(id >= voxels)
		return;
	
	double2 amp = data[id];
	
	ulonglong2 input = idx[id];
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);

//   	unsigned short thetaInd  = in1->w;
//   	unsigned short maxTheta  = in1->z;
//   	unsigned short phiIndex  = in1->y;
//   
//   	unsigned short maxPhi = in1->x;
//   	unsigned short rIndex = in2->x;

	// Can we think of a way to do this NOT cartesean?
 	double qx, qy, qz, q = double(in2->x) * stepSize;
		
	//amp.x = sqrt(q);	amp.y = 0.0;	data[id] = amp;	return;

	{
		double sinth, costh;
		sincospi(double(in1->w) / double(in1->z-1), &sinth, &costh);//, ph = 2.0 * M_PI * double(in1->y) / double(in1->x);
		double snPhi, csPhi;
		if(in1->z-1 == 0) {
			sinth = 0.0;
			costh = 1.0;
		} else {	// Dummy code to try and keep things synced
			snPhi = 0.0;
			csPhi = 1.0;
		}

		sincospi(2.0 * double(in1->y) / double(in1->x), &snPhi, &csPhi);

		qx = q * sinth * csPhi;
 		qy = q * sinth * snPhi;
 		qz = q * costh;
		if(in1->z-1 == 0)
			qz = q;
		else
			sinth = q;
	}

	double dotA = qx * av0 + qy * av1 + qz * av2;
	double dotB = qx * bv0 + qy * bv1 + qz * bv2;
	double dotC = qx * cv0 + qy * cv1 + qz * cv2;
	
	if(closeToZero2(dotA)) {
		amp.x *= Na;
		amp.y *= Na;
	} else {
		double2 oAmp = amp;
		double snh, csch, snn, csn;
		snh = sin(Na * dotA / 2.0);
		csch = 1.0 / sin(dotA / 2.0);
		
		sincos(dotA * (1.0 - Na) / 2.0, &snn, &csn);
		
		amp.x = csch * snh * (oAmp.x * csn + oAmp.y * snn);	// Real
		amp.y = csch * snh * (oAmp.y * csn - oAmp.x * snn);	// Imag
	}
	
	if(closeToZero2(dotB)) {
		amp.x *= Nb;
		amp.y *= Nb;
	} else {
		double2 oAmp = amp;
		double snh, csch, snn, csn;
		snh = sin(Nb * dotB / 2.0);
		csch = 1.0 / sin(dotB / 2.0);
		sincos(dotB * (1.0 - Nb) / 2.0, &snn, &csn);
		
		amp.x = csch * snh * (oAmp.x * csn + oAmp.y * snn);	// Real
		amp.y = csch * snh * (oAmp.y * csn - oAmp.x * snn);	// Imag
	}

	if(closeToZero2(dotC)) {
		amp.x *= Nc;
		amp.y *= Nc;
	} else {
		double2 oAmp = amp;
		double snh, csch, snn, csn;
		snh = sin(Nc * dotC / 2.0);
		csch = 1.0 / sin(dotC / 2.0);
		sincos(dotC * (1.0 - Nc) / 2.0, &snn, &csn);

		amp.x = csch * snh * (oAmp.x * csn + oAmp.y * snn);	// Real
		amp.y = csch * snh * (oAmp.y * csn - oAmp.x * snn);	// Imag
	}

	data[id] = amp;
}

int
GPUCalcSpaceSymSphr(u64 voxels, unsigned short dimx, double stepSize, double *data, const u64 *idx,
					const double av0, const double av1, const double av2, const double bv0, const double bv1,
					const double bv2, const double cv0, const double cv1, const double cv2, const double Na,
					const double Nb, const double Nc,
					progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop) {
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;

	const u64 N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	double2 *dData;
	ulonglong2 *dIdx;

	err = cudaMalloc(&dData, sizeof(double2) * voxels);
	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dData);
		return -1000 - (int)err;
	}

	err = cudaMalloc(&dIdx, sizeof(ulonglong2) * voxels);
	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dIdx);
		cudaFree(dData);		
		return -1000 - (int)err;
	}
	
	err = cudaMemcpy(dData, data, sizeof(double2) * voxels, cudaMemcpyHostToDevice);
	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dIdx);
		cudaFree(dData);
		return -1000 - (int)err;
	}

	err = cudaMemcpy(dIdx, idx, sizeof(ulonglong2) * voxels, cudaMemcpyHostToDevice);
	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dIdx);
		cudaFree(dData);
		return -1000 - (int)err;
	}
	endMem = clock();
	
	CalcSpaceFillingKernelSphr<<<grid, threads>>>(voxels, dimx, stepSize, dData, dIdx,
													av0, av1, av2, bv0, bv1, bv2,
													cv0, cv1, cv2, Na, Nb, Nc);
	// Check for launch errors
	err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		printf("Launch error on device: %d", (int)err);
		cudaFree(dData);
		cudaFree(dIdx);
		return -1000 - (int)err;
	}

	cudaThreadSynchronize();
	endKern = clock();
	// Check for execution errors
	err = cudaGetLastError();	
	if ( cudaSuccess != err ) {
		printf("Execution error on device: %d", (int)err);
		cudaFree(dData);
		cudaFree(dIdx);
		return -1000 - (int)err;
	}
	endAll = clock();

	printf("CUDA space filling timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
			double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);

	err = cudaMemcpy(data, dData, sizeof(double) * voxels * 2, cudaMemcpyDeviceToHost);
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);
	
	cudaFree(dIdx);
	cudaFree(dData);

	if ( cudaSuccess != err ) {
		printf("Error freeing memory on device: %d", (int)err);
		return -1000 - (int)err;
	}

	return 0;

}

