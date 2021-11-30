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
#include <math.h>
#include <time.h>
#include <assert.h>

#include "Common.h"

// CUDA runtime
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "CommonJacobGPUMethods.cu"	// For closeToZero

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

// __device__ bool closeToZero(double x) {
// 	return (fabs(x) < 1.0e-9);
// }


__global__ void CalcSpaceFillingKernelCart(u64 voxels, unsigned short dimx,
											  double stepSize, double2 *data, const double2 *idx,
											  const double av0, const double av1, const double av2, 
											  const double bv0, const double bv1, const double bv2, 
											  const double cv0, const double cv1, const double cv2,
											  const double Na, const double Nb, const double Nc ) {
	int tid = threadIdx.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid);
	if(id >= voxels)
		return;
	
	double2 amp = data[id];
		
	double2 input = idx[id];
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);

	const unsigned short xi   = in1->w;
	const unsigned short yi   = in1->z;
	const unsigned short zi   = in1->y;

	const unsigned short dimy = in1->x;
	const unsigned short dimz = in2->x;

 	double qx, qy, qz;
 	qx = (double(xi) - (double(dimx - 1) / 2.0)) * stepSize;
 	qy = (double(yi) - (double(dimy - 1) / 2.0)) * stepSize;
 	qz = (double(zi) - (double(dimz - 1) / 2.0)) * stepSize;

	double dotA = qx * av0 + qy * av1 + qz * av2;
	double dotB = qx * bv0 + qy * bv1 + qz * bv2;
	double dotC = qx * cv0 + qy * cv1 + qz * cv2;

	if(closeToZero(dotA)) {
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
	
	if(closeToZero(dotB)) {
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

	if(closeToZero(dotC)) {
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


__global__ void CalcAddGrids(double2 *da, double2 *db, u64 voxels) {
	int tid = threadIdx.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid);

	if(id >= voxels)
		return;
		
	double2 res = da[id];
	double2 add = db[id];

	res.x += add.x;
	res.y += add.y;

	da[id] = res;
}



int
GPUCalcSpaceSymCart(u64 voxels, unsigned short dimx, double stepSize, double *data, const double *idx,
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

	double2 *dData, *dIdx;

	err = cudaMalloc(&dData, sizeof(double2) * voxels);
	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dData);
		return -1000 - (int)err;
	}

	err = cudaMalloc(&dIdx, sizeof(double2) * voxels);
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

	err = cudaMemcpy(dIdx, idx, sizeof(double2) * voxels, cudaMemcpyHostToDevice);
	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dIdx);
		cudaFree(dData);
		return -1000 - (int)err;
	}
	endMem = clock();
	
	CalcSpaceFillingKernelCart<<<grid, threads>>>(voxels, dimx, stepSize, (double2*)dData, (double2*)dIdx,
													av0, av1, av2, bv0, bv1, bv2,
													cv0, cv1, cv2, Na, Nb, Nc);
	// Check for launch errors
	err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		cudaFree(dData);
		cudaFree(dIdx);
		return -1000 - (int)err;
	}

	cudaThreadSynchronize();
	endKern = clock();
	// Check for execution errors
	err = cudaGetLastError();	
	if ( cudaSuccess != err ) {
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
	
	cudaFree(dData);

	if ( cudaSuccess != err )
		return -1000 - (int)err;

	return 0;

}

template <typename TNAME>
int GPUAddGridsT(u64 voxels, TNAME *aData, TNAME *bData) {
////////////////////////////////////////////////////////////////////////////////
//Thrust version

	thrust::device_vector<TNAME> dTDataA;
	thrust::device_vector<TNAME> dTDataB;
	try {
		dTDataA.assign(aData, aData + 2*voxels);
		dTDataB.assign(bData, bData + 2*voxels);
	} catch(std::bad_alloc &e) {
		printf("Error allocating data (thrust)\n");
		return -65;
	} catch(thrust::system_error &e) {
		printf("Thrust error: %s", e.what());
		return -66;
	}

	thrust::transform(dTDataA.begin(), dTDataA.end(), 
							dTDataB.begin(), dTDataA.begin(), thrust::plus<TNAME>());

	thrust::copy(dTDataA.begin(), dTDataA.end(), aData);

	return 0;
// End thrust version
////////////////////////////////////////////////////////////////////////////////
}
int GPUAddGrids(u64 voxels, double *aData, double *bData)
{
	return GPUAddGridsT<double>(voxels, aData, bData);
        /*
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 16;

	const u64 N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	double2 *dDataA, *dDataB;

	err = cudaMalloc(&dDataA,	sizeof(double2) * voxels);
	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dDataA);
		return -1000 - (int)err;
	}

	err = cudaMalloc(&dDataB,	sizeof(double2) * voxels);

	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dDataA);
		cudaFree(dDataB);
		return -1000 - (int)err;
	}

	err = cudaMemcpy(dDataA, aData,	sizeof(double2) * voxels, cudaMemcpyHostToDevice);
	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dDataA);
		cudaFree(dDataB);
		return -1000 - (int)err;
	}
	err = cudaMemcpy(dDataB, bData,	sizeof(double2) * voxels, cudaMemcpyHostToDevice);

	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dDataA);
		cudaFree(dDataB);
		return -1000 - (int)err;
	}
	
	endMem = clock();

	CalcAddGrids<<<grid, threads>>>(dDataA, dDataB, voxels);

	// Check for launch errors
	err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		cudaFree(dDataA);
		cudaFree(dDataB);
		return -1000 - (int)err;
	}

	cudaThreadSynchronize();
	endKern = clock();
	// Check for execution errors
	err = cudaGetLastError();	
	if ( cudaSuccess != err ) {
		cudaFree(dDataA);
		cudaFree(dDataB);
		return -1000 - (int)err;
	}
	endAll = clock();

	printf("CUDA add grids timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
			double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);

	err = cudaMemcpy(aData, dDataA, sizeof(double) * voxels * 2, cudaMemcpyDeviceToHost);
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);
	
	cudaFree(dDataA);
	cudaFree(dDataB);

	if ( cudaSuccess != err )
		return -1000 - (int)err;

	return 0;
        */
}
