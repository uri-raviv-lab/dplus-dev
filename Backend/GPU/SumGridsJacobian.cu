// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <assert.h>

#include <random>
#include "Common.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cuComplex.h>

#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0 ) // Turn off all warnings (bypasses #pragma warning(default : X )
#endif
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif

#include <typeinfo>  //for 'typeid'

#include "CommonJacobGPUMethods.cu"


// Inner translation and rotation for memory set 1
__device__ __constant__ double _GROTMATA_ [9];
__device__ __constant__ double _GTRANSVA_ [3];

// Inner translation and rotation for memory set 2
__device__ __constant__ double _GROTMATB_ [9];
__device__ __constant__ double _GTRANSVB_ [3];

template <typename fType, typename dcfType /* complex data type */, typename cfType, int memorySetID>
__global__ void SumGridsKernel(const int offset, const fType stepSize,
								const int thetaDivs, const int phiDivs,
								dcfType *ds, fType *inAmp,
								fType *sumAmp, const long long voxels
								) {
	;
/*
#if memorySetID == 2
	#define constMat	_GROTMATB_
	#define constTrans	_GTRANSVB_
#elif memorySetID == 1
	#define constMat	_GROTMATA_
	#define constTrans	_GTRANSVA_
#else
	#error "The fourth template of SumGridsKernel must be either 1 or 2"
#endif

*/

	double *constMat, *constTrans;
	switch(memorySetID) {
	case 2:
		constMat	= &_GROTMATB_[0];
		constTrans	= &_GTRANSVB_[0];
		break;
	default:
	case 1:
		constMat	= &_GROTMATA_[0];
		constTrans	= &_GTRANSVA_[0];
		break;
	}

	long long id = (blockIdx.x * blockDim.x + threadIdx.x) + offset;
#ifndef ADDITIONAL_ALLOC
	if(id >= voxels)
		return;
#endif

	fType q, qx, qy, qz;

	fType newTheta, newPhi;
		
	long long bot;
	int lqi;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &lqi, &bot);

	cfType res;
	res.x = 0.0;
	res.y = 0.0;

	bot++;

	// Rotate the vector by the inner objects rotations
 	fType tmpx = qx, tmpy = qy, tmpz = qz;
  	qx = constMat[0] * tmpx + constMat[3] * tmpy + constMat[6] * tmpz;
  	qy = constMat[1] * tmpx + constMat[4] * tmpy + constMat[7] * tmpz;
  	qz = constMat[2] * tmpx + constMat[5] * tmpy + constMat[8] * tmpz;
	
	// Convert to polar
	if(fabs(qz) > q) q = fabs(qz);
	newTheta = acos(qz / q);
	newPhi	 = atan2(qy, qx);

	if(newPhi < 0.0)
 		newPhi += M_2PI;

	if(id == 0) {
		res = ((cfType*)inAmp)[0];
	} else {
  		res = GetAmpAtPointInPlaneJacob<fType, dcfType, cfType>(
			lqi, newTheta, newPhi, thetaDivs, phiDivs,
  							inAmp+2*bot, ds+bot);
	}

	fType snn, csn;
	// Translate the inner object
	sincos(tmpx * constTrans[0] + tmpy * constTrans[1] + tmpz * constTrans[2], &snn, &csn);
	
	id *= 2;
	
	sumAmp[id  ] += res.x * csn - res.y * snn;
	sumAmp[id+1] += res.y * csn + res.x * snn;

}

#define FREE_GPUSumGridsJacobSphrTempl_MEMORY \
	cudaFree(d_data1);			\
	cudaFree(d_data2);			\
	cudaFree(d_ds1);			\
	cudaFree(d_ds2);			\
	cudaFree(d_sum);			
/*	cudaFree(h_pinned_data1);	\
	cudaFree(h_pinned_data2);	\*/

				   
template <typename FLOAT_TYPE, typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
int GPUSumGridsJacobSphrTempl(long long voxels, int thDivs, int phDivs, FLOAT_TYPE stepSz,
								RES_FLOAT_TYPE **inAmpData,  DCF_FLOAT_TYPE **inD, FLOAT_TYPE *trans, FLOAT_TYPE *rots,
								int numGrids, RES_FLOAT_TYPE *outAmpData, 
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	typedef typename std::conditional<sizeof(RES_FLOAT_TYPE) == sizeof(double), double2, float2>::type RES_CF_TYPE;

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const unsigned int N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	cudaStream_t kernelStream, memoryStream;
	CHKERR(cudaStreamCreate(&kernelStream));
	CHKERR(cudaStreamCreate(&memoryStream));

	// Memory arrays
	//RES_FLOAT_TYPE *h_pinned_data1, *h_pinned_data2;
	RES_FLOAT_TYPE *d_data1, *d_data2, *d_sum;
	DCF_FLOAT_TYPE *d_ds1, *d_ds2;

	CHKERR( cudaMalloc(&d_sum, sizeof(RES_FLOAT_TYPE) * 2 * voxels) );
	CHKERR( cudaMalloc(&d_data1, sizeof(RES_FLOAT_TYPE) * 2 * voxels) );
	CHKERR( cudaMalloc(&d_data2, sizeof(RES_FLOAT_TYPE) * 2 * voxels) );
	CHKERR( cudaMalloc(&d_ds1, sizeof(DCF_FLOAT_TYPE) * voxels) );
	CHKERR( cudaMalloc(&d_ds2, sizeof(DCF_FLOAT_TYPE) * voxels) );

	err = cudaPeekAtLastError();
	if ( cudaSuccess != err ) {
		FREE_GPUSumGridsJacobSphrTempl_MEMORY;
		printf("Error allocating required device memory to sum grids.\n");
		return -err;
	}

	CHKERR(cudaMemset(d_sum, 0, sizeof(RES_FLOAT_TYPE) * 2 * voxels));

	CHKERR(cudaMemcpy(d_data1,	inAmpData[0],	sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(d_ds1,	inD[0],			sizeof(DCF_FLOAT_TYPE) * voxels,		cudaMemcpyHostToDevice));

	FLOAT_TYPE snA, snB, snC, csA, csB, csC;
	snA = sin(rots[0]); snB = sin(rots[1]); snC = sin(rots[2]);
	csA = cos(rots[0]); csB = cos(rots[1]); csC = cos(rots[2]);
	FLOAT_TYPE rotmat[9];
	rotmat[0] = csB * csC;				rotmat[1] = -csB*snC;				rotmat[2] = snB;
	rotmat[3] = csC*snA*snB+csA*snC;	rotmat[4] = csA*csC-snA*snB*snC;	rotmat[5] = -csB*snA;
	rotmat[6] = snA*snC-csA*csC*snB;	rotmat[7] = csC*snA+csA*snB*snC;	rotmat[8] = csA*csB;

	CHKERR(cudaMemcpyToSymbol(_GROTMATA_,	&rotmat[0],		sizeof(FLOAT_TYPE) * 9));
	CHKERR(cudaMemcpyToSymbol(_GTRANSVA_,	&trans[0],	sizeof(FLOAT_TYPE) * 3));
	
	endMem = clock();

	bool flip = false;
	int ind = 0;
	do {
		////////////////////////////////////////////////////////////////
		// Run the kernel
/*
		TRACE_KERNEL("SumGridsKernel");
		SumGridsKernel<RES_FLOAT_TYPE, DCF_FLOAT_TYPE, RES_CF_TYPE, (flip ? 2 : 1)><<< grid, threads, 0, kernelStream >>>
				(0, stepSz, thDivs, phDivs, (flip ? d_ds2 : d_ds1), (flip ? d_data2 : d_data1), d_sum);
*/
		if (flip)
		{
			TRACE_KERNEL("SumGridsKernel");
			SumGridsKernel<RES_FLOAT_TYPE, DCF_FLOAT_TYPE, RES_CF_TYPE, 2> << < grid, threads, 0, kernelStream >> >
				(0, stepSz, thDivs, phDivs, d_ds2, d_data2, d_sum, voxels);
		}
		else
		{
			TRACE_KERNEL("SumGridsKernel");
			SumGridsKernel<RES_FLOAT_TYPE, DCF_FLOAT_TYPE, RES_CF_TYPE, 1> << < grid, threads, 0, kernelStream >> >
				(0, stepSz, thDivs, phDivs, d_ds1, d_data1, d_sum, voxels);
		}

		ind++;
		flip = !flip;

		// Copy data asynchrounously
		if(ind < numGrids) {
			CHKERR(cudaMemcpyAsync((flip ? d_data2 : d_data1),	inAmpData[ind],	sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice, memoryStream));
			CHKERR(cudaMemcpyAsync((flip ? d_ds2 : d_ds1),		inD[ind],		sizeof(DCF_FLOAT_TYPE) * voxels,		cudaMemcpyHostToDevice, memoryStream));
			// Grid rotation
			snA = sin(rots[3*ind+0]); snB = sin(rots[3*ind+1]); snC = sin(rots[3*ind+2]);
			csA = cos(rots[3*ind+0]); csB = cos(rots[3*ind+1]); csC = cos(rots[3*ind+2]);
			rotmat[0] = csB * csC;				rotmat[1] = -csB*snC;				rotmat[2] = snB;
			rotmat[3] = csC*snA*snB+csA*snC;	rotmat[4] = csA*csC-snA*snB*snC;	rotmat[5] = -csB*snA;
			rotmat[6] = snA*snC-csA*csC*snB;	rotmat[7] = csC*snA+csA*snB*snC;	rotmat[8] = csA*csB;

			if(flip) {
				CHKERR(cudaMemcpyToSymbolAsync(_GROTMATB_,	&rotmat[0],		sizeof(FLOAT_TYPE) * 9, 0, cudaMemcpyHostToDevice, memoryStream));
				CHKERR(cudaMemcpyToSymbolAsync(_GTRANSVB_,	&trans[3*ind],	sizeof(FLOAT_TYPE) * 3, 0, cudaMemcpyHostToDevice, memoryStream));
			} else {
				CHKERR(cudaMemcpyToSymbolAsync(_GROTMATA_,	&rotmat[0],		sizeof(FLOAT_TYPE) * 9, 0, cudaMemcpyHostToDevice, memoryStream));
				CHKERR(cudaMemcpyToSymbolAsync(_GTRANSVA_,	&trans[3*ind],	sizeof(FLOAT_TYPE) * 3, 0, cudaMemcpyHostToDevice, memoryStream));
			}

		}

		// sync
		cudaDeviceSynchronize();

	} while(ind < numGrids);
	endKern = clock();

	CHKERR(cudaMemcpy(outAmpData, d_sum, sizeof(RES_FLOAT_TYPE) * 2 * voxels, cudaMemcpyDeviceToHost));
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);

	FREE_GPUSumGridsJacobSphrTempl_MEMORY;

	endAll = clock();

	printf("Sum grids CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
		double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);
	
	if(progfunc && progargs)
		progfunc(progargs, progmax);

	if ( cudaSuccess != err )
		return -1000 - (int)err;

	return 0;

}

///////////////////////////////////////////////////////////////////////////////
// Exposed functions that call the internal templated function
// float, float
/* Disable single precision
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, float stepSz,
								float **inAmpData,  float **inD, float *trans, float *rots,
								int numGrids, float *outAmpData, 
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	return GPUSumGridsJacobSphrTempl <float, float, float2>
		(voxels, thDivs, phDivs, stepSz, inAmpData, (float2**)inD, trans, rots,
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);
}
*/
// double, double
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz,
								double **inAmpData,  double **inD, double *trans, double *rots,
								int numGrids, double *outAmpData, 
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	return GPUSumGridsJacobSphrTempl <double, double, double2>
		(voxels, thDivs, phDivs, stepSz, inAmpData, (double2**)inD, trans, rots,
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);
}

// double, float
/* Disable single precision
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz,
								double **inAmpData,  float **inD, double *trans, double *rots,
								int numGrids, double *outAmpData, 
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	return GPUSumGridsJacobSphrTempl <double, double, float2>
		(voxels, thDivs, phDivs, stepSz, inAmpData, (float2**)inD, trans, rots,
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);
}
*/
