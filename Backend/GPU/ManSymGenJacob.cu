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
#include <cuComplex.h>

#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0) // Disabling 4305 (name/decoration too long)
#endif
#include <thrust/device_vector.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif
#include <typeinfo>  //for 'typeid'

#include "CommonJacobGPUMethods.cu"

__device__ __constant__ double _GROTMAT_ [9];
__device__ __constant__ double _GTRANSV_ [3];

template <typename fType>
__global__ void DummyKernel() {
	typedef typename std::conditional<sizeof(fType) == sizeof(double), cuDoubleComplex, cuFloatComplex>::type cfType;
}

template <typename fType, typename dcfType /* complex data type */, typename cfType>
__global__ void ConstructManualSymmetryJacobianKernel(const int offset, const fType stepSize,
										const int thetaDivs, const int phiDivs,
										int numOfCopies, dcfType *ds,
										fType *inAmp, fType *outAmp,
										fType *x, fType *y, fType *z,
										fType *alpha, fType *beta, fType *gamma,
										const fType scale, const long long voxels
										)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x) + offset;
#ifndef ADDITIONAL_ALLOC
	if(id >= voxels)
		return;
#endif

	fType q, qx, qy, qz;
	long long bot;
	int lqi;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &lqi, &bot);

	cfType res, tmp;
	res.x = 0.0;
	res.y = 0.0;

	// This is so the pointers point to the first element on the plane and not to the last 
	// element on the previous plane
	bot++;

	for(int i = 0; i < numOfCopies; i++) {
		fType newTheta, newPhi;
		fType csA, csB, csC;
		fType snA, snB, snC;
		sincos(alpha[i], &snA, &csA);
		sincos(beta [i], &snB, &csB);
		sincos(gamma[i], &snC, &csC);

		fType preRotX = qx, preRotY = qy, preRotZ = qz;
 		
// 		// Rotate the vector by the Manual Symmetry rotations first
 		fType rMat[9];	// This is a prime target for shared memory...
 		rMat[0] = csB * csC;			rMat[1] = -csB*snC;				rMat[2] = snB;
 		rMat[3] = csC*snA*snB+csA*snC;	rMat[4] = csA*csC-snA*snB*snC;	rMat[5] = -csB*snA;
 		rMat[6] = snA*snC-csA*csC*snB;	rMat[7] = csC*snA+csA*snB*snC;	rMat[8] = csA*csB;

  		qx = rMat[0] * preRotX + rMat[3] * preRotY + rMat[6] * preRotZ;
  		qy = rMat[1] * preRotX + rMat[4] * preRotY + rMat[7] * preRotZ;
  		qz = rMat[2] * preRotX + rMat[5] * preRotY + rMat[8] * preRotZ;

		// Rotate the vector by the inner objects rotations last
 		fType tmpx = qx, tmpy = qy, tmpz = qz;
  	 	qx = _GROTMAT_[0] * tmpx + _GROTMAT_[3] * tmpy + _GROTMAT_[6] * tmpz;
  	 	qy = _GROTMAT_[1] * tmpx + _GROTMAT_[4] * tmpy + _GROTMAT_[7] * tmpz;
  	 	qz = _GROTMAT_[2] * tmpx + _GROTMAT_[5] * tmpy + _GROTMAT_[8] * tmpz;

		if(fabs(qz) > q) q = fabs(qz);
		// Convert to polar
		newTheta = acos(qz / q);
		newPhi	 = atan2(qy, qx);

		qx = preRotX; qy = preRotY; qz = preRotZ;

		if(newPhi < 0.0)
 			newPhi += M_2PI;

		cfType tmpRes;
		if(id == 0) {
			tmpRes = ((cfType*)inAmp)[0];
		} else {
  			tmpRes = GetAmpAtPointInPlaneJacob<fType, dcfType, cfType>(
				lqi, newTheta, newPhi, thetaDivs, phiDivs,
  								inAmp+2*bot, ds+bot);
		}

		sincos(tmpx * _GTRANSV_[0] + tmpy * _GTRANSV_[1] + tmpz * _GTRANSV_[2], &snA, &csA);

 		tmp.x = tmpRes.x * csA - tmpRes.y * snA;
 		tmp.y = tmpRes.y * csA + tmpRes.x * snA;

 		sincos(preRotX * x[i] + preRotY * y[i] + preRotZ * z[i], &snA, &csA);
 		res.x += tmp.x * csA - tmp.y * snA;
 		res.y += tmp.y * csA + tmp.x * snA;

	}

	outAmp[id * 2    ] += res.x * scale;
	outAmp[id * 2 + 1] += res.y * scale;

}

template <typename FLOAT_TYPE, typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
int GPUCalcManSymJacobSphrTempl(long long voxels, int thDivs, int phDivs, int numCopies, FLOAT_TYPE stepSz,
								RES_FLOAT_TYPE *inAmpData, RES_FLOAT_TYPE *outData, DCF_FLOAT_TYPE *inD,
								FLOAT_TYPE *mx, FLOAT_TYPE *my, FLOAT_TYPE *mz,
								FLOAT_TYPE *ma, FLOAT_TYPE *mb, FLOAT_TYPE *mc, FLOAT_TYPE scale,
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	typedef typename std::conditional<sizeof(RES_FLOAT_TYPE) == sizeof(double), double2, float2>::type RES_CF_TYPE;

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const u64 N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	////////////////////////////
	// Device arrays
	RES_FLOAT_TYPE *dOutAmp, *dInAmp;
	DCF_FLOAT_TYPE *dDs;
    FLOAT_TYPE *dLocX, *dLocY, *dLocZ;
    FLOAT_TYPE *dLocA, *dLocB, *dLocC;

	////////////////////////////////////////////////////////////////
	// Mallocs
	CHKERR(cudaMalloc(&dOutAmp,	sizeof(RES_FLOAT_TYPE) * 2 * voxels));
	CHKERR(cudaMalloc(&dInAmp,	sizeof(RES_FLOAT_TYPE) * 2 * voxels));
	CHKERR(cudaMalloc(&dDs,		sizeof(DCF_FLOAT_TYPE) * voxels));
	// Consider moving to shared memory
	CHKERR(cudaMalloc(&dLocX,		sizeof(FLOAT_TYPE) * numCopies));
	CHKERR(cudaMalloc(&dLocY,		sizeof(FLOAT_TYPE) * numCopies));
	CHKERR(cudaMalloc(&dLocZ,		sizeof(FLOAT_TYPE) * numCopies));
	CHKERR(cudaMalloc(&dLocA,		sizeof(FLOAT_TYPE) * numCopies));
	CHKERR(cudaMalloc(&dLocB,		sizeof(FLOAT_TYPE) * numCopies));
	CHKERR(cudaMalloc(&dLocC,		sizeof(FLOAT_TYPE) * numCopies));

	if ( cudaSuccess != cudaPeekAtLastError() ) {
		printf("Error allocating memory on device: %d", (int)err);
		cudaFree(dOutAmp);
		cudaFree(dInAmp);
		cudaFree(dDs);
		cudaFree(dLocX);
		cudaFree(dLocY);
		cudaFree(dLocZ);
		cudaFree(dLocA);
		cudaFree(dLocB);
		cudaFree(dLocC);
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Memcpys
	FLOAT_TYPE snA, snB, snC, csA, csB, csC;
	snA = sin(ma[0]); snB = sin(mb[0]); snC = sin(mc[0]);
	csA = cos(ma[0]); csB = cos(mb[0]); csC = cos(mc[0]);
	FLOAT_TYPE rotmat[9], trVec[3];
	rotmat[0] = csB * csC;				rotmat[1] = -csB*snC;				rotmat[2] = snB;
	rotmat[3] = csC*snA*snB+csA*snC;	rotmat[4] = csA*csC-snA*snB*snC;	rotmat[5] = -csB*snA;
	rotmat[6] = snA*snC-csA*csC*snB;	rotmat[7] = csC*snA+csA*snB*snC;	rotmat[8] = csA*csB;

	trVec[0] = mx[0];	trVec[1] = my[0];	trVec[2] = mz[0];

	CHKERR(cudaMemcpyToSymbol(_GROTMAT_,	&rotmat[0],	sizeof(FLOAT_TYPE) * 9));
	CHKERR(cudaMemcpyToSymbol(_GTRANSV_,	&trVec[0],	sizeof(FLOAT_TYPE) * 3));

	CHKERR(cudaMemcpy(dOutAmp,		outData,		sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dInAmp,		inAmpData,		sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dDs,			inD,			sizeof(DCF_FLOAT_TYPE) * voxels,		cudaMemcpyHostToDevice));

	// The +1 is to skip the translation/rotation of the basic object (that's in constant memory)
	CHKERR(cudaMemcpy(dLocX,		mx+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dLocY,		my+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dLocZ,		mz+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dLocA,		ma+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dLocB,		mb+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dLocC,		mc+1,			sizeof(FLOAT_TYPE) * numCopies,			cudaMemcpyHostToDevice));

	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dOutAmp);
		cudaFree(dInAmp);
		cudaFree(dDs);
		cudaFree(dLocX);
		cudaFree(dLocY);
		cudaFree(dLocA);
		cudaFree(dLocB);
		cudaFree(dLocC);
		cudaFree(dLocZ);
		return -3000 - err;
	}

	endMem = clock();

	////////////////////////////////////////////////////////////////
	// Run the kernel
	TRACE_KERNEL("ConstructManualSymmetryJacobianKernel");
	ConstructManualSymmetryJacobianKernel<FLOAT_TYPE, DCF_FLOAT_TYPE, RES_CF_TYPE><<< grid, threads >>>
		(0, stepSz, thDivs, phDivs, numCopies, dDs, dInAmp, dOutAmp, dLocX, dLocY, dLocZ,
		 dLocA, dLocB, dLocC, scale, voxels);

	// Check for launch errors
	err = cudaPeekAtLastError();
	if ( cudaSuccess != err ) {
		printf("Launch error: %d", (int)err);
		cudaFree(dOutAmp);
		cudaFree(dInAmp);
		cudaFree(dDs);
		cudaFree(dLocX);
		cudaFree(dLocY);
		cudaFree(dLocA);
		cudaFree(dLocB);
		cudaFree(dLocC);
		cudaFree(dLocZ);
		return -3000 - err;
	}

	cudaThreadSynchronize();
	err = cudaPeekAtLastError();
	if ( cudaSuccess != err )
		printf("Error in kernel: %d\n", err);

	endKern = clock();

	CHKERR(cudaMemcpy(outData, dOutAmp, sizeof(FLOAT_TYPE) * 2 * voxels, cudaMemcpyDeviceToHost));
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);
	cudaFree(dOutAmp);
	cudaFree(dInAmp);
	cudaFree(dDs);
	cudaFree(dLocX);
	cudaFree(dLocY);
	cudaFree(dLocA);
	cudaFree(dLocB);
	cudaFree(dLocC);
	cudaFree(dLocZ);

	endAll = clock();

	printf("Manual Symmetry CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
		double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);
	
	if(progfunc && progargs)
		progfunc(progargs, progmin + (progmax - progmin));

	if ( cudaSuccess != err )
		return -1000 - (int)err;

	cudaDeviceReset();	// For profiling

	return 0;
}


///////////////////////////////////////////////////////////////////////////////
// Exposed functions that call the internal templated function

 #define EXPOSE_GPUCalcManSymJacobSphr_MACRO(T1, T2)												\
	int GPUCalcManSymJacobSphr																		\
		(long long voxels, int thDivs, int phDivs, int numCopies, T1 stepSize, T1 *inModel,			\
 		T1 *outData, T2 *ds, T1 *locX, T1 *locY, T1 *locZ, T1 *locA, T1 *locB, T1 *locC, T2 scale,	\
 		progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)			\
{																									\
	return GPUCalcManSymJacobSphrTempl<T1, T1, T2##2>												\
				(voxels, thDivs, phDivs, numCopies, stepSize, inModel, outData, (T2##2*)ds, locX,	\
				locY, locZ, locA, locB, locC, scale, progfunc, progargs, progmin, progmax, pStop);	\
}

EXPOSE_GPUCalcManSymJacobSphr_MACRO(double, double)
/* Disable single precision
EXPOSE_GPUCalcManSymJacobSphr_MACRO(double,	 float)
EXPOSE_GPUCalcManSymJacobSphr_MACRO(float,   float)
*/
#undef EXPOSE_GPUCalcManSymJacobSphr_MACRO
