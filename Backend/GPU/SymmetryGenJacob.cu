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

#include <thrust/device_vector.h>

#include <typeinfo>  //for 'typeid'

#include "CommonJacobGPUMethods.cu"


// Inner translation and rotation
__device__ __constant__ double _GROTMAT_ [9];
__device__ __constant__ double _GTRANSV_ [3];

// Unit vectors and repititions
__device__ __constant__ double _GUNITVECS_ [9];
__device__ __constant__ double _GUNITREPS_ [3];


template <typename fType, typename iType, typename dcfType, typename cfType>
__global__ void CalcSpaceFillingKernelJcb(const int offset, fType stepSize, fType scale,
										   const iType thetaDivs, const iType phiDivs,
										   dcfType *ds,
										   fType *inAmp, fType *outAmp,
										   const long long voxels)
{
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

	fType dotA = qx * _GUNITVECS_[0] + qy * _GUNITVECS_[1] + qz * _GUNITVECS_[2];
 	fType dotB = qx * _GUNITVECS_[3] + qy * _GUNITVECS_[4] + qz * _GUNITVECS_[5];
 	fType dotC = qx * _GUNITVECS_[6] + qy * _GUNITVECS_[7] + qz * _GUNITVECS_[8];

	// Rotate the vector by the inner objects rotations
 	fType tmpx = qx, tmpy = qy, tmpz = qz;
  	qx = _GROTMAT_[0] * tmpx + _GROTMAT_[3] * tmpy + _GROTMAT_[6] * tmpz;
  	qy = _GROTMAT_[1] * tmpx + _GROTMAT_[4] * tmpy + _GROTMAT_[7] * tmpz;
  	qz = _GROTMAT_[2] * tmpx + _GROTMAT_[5] * tmpy + _GROTMAT_[8] * tmpz;
	
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

	fType snh, csch, snn, csn;

	if(closeToZero(dotA) || _GUNITREPS_[0] == 1.0) {
		res.x *= _GUNITREPS_[0];
		res.y *= _GUNITREPS_[0];
	} else {
		fType oAmpR = res.x, oAmpI = res.y;
		snh = sin(_GUNITREPS_[0] * dotA / 2.0);
		csch = 1.0 / sin(dotA / 2.0);
		
		sincos(dotA * (1.0 - _GUNITREPS_[0]) / 2.0, &snn, &csn);
		
		res.x = csch * snh * (oAmpR * csn + oAmpI * snn);	// Real
		res.y = csch * snh * (oAmpI * csn - oAmpR * snn);	// Imag
	}

	if(closeToZero(dotB) || _GUNITREPS_[1] == 1.0) {
		res.x *= _GUNITREPS_[1];
		res.y *= _GUNITREPS_[1];
	} else {
		fType oAmpR = res.x, oAmpI = res.y;
		snh = sin(_GUNITREPS_[1] * dotB / 2.0);
		csch = 1.0 / sin(dotB / 2.0);
		sincos(dotB * (1.0 - _GUNITREPS_[1]) / 2.0, &snn, &csn);
		
		res.x = csch * snh * (oAmpR * csn + oAmpI * snn);	// Real
		res.y = csch * snh * (oAmpI * csn - oAmpR * snn);	// Imag
	}

	if(closeToZero(dotC) || _GUNITREPS_[2] == 1.0) {
		res.x *= _GUNITREPS_[2];
		res.y *= _GUNITREPS_[2];
	} else {
		fType oAmpR = res.x, oAmpI = res.y;
		snh = sin(_GUNITREPS_[2] * dotC / 2.0);
		csch = 1.0 / sin(dotC / 2.0);
		sincos(dotC * (1.0 - _GUNITREPS_[2]) / 2.0, &snn, &csn);

		res.x = csch * snh * (oAmpR * csn + oAmpI * snn);	// Real
		res.y = csch * snh * (oAmpI * csn - oAmpR * snn);	// Imag
	}

	// Translate the inner object -- TODO: If moved to the begining, we can save two doubles (snn, csn) instead of tmp[x]
	sincos(tmpx * _GTRANSV_[0] + tmpy * _GTRANSV_[1] + tmpz * _GTRANSV_[2], &snn, &csn);
	
	id *= 2;
	
	outAmp[id  ] += (res.x * csn - res.y * snn) * scale;
	outAmp[id+1] += (res.y * csn + res.x * snn) * scale;

}


#define FREE_GPUCalcSpcFillSymJacobSphrTempl_MEMORY \
	cudaFree(dOutAmp);	\
	cudaFree(dInAmp);	\
	cudaFree(dDs);
				   
template <typename FLOAT_TYPE, typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
int GPUCalcSpcFillSymJacobSphrTempl(long long voxels, int thDivs, int phDivs, FLOAT_TYPE stepSz,
									RES_FLOAT_TYPE *inAmpData, RES_FLOAT_TYPE *outData, DCF_FLOAT_TYPE *inD,
									FLOAT_TYPE *vectorMatrix /*This is the three unit cell vectors*/,
									FLOAT_TYPE *repeats /*This is the repeats in the dimensions*/,
									FLOAT_TYPE *innerRots /*This is the three angles of the inner objects rotations*/,
									FLOAT_TYPE *innerTrans /*This is the translation of the inner object*/,
									FLOAT_TYPE scale,
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

	////////////////////////////
	// Device arrays
	RES_FLOAT_TYPE *dOutAmp, *dInAmp;
	DCF_FLOAT_TYPE *dDs;

	////////////////////////////////////////////////////////////////
	// Mallocs
	CHKERR(cudaMalloc(&dOutAmp,	sizeof(RES_FLOAT_TYPE) * 2 * voxels));
	CHKERR(cudaMalloc(&dInAmp,	sizeof(RES_FLOAT_TYPE) * 2 * voxels));
	CHKERR(cudaMalloc(&dDs,		sizeof(DCF_FLOAT_TYPE) * voxels));

	if ( cudaSuccess != cudaPeekAtLastError() ) {
		printf("Error allocating memory on device: %d", (int)err);
		FREE_GPUCalcSpcFillSymJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Memcpys
	FLOAT_TYPE snA, snB, snC, csA, csB, csC;
	snA = sin(innerRots[0]); snB = sin(innerRots[1]); snC = sin(innerRots[2]);
	csA = cos(innerRots[0]); csB = cos(innerRots[1]); csC = cos(innerRots[2]);
	FLOAT_TYPE rotmat[9];
	rotmat[0] = csB * csC;				rotmat[1] = -csB*snC;				rotmat[2] = snB;
	rotmat[3] = csC*snA*snB+csA*snC;	rotmat[4] = csA*csC-snA*snB*snC;	rotmat[5] = -csB*snA;
	rotmat[6] = snA*snC-csA*csC*snB;	rotmat[7] = csC*snA+csA*snB*snC;	rotmat[8] = csA*csB;

	CHKERR(cudaMemcpyToSymbol(_GROTMAT_,	&rotmat[0],		sizeof(FLOAT_TYPE) * 9));
	CHKERR(cudaMemcpyToSymbol(_GTRANSV_,	&innerTrans[0],	sizeof(FLOAT_TYPE) * 3));

	CHKERR(cudaMemcpyToSymbol(_GUNITVECS_,	&vectorMatrix[0],	sizeof(FLOAT_TYPE) * 9));
	CHKERR(cudaMemcpyToSymbol(_GUNITREPS_,	&repeats[0],		sizeof(FLOAT_TYPE) * 3));

	CHKERR(cudaMemcpy(dOutAmp,		outData,		sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dInAmp,		inAmpData,		sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dDs,			inD,			sizeof(DCF_FLOAT_TYPE) * voxels,		cudaMemcpyHostToDevice));

	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		FREE_GPUCalcSpcFillSymJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	endMem = clock();

	////////////////////////////////////////////////////////////////
	// Run the kernel
	TRACE_KERNEL("CalcSpaceFillingKernelJcb");
	CalcSpaceFillingKernelJcb<FLOAT_TYPE, int, DCF_FLOAT_TYPE, RES_CF_TYPE><<< grid, threads >>>
		(0, stepSz, scale, thDivs, phDivs, dDs, dInAmp, dOutAmp, voxels);

	err = cudaPeekAtLastError();
	if ( cudaSuccess != err ) {
		printf("Launch error: %d", (int)err);
		FREE_GPUCalcSpcFillSymJacobSphrTempl_MEMORY;
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
	
	FREE_GPUCalcSpcFillSymJacobSphrTempl_MEMORY;

	endAll = clock();

	printf("Space filling symmetry CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
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
#define EXPOSE_GPUCalcSpcFillSymJacobSphr_MACRO(T1, DT)											\
int GPUCalcSpcFillSymJacobSphr(long long voxels, int thDivs, int phDivs, T1 stepSize,			\
			T1 *inModel, T1 *outData, DT *ds,													\
			T1 *vectorMatrix /*This is the three unit cell vectors*/,							\
			T1 *repeats /*This is the repeats in the dimensions*/,								\
			T1 *innerRots /*This is the three angles of the inner objects rotations*/,			\
			T1 *innerTrans /*This is the translation of the inner object*/,						\
			T1 scale,																			\
			progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)	\
{																								\
return GPUCalcSpcFillSymJacobSphrTempl<T1, T1, DT##2>											\
		(voxels, thDivs, phDivs, stepSize, inModel, outData, (DT##2*)ds, vectorMatrix,			\
		repeats, innerRots, innerTrans, scale, progfunc, progargs, progmin, progmax, pStop);	\
}

EXPOSE_GPUCalcSpcFillSymJacobSphr_MACRO(double, double)
/* Disable single precision
EXPOSE_GPUCalcSpcFillSymJacobSphr_MACRO(double, float)
EXPOSE_GPUCalcSpcFillSymJacobSphr_MACRO(float, float)
*/
#undef EXPOSE_GPUCalcSpcFillSymJacobSphr_MACRO
