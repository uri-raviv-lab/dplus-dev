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
#include <thrust/pair.h>
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif

#include <typeinfo>  //for 'typeid'

#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"


template <typename fType, typename dcfType /* complex data type */, typename cfType>
__global__ void MCOAJacobianKernel(const int offset, const fType stepSize,
										const int thetaDivs, const int phiDivs,
										dcfType *ds, fType *inAmp,
										const int qInd,	// The index of the lower plane
										const int m,	// The number of q-values between the planes (lower, upper]
										fType *qs,	// The list of m q-values that are being averaged
										fType *thetas, fType *phis,	// A list of random angles to integrate over
										fType *reses /*a [avePoints][m] matrix where m is the number of q-values between the two planes*/,
										const int avePoints
										) {
	;

	long long id = (blockIdx.x * blockDim.x + threadIdx.x) + offset;
#ifndef ADDITIONAL_ALLOC
	if(id >= avePoints)
		return;
#endif

	cfType amps[4], tmpAmp;
	dcfType d1, d2;

#pragma unroll 4
	for (int i = -1; i <= 2; i++) {
		long long lqi = (long long)(i + qInd - 1);	// For the base
		long long bot = (lqi * phiDivs * (lqi + 1) * (3 + thetaDivs + 2 * thetaDivs * lqi)) / 6;
		bot++;
		lqi++;	// The actual layer
		switch (lqi)
		{
		case -1:
		{
			bot = 1;
			fType new_phi = phis[id] + M_PI;
			// new_phi range - [0, 2pi]
			if (double(new_phi) < 0)
				new_phi = M_2PI + (double(new_phi) - (int(double(new_phi) / M_2PI)* M_2PI));
			if (double(new_phi) >= M_2PI)
				new_phi = double(new_phi) - (int(double(new_phi)/ M_2PI)*M_2PI);

			fType new_theta = M_PI - thetas[id];
			// new_theta range - [0, pi]
			if (double(new_theta) < 0)
				new_theta = 0.;
			if (double(new_theta) > M_PI)
				new_theta = M_PI;

			amps[1 + i] = GetAmpAtPointInPlaneJacob<fType, dcfType, cfType>(
				1, new_theta, new_phi, thetaDivs, phiDivs, inAmp + 2 * bot, ds + bot);
			break;
		}
		case 0:
			amps[1 + i] = ((cfType*)inAmp)[0];
			break;
		default:
			// GetAmpAtPointInPlaneJacob uses 41 registers
			amps[1 + i] = GetAmpAtPointInPlaneJacob<fType, dcfType, cfType>(
				lqi, thetas[id], phis[id], thetaDivs, phiDivs, inAmp + 2 * bot, ds + bot);
			break;
		}
	}

	FourPointEvenlySpacedSpline<cfType, dcfType>(amps[0], amps[1], amps[2], amps[3], &d1, &d2);	// Requires 10 registers

	fType t;
	for(int j = 0; j < m; j++) {
		t = (qs[j] - qInd * stepSize) / stepSize;	// t[m] can be in constant/shared memory

		tmpAmp.x = amps[1].x + d1.x * t +
		  (3.0 * (amps[2].x - amps[1].x) - 2.0 * d1.x - d2.x) * (t*t) + 
		  (2.0 * (amps[1].x - amps[2].x) + d1.x + d2.x) * (t*t*t);
		tmpAmp.y = amps[1].y + d1.y * t +
		  (3.0 * (amps[2].y - amps[1].y) - 2.0 * d1.y - d2.y) * (t*t) + 
		  (2.0 * (amps[1].y - amps[2].y) + d1.y + d2.y) * (t*t*t);

		reses[id + j * avePoints] = tmpAmp.x*tmpAmp.x + tmpAmp.y*tmpAmp.y;

	}

}

#define FREE_GPUCalcMCOAJacobSphrTempl_MEMORY \
	cudaFree(dInAmp);		\
	cudaFree(dqs);			\
	cudaFree(dDs);			\
	cudaFree(dPhis1);		\
	cudaFree(dPhis2);		\
	cudaFree(dInten);		\
	cudaFreeHost(hphis);	\
	cudaFreeHost(hIntenConv);\
	hIntenConv = NULL;		\
	cudaStreamDestroy(kernelStream);	\
	cudaStreamDestroy(memoryStream);
				   

template <typename FLOAT_TYPE, typename RES_FLOAT_TYPE, typename DCF_FLOAT_TYPE>
int GPUCalcMCOAJacobSphrTempl(long long voxels, int thDivs, int phDivs, FLOAT_TYPE stepSz,
								RES_FLOAT_TYPE *inAmpData,  DCF_FLOAT_TYPE *inD, 
								RES_FLOAT_TYPE *qs, RES_FLOAT_TYPE *intensities, int qPoints,
								long long maxIters, RES_FLOAT_TYPE convergence,
								progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	typedef typename std::conditional<sizeof(RES_FLOAT_TYPE) == sizeof(double), double2, float2>::type RES_CF_TYPE;

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();
#define ITERS_PER_KERNEL 1024*8

	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const u64 N = (ITERS_PER_KERNEL / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	int baseLen = (1 + int( qPoints * ( stepSz / qs[qPoints-1]) ) );
	RES_FLOAT_TYPE *hIntenConv;
	CHKERR(cudaMallocHost((void**)&hIntenConv, sizeof(RES_FLOAT_TYPE) * baseLen * ((maxIters / ITERS_PER_KERNEL) + 1) ) );
	cudaStream_t kernelStream, memoryStream;
	CHKERR(cudaStreamCreate(&kernelStream));
	CHKERR(cudaStreamCreate(&memoryStream));

	////////////////////////////
	// Device arrays
	RES_FLOAT_TYPE *dInAmp;
	RES_FLOAT_TYPE *dqs;
	DCF_FLOAT_TYPE *dDs;

	RES_FLOAT_TYPE *dThetas1;
	RES_FLOAT_TYPE *dThetas2;
	RES_FLOAT_TYPE *dPhis1;
	RES_FLOAT_TYPE *dPhis2;
	RES_FLOAT_TYPE *dInten;
		
	// The random angles to be copied to the kernel
	RES_FLOAT_TYPE *hphis, *hthetas;

	////////////////////////////////////////////////////////////////
	// Mallocs
	CHKERR(cudaMallocHost((void**)&hphis, sizeof(RES_FLOAT_TYPE) * 2 * ITERS_PER_KERNEL ) );
	//CHKERR(cudaMallocHost((void**)&hthetas, sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL ) );
	hthetas = hphis + ITERS_PER_KERNEL;

	CHKERR(cudaMalloc(&dInAmp,		sizeof(RES_FLOAT_TYPE) * 2 * voxels));
	CHKERR(cudaMalloc(&dqs,			sizeof(RES_FLOAT_TYPE) * qPoints));
	CHKERR(cudaMalloc(&dDs,			sizeof(DCF_FLOAT_TYPE) * voxels));

// 	CHKERR(cudaMalloc(&dThetas1,	sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL));
// 	CHKERR(cudaMalloc(&dThetas2,	sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL));
	CHKERR(cudaMalloc(&dPhis1,		sizeof(RES_FLOAT_TYPE) * 2 * ITERS_PER_KERNEL));
	CHKERR(cudaMalloc(&dPhis2,		sizeof(RES_FLOAT_TYPE) * 2 * ITERS_PER_KERNEL));
	dThetas1 = dPhis1 + ITERS_PER_KERNEL;
	dThetas2 = dPhis2 + ITERS_PER_KERNEL;

	CHKERR(cudaMalloc(&dInten,		sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL * baseLen ) );

	////////////////////////////////////////////////////////////////
	// Memcpys
	cudaMemset(dInten, 0, sizeof(RES_FLOAT_TYPE) * baseLen * ITERS_PER_KERNEL);
	CHKERR(cudaMemcpy(dInAmp,	inAmpData,	sizeof(RES_FLOAT_TYPE) * 2 * voxels,	cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dDs,		inD,		sizeof(DCF_FLOAT_TYPE) * voxels,		cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(dqs,		qs,			sizeof(RES_FLOAT_TYPE) * qPoints,		cudaMemcpyHostToDevice));

	if ( cudaSuccess != cudaPeekAtLastError() ) {
		printf("Error allocating memory on device: %d", (int)cudaPeekAtLastError());
		FREE_GPUCalcMCOAJacobSphrTempl_MEMORY;
		return -err;
	}

	endMem = clock();

	int qInd = 0;
	if(qs[0] == 0.0) {
		intensities[0] = inAmpData[0] * inAmpData[0] + inAmpData[1] * inAmpData[1];
		qInd++;
	}

	int lowerLayer = 0;
	//auto qtmpBegin = Q.begin();
	//while (*qtmpBegin > lowerLayer * stepSize) // Minimum q val
	//	tmpLayer++;
	std::mt19937 rng;
	rng.seed(static_cast<unsigned int>(std::time(0)));
	std::uniform_real_distribution<RES_FLOAT_TYPE> u2(0.0, 2.0);

	bool flip = false;
	// Initialize first set of random thetas and phis (MT) and copy to device
	for(int i = 0; i < ITERS_PER_KERNEL; i++) {
		hphis[i] = RES_FLOAT_TYPE(M_PI) * u2(rng);
		hthetas[i] = acos(u2(rng) - RES_FLOAT_TYPE(1));
	}
// 	CHKERR(cudaMemcpyAsync(dThetas1, hthetas, sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL, cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dPhis1, hphis, sizeof(RES_FLOAT_TYPE) * 2 * ITERS_PER_KERNEL, cudaMemcpyHostToDevice, memoryStream));
	
	do {
		// Find the points that need to be integrated
		while(qs[qInd] > double(lowerLayer+1) * stepSz) {
			lowerLayer++;
		}
		int qEnd;
		for(qEnd = qInd + 1; qEnd < qPoints; qEnd++) {
			if(qs[qEnd] > double(lowerLayer+1) * stepSz)
				break;
		}
		qEnd--;
		int len = qEnd - qInd + 1;

		// Not enough allocated memory
		if(len > baseLen) {
			baseLen = len;
			CHKERR(cudaFree(dInten));
			CHKERR(cudaMalloc(&dInten, sizeof(RES_FLOAT_TYPE) * len * ITERS_PER_KERNEL));
			cudaMemsetAsync(dInten, 0, sizeof(RES_FLOAT_TYPE) * len * ITERS_PER_KERNEL, memoryStream);
			if(hIntenConv)
				cudaFreeHost(hIntenConv);
			CHKERR(cudaMallocHost((void**)&hIntenConv, sizeof(RES_FLOAT_TYPE) * len * ((maxIters / ITERS_PER_KERNEL) + 1) ) );
			//hIntenConv = new RES_FLOAT_TYPE[len * ((maxIters / ITERS_PER_KERNEL) + 1)];
			memset(hIntenConv, 0, sizeof(RES_FLOAT_TYPE) * len * ((maxIters / ITERS_PER_KERNEL) + 1));
		}

		// Run kernel on section until it converges

		bool converged = false;
		int loopCtr = 0;
		do {
			CHKERR(cudaStreamSynchronize(memoryStream));
			// Run kernel with alternating sets of angle pointers
			TRACE_KERNEL("MCOAJacobianKernel");
			MCOAJacobianKernel<RES_FLOAT_TYPE, DCF_FLOAT_TYPE, RES_CF_TYPE><<< grid, threads, 0, kernelStream >>>
				(0, stepSz, thDivs, phDivs, dDs, dInAmp,
				lowerLayer, qEnd - qInd + 1, dqs + qInd,
				(flip ? dThetas2 : dThetas1),
				(flip ? dPhis2 : dPhis1),
				dInten, ITERS_PER_KERNEL);
			
			err = cudaPeekAtLastError();
			if ( cudaSuccess != err ) {
				printf("Launch error: %d", (int)err);
				FREE_GPUCalcMCOAJacobSphrTempl_MEMORY;
				return -3000 - err;
			}

			// Initialize random thetas and phis (MT) and copy to device
			for(int i = 0; i < ITERS_PER_KERNEL; i++) {
				hphis[i] = RES_FLOAT_TYPE(M_PI) * u2(rng);
				hthetas[i] = acos(u2(rng) - RES_FLOAT_TYPE(1));
			}

			//CHKERR(cudaMemcpyAsync( (flip ?  dThetas1 : dThetas2), hthetas, sizeof(RES_FLOAT_TYPE) * ITERS_PER_KERNEL, cudaMemcpyHostToDevice, memoryStream));
			CHKERR(cudaMemcpyAsync( (flip ?  dPhis1 : dPhis2), hphis, sizeof(RES_FLOAT_TYPE) * 2 * ITERS_PER_KERNEL, cudaMemcpyHostToDevice, memoryStream));

			CHKERR(cudaStreamSynchronize(kernelStream));
			err = cudaPeekAtLastError();
			if ( cudaSuccess != err ) {
				printf("Error in kernel: %d", (int)err);
				FREE_GPUCalcMCOAJacobSphrTempl_MEMORY;
				return -3000 - err;
			}

#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0) // Disabling 4305 (name/decoration too long)
#endif
			// Collect results
			thrust::device_vector<int> dptr_intind(len);
			thrust::device_vector<RES_FLOAT_TYPE> dptr_intensity(len);

			thrust::reduce_by_key
				(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(ITERS_PER_KERNEL)),	//keys_first
				thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(ITERS_PER_KERNEL)) + (ITERS_PER_KERNEL*len),	// keys_last
				thrust::device_ptr<RES_FLOAT_TYPE>(dInten),			// values_first
				dptr_intind.data(),	// keys_output
				dptr_intensity.data(),				// values_output
				thrust::equal_to<int>(),			// binary_pred
				thrust::plus<RES_FLOAT_TYPE>());	// binary_op

			cudaMemcpy((hIntenConv + loopCtr * len),
						thrust::raw_pointer_cast(dptr_intensity.data()), sizeof(RES_FLOAT_TYPE) * len, cudaMemcpyDeviceToHost);
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif

			if(loopCtr == 0) {
				for(int i = 0; i < len; i++) {
					hIntenConv[i] /= RES_FLOAT_TYPE(ITERS_PER_KERNEL);
				}
			} else {
				for(int i = 0; i < len; i++) {
					hIntenConv[loopCtr*len + i] = (hIntenConv[(loopCtr-1)*len + i] * RES_FLOAT_TYPE(loopCtr * ITERS_PER_KERNEL) + 
													hIntenConv[loopCtr*len + i] ) /
													RES_FLOAT_TYPE((loopCtr+1) * ITERS_PER_KERNEL);
				}
			}	// if/else
			// Convergence Place TODO FIXME
			// Check convergence
			if(loopCtr > 2 && convergence > 0.0) {
				bool tmp = true;
				for(int i = 0; i < len; i++) {
					if((fabs(1.0 - (hIntenConv[loopCtr*len + i] / hIntenConv[(loopCtr-1)*len + i]) ) > convergence) || 
						(fabs(1.0 - (hIntenConv[loopCtr*len + i] / hIntenConv[(loopCtr-2)*len + i]) ) > convergence) ||
						(fabs(1.0 - (hIntenConv[loopCtr*len + i] / hIntenConv[(loopCtr-3)*len + i]) ) > convergence)) {
						tmp = false;
						break;
					} // if
				} // for i
				converged = tmp;
			} // if loopCtr > 2

			flip = !flip;
		} while(++loopCtr * ITERS_PER_KERNEL < maxIters && !converged);
		
		// Copy the final results to the result array
		for(int i = 0; i < len; i++) {
			intensities[qInd + i] = hIntenConv[(loopCtr-1)*len + i];
		}

		qInd = qEnd + 1;	// TODO Check the +1

		lowerLayer++;
		if(progfunc && progargs)
			progfunc(progargs, progmin + (progmax - progmin) * double(qEnd) / double(qPoints));

		if(pStop && *pStop)
			break;

	} while(qInd < qPoints);	// For all qs

	endKern = clock();

	FREE_GPUCalcMCOAJacobSphrTempl_MEMORY;	

	endAll = clock();

	printf("OA CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
		double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);

	if(progfunc && progargs)
		progfunc(progargs, progmin + (progmax - progmin));

	if ( cudaSuccess != err )
		return -1000 - (int)err;

	return 0;

}

///////////////////////////////////////////////////////////////////////////////
// Exposed functions that call the internal templated function
 #define EXPOSE_GPUCalcMCOAJacobSphr_MACRO(T1, DT)												\
	int GPUCalcMCOAJacobSphr																	\
	(long long voxels, int thDivs, int phDivs, T1 stepSz, T1 *inAmpData,  DT *inD, T1 *qs,		\
	T1 *intensities, int qPoints, long long maxIters, T1 convergence,							\
		progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)		\
{																								\
	 int res = GPUCalcMCOAJacobSphrTempl(voxels, thDivs, phDivs, stepSz, inAmpData,				\
		(DT##2 *)inD, qs, intensities, qPoints, maxIters, convergence,							\
								progfunc, progargs, progmin, progmax, pStop);					\
	 cudaDeviceReset(); return res;																\
}

EXPOSE_GPUCalcMCOAJacobSphr_MACRO(double, double)
/* Disable single precision
EXPOSE_GPUCalcMCOAJacobSphr_MACRO(double, float)
EXPOSE_GPUCalcMCOAJacobSphr_MACRO(float, float)
*/
#undef EXPOSE_GPUCalcMCOAJacobSphr_MACRO
