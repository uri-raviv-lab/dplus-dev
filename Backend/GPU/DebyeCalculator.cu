#include "DebyeCalculator.cuh"
#include "CommonPDB.cuh"
#include "CommonCUDA.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "maps/maps.cuh"

#include "Atomic Form Factor.h"

#include <time.h>

#define BLOCK_WIDTH 128

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>
#include <cub/block/block_reduce.cuh>
 
template <typename T> __device__ inline T sq(T x) { return x * x; }
template<typename T>
__device__ __forceinline__ T sinc(T x) {
	if(fabs(x) < 0.001) {
		return 1.0;
	}
	return sin(x) / x;
}


template <typename T>
float MULTIPLY_AFFS(const T &a, const T &b);

template <> 
__host__ __device__ __forceinline__ float MULTIPLY_AFFS(const float &a, const float &b){ return a*b; }
template <> 
__host__ __device__ __forceinline__ float MULTIPLY_AFFS(const float2 &a, const float2 &b){ return a.x*b.x + a.y*b.y; }

template <typename T1, typename T2>
T1 MULTIPLY(const T1 &a, const T2 &b);
template <>
__host__ __device__ __forceinline__ float2 MULTIPLY(const float2 &a, const float &b){ return make_float2(a.x*b, a.y*b); }
template <>
__host__ __device__ __forceinline__ float MULTIPLY(const float &a, const float &b){ return a*b; }

/*******************************************************************************
Debye32x32
Calculates the lower triangle of the Debye formalism
The actual matrix is broken down as follows:
	*
	**
	***
	****
	*****

Each '*' is actually a 32x32 block to be calculated and summed before being
saved to global memory and subsequently summed in turn (separate kernel).
*******************************************************************************/
template<typename FLOAT_TYPE, bool bUseBFactors, typename AFF_TYPE = float, int BLOCK_WIDTH_t = BLOCK_WIDTH>
__global__ void Debye32x32(const float4 * __restrict__ gLocs, const float *BFactors,
						const AFF_TYPE * __restrict__ affs, FLOAT_TYPE *reses,
						float qAng
						)
{
	unsigned int id = (blockIdx.x);
/*
	if(id = 3 && threadIdx.x == 0)
		printf("[%lld] --> %d\t%d\t%d"
		"\n",
		id, blockDim.x, gridDim.x, warpSize);
*/
//	if(id >= gridDim.x)
//		return;


// 	printf(
// 		"** = %u * %u\t{%u}\n",
//  		/*id,*/ blockIdx.x, blockDim.x, gridDim.x
// 		);
	unsigned int rt = sqrt(1 + 8.0f * id + 0.5f);
	unsigned int firsti = (unsigned int)( ( (1 + rt ) / 2) - 1);
	if(rt*rt > 1 + 8 * id)
		firsti--;
	unsigned int firstj = (id - ( (firsti * (firsti + 1) ) / 2));

/*
	if(id < 1024 && threadIdx.x == 4)
		printf(
			"[%lld] %d --> {%d, %d}\n",
			id,
			gridDim.x,
			int( ( (1 + sqrt(1 + 8 * id + 0.5f) ) / 2) - 1),
			(id - ( (firsti * (firsti + 1) ) / 2))
			);
*/
	firsti *= BLOCK_WIDTH_t;
	firstj *= BLOCK_WIDTH_t;

	__shared__ float4 sLocs[BLOCK_WIDTH_t * 2];
	__shared__ float sBFactors[BLOCK_WIDTH_t * 2];
	__shared__ AFF_TYPE sAffJ[BLOCK_WIDTH_t];
	

	if(bUseBFactors) sBFactors[BLOCK_WIDTH_t + threadIdx.x] = BFactors[firsti + threadIdx.x];
	sLocs[BLOCK_WIDTH_t + threadIdx.x] = gLocs[threadIdx.x + firsti];

	if(bUseBFactors) sBFactors[threadIdx.x] = BFactors[firstj + threadIdx.x];
	sLocs[threadIdx.x] = gLocs[threadIdx.x + firstj];
	sAffJ[threadIdx.x] = affs [threadIdx.x + firstj];

	__syncthreads();

	FLOAT_TYPE subTot = 0.0f;

	AFF_TYPE affi = affs[firsti + threadIdx.x];

	float qqDiv = qAng * qAng / (16.f * M_PI * M_PI);

	if(bUseBFactors)	// Debye-Waller
	{
		affi = MULTIPLY(affi, expf(-(sBFactors[BLOCK_WIDTH_t + threadIdx.x] * qqDiv)));
	}

// #pragma unroll // actually slows it down
	for(int j = 0; j < BLOCK_WIDTH_t; j++)
	{
		AFF_TYPE affj = sAffJ[j];
		if(bUseBFactors)	// Debye-Waller
		{
			affj = MULTIPLY(affj, exp(-(sBFactors[j] * qqDiv)));
		}

		float distance = sqrtf(
						sq(sLocs[BLOCK_WIDTH_t+threadIdx.x].x - sLocs[j].x) + 
						sq(sLocs[BLOCK_WIDTH_t+threadIdx.x].y - sLocs[j].y) + 
						sq(sLocs[BLOCK_WIDTH_t+threadIdx.x].z - sLocs[j].z)
						);		 

		float tmp = (MULTIPLY_AFFS(affi, affj) * sinc((qAng * 10.f) * distance)) * 
					// Check if on the main diagonal
					((firsti + threadIdx.x == firstj + j) ? 1.f : 2.f) *
					// Only the lower triangle
					((firsti + threadIdx.x >= firstj + j) ? 1.f : 0.f)
					;

		subTot += tmp;
	}
	__syncthreads();

	typedef cub::BlockReduce<FLOAT_TYPE, BLOCK_WIDTH_t> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	FLOAT_TYPE res = BlockReduce(temp_storage).Sum(subTot);

	__syncthreads();

	if(threadIdx.x == 0) 
	{
		reses[id] = res;
	}

}

template<typename FLOAT_TYPE, typename AFF_TYPE = float, int BLOCK_WIDTH_T = BLOCK_WIDTH>
int GPUCalcDebyeV2Template(int numQValues, float qMin, float qMax,
				   FLOAT_TYPE *outData,
				   int numAtoms, const int *atomsPerIon,
				   float4 *loc, u8 *ionInd,
				   float2 *anomalousVals,
				   bool bBfactors, float *BFactors,
				   float *coeffs, bool bSol,
				   bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
				   progressFunc progfunc, void *progargs,
				   double progmin, double progmax, int *pStop)
{
	cudaError err = cudaSuccess;
	
	cudaStream_t calcstream, memoryStream;
	CHKERR(cudaStreamCreate(&calcstream));
	CHKERR(cudaStreamCreate(&memoryStream));

	int numUnIons;
	std::size_t totalAtoms = 0;
	for(numUnIons = 0; totalAtoms < numAtoms && atomsPerIon[numUnIons] > 0; numUnIons++)
		totalAtoms += atomsPerIon[numUnIons];	// Count the number of unique ions

	float stepSize = (qMax - qMin) / float(numQValues-1);

	int comb = bSolOnly ? 0x00 : CALC_ATOMIC_FORMFACTORS;
	comb |= ((bSol && solvED != 0.0) ? CALC_DUMMY_SOLVENT : 0x00);
	comb |= (anomalousVals ? CALC_ANOMALOUS : 0x00);

	float4 *d_pdbLocs = NULL;
	AFF_TYPE *d_affsA = NULL;
	AFF_TYPE *d_affsB = NULL;
	FLOAT_TYPE *d_Ires = NULL;
	AFF_TYPE * h_pinned_affs = NULL;

	int roundedNumAtoms = maps::RoundUp((unsigned int)(numAtoms), BLOCK_WIDTH_T) * BLOCK_WIDTH_T;

	CHKERR(cudaMallocHost(&h_pinned_affs, sizeof(AFF_TYPE) * roundedNumAtoms));
	atomicFFCalculator affCalculator(comb, numAtoms, numUnIons, coeffs, atomsPerIon);

	if (comb & CALC_DUMMY_SOLVENT)
	{
		std::vector<float> ionRads(numUnIons);
		int off = 0;
		for (int i = 0; i < numUnIons; i++) {
			ionRads[i] = atmRad[atmInd[off]];
//			std::fill_n(ionRads.begin() + off, atomsPerIon[i], atmRad[atmInd[off]]);
			off += atomsPerIon[i];
		}
		affCalculator.SetSolventED(solvED, c1, ionRads.data(), bSolOnly);
	}

	if (comb & CALC_ANOMALOUS)
		affCalculator.SetAnomalousFactors(anomalousVals/*float2 sizeof numAtoms*/);

	CHKERR(cudaMallocHost(&h_pinned_affs, sizeof(AFF_TYPE) * roundedNumAtoms));

	CHKERR(cudaMalloc(&d_pdbLocs, sizeof(float4)		* roundedNumAtoms));
	CHKERR(cudaMalloc(&d_Ires, sizeof(FLOAT_TYPE)	* numQValues));
	CHKERR(cudaMalloc(&d_affsA, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMalloc(&d_affsB, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMemset(d_pdbLocs, 0, sizeof(float4)		* roundedNumAtoms));
	CHKERR(cudaMemset(d_affsA, 0, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMemset(d_affsB, 0, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMemset(d_Ires, 0, sizeof(FLOAT_TYPE)	* numQValues));

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = 1;
	dimGrid.x = ((numQValues % BLOCK_WIDTH == 0) ? (numQValues / BLOCK_WIDTH) : 
					(numQValues / BLOCK_WIDTH + 1));
	dimGrid.y = 1;

	if (progfunc)
	{
		if (progfunc)
			progfunc(progargs, 0.0);
	}

	CHKERR(cudaMemcpyAsync(d_pdbLocs, loc, sizeof(float4) * numAtoms, cudaMemcpyHostToDevice, memoryStream));

	if(bBfactors && false)
	{
//		CHKERR(cudaMalloc(&d_BFactors, sizeof(float) * numAtoms));
//		CHKERR(cudaMemcpyAsync(d_BFactors, BFactors, sizeof(float) * numAtoms, cudaMemcpyHostToDevice, memoryStream));
	}


	FLOAT_TYPE *d_triangularMatrices[2];
//	int tmSz = ( (numAtoms * (numAtoms - 1)) / 2) + numAtoms;
	long long numBlocksX = ((roundedNumAtoms % BLOCK_WIDTH == 0) ? (roundedNumAtoms / BLOCK_WIDTH) : 
					(roundedNumAtoms / BLOCK_WIDTH + 1));
	long long numBlocks = ( (numBlocksX * (numBlocksX + 1)) / 2);

	CHKERR(cudaMalloc(&d_triangularMatrices[0], sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMemset(d_triangularMatrices[0], 0, sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMalloc(&d_triangularMatrices[1], sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMemset(d_triangularMatrices[1], 0, sizeof(FLOAT_TYPE) * numBlocks) );
	
	comb |= (bBfactors ? 0x16 : 0x00);

	CHKERR(cudaStreamSynchronize(calcstream));

	for(int i = 0; i < numQValues; i++) 
	{
		if (progfunc)
		{
			if (progfunc)
				progfunc(progargs, (double(i+1) / double(numQValues+1) ) );

		}
		AFF_TYPE * d_nextAffs = (i % 2 == 0) ? d_affsA : d_affsB;

		affCalculator.GetAllAFFs(h_pinned_affs, qMin + stepSize * i);
		CHKERR(cudaMemcpyAsync(d_nextAffs, h_pinned_affs, sizeof(AFF_TYPE) * numAtoms, cudaMemcpyHostToDevice, memoryStream));
		CHKERR(cudaStreamSynchronize(memoryStream));

/*		if(bBfactors)
		{
			TRACE_KERNEL("Debye32x32");
			Debye32x32<FLOAT_TYPE, true, AFF_TYPE, BLOCK_WIDTH><<<numBlocks, BLOCK_WIDTH, 0, calcstream>>>
				(numAtoms, d_pdbLocs, d_BFactors, d_nextAffs,
				d_triangularMatrices[i%2], (qMin + i * stepSize) / 10.);
		}
		else
		{*/		

//		printf("Locations:\n"); TRACE_KERNEL("PrintFloat4DeviceMemory"); PrintFloat4DeviceMemory <<<roundedNumAtoms/BLOCK_WIDTH_T + 1, BLOCK_WIDTH_T >>> (d_pdbLocs, roundedNumAtoms); CHKERR(cudaDeviceSynchronize());
//		printf("Affs:\n"); TRACE_KERNEL("PrintFloatDeviceMemory"); PrintFloatDeviceMemory <<<roundedNumAtoms/BLOCK_WIDTH_T + 1, BLOCK_WIDTH_T >>> (d_nextAffs, roundedNumAtoms); CHKERR(cudaDeviceSynchronize());
		TRACE_KERNEL("Debye32x32");
		Debye32x32<FLOAT_TYPE, false, AFF_TYPE, BLOCK_WIDTH_T> <<<numBlocks, BLOCK_WIDTH_T, 0, calcstream >>>
			(d_pdbLocs, NULL, d_nextAffs,
			d_triangularMatrices[i%2], (qMin + i * stepSize) / 10.f);
	//	}

		CHKERR(cudaStreamSynchronize(calcstream));

		// nvcc/gcc on ubuntu choke on the cast, unable to find matching cubReduce
		cubReduce<FLOAT_TYPE>(d_triangularMatrices[i % 2], d_Ires + i, /*unsigned int*/(numBlocks), memoryStream);
	} // for i


	CHKERR(cudaStreamSynchronize(memoryStream));

	CHKERR(cudaMemcpy(outData, d_Ires, sizeof(FLOAT_TYPE) * numQValues, cudaMemcpyDeviceToHost));

	deviceFreePointer(d_pdbLocs);
	deviceFreePointer(d_affsA);
	deviceFreePointer(d_affsB);
	deviceFreePointer(d_Ires);
	deviceFreePointer(d_triangularMatrices[0]);
	deviceFreePointer(d_triangularMatrices[1]);

	CHKERR(cudaStreamDestroy(calcstream));
	CHKERR(cudaStreamDestroy(memoryStream));

#ifdef STANDALONE_DEBYE
	cudaDeviceReset();
#endif
	
	return 0;
}


///////////////////////////////////////////////////////////////////////////////
// Exposed functions that call the internal templated function
/* Disable single precision
int GPUCalcDebyeV2(int numQValues, float qMin, float qMax, float *outData,
	int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd,
	float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs,
	bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	if (anomalousVals == NULL)
	{
		return GPUCalcDebyeV2Template<float, float>(numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, loc, ionInd, NULL, bBfactors, BFactors, coeffs, bSol, bSolOnly,
			atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);
	}
	else
	{
		return GPUCalcDebyeV2Template<float, float2>(numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, loc, ionInd, anomalousVals, bBfactors, BFactors, coeffs, bSol, bSolOnly,
			atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);
	}
}
*/
int GPUCalcDebyeV2(int numQValues, float qMin, float qMax, double *outData,
	int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd,
	float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs,
	bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	if (anomalousVals == NULL)
	{
		return GPUCalcDebyeV2Template<double, float>(numQValues, qMin, qMax, outData, numAtoms,
		atomsPerIon, loc, ionInd, NULL, bBfactors, BFactors, coeffs, bSol, bSolOnly,
		atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);
	}
	else
	{
		return GPUCalcDebyeV2Template<double, float2>(numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, loc, ionInd, anomalousVals, bBfactors, BFactors, coeffs, bSol, bSolOnly,
			atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);
	}
}

/******************************************************************************/
// MAPS version
/*
template <typename T>
__host__ __device__ __forceinline__ T conj_if_complex(const T& x);

template<> __host__ __device__ __forceinline__ float conj_if_complex<float>(const float& x) { return x; }
template<> __host__ __device__ __forceinline__ double conj_if_complex<double>(const double& x) { return x; }
template<> __host__ __device__ __forceinline__ float2 conj_if_complex<float2>(const float2& x) { return make_float2(x.x, -x.y); }
template<> __host__ __device__ __forceinline__ double2 conj_if_complex<double2>(const double2& x) { return make_double2(x.x, -x.y); }
*/

template <typename T>
__host__ __device__ __forceinline__ T multConjRealOrComplex(const T& a, const T& b);

template<> __host__ __device__ __forceinline__ float multConjRealOrComplex<float>(const float& a, const float& b) { return a*b; }
template<> __host__ __device__ __forceinline__ double multConjRealOrComplex<double>(const double& a, const double& b) { return a*b; }
template<> __host__ __device__ __forceinline__ float2 multConjRealOrComplex<float2>(const float2& a, const float2& b) { return make_float2((a.x*b.x - a.y*-b.y), (a.x*-b.y + a.y*b.x)); }
template<> __host__ __device__ __forceinline__ double2 multConjRealOrComplex<double2>(const double2& a, const double2& b) { return make_double2((a.x*b.x - a.y*-b.y), (a.x*-b.y + a.y*b.x)); }

template <typename T1, typename T2>
__host__ __device__ __forceinline__ T1 iHaveToImplementTimes(const T1& a, const T2& b);

template<> __host__ __device__ __forceinline__ float2  iHaveToImplementTimes<float2, float>(const float2& a, const float& b) { return make_float2((a.x*b - a.y*-b), (a.y*-b + a.x*b)); }
template<> __host__ __device__ __forceinline__ double2 iHaveToImplementTimes<double2, double>(const double2& a, const double& b) { return make_double2((a.x*b - a.y*-b), (a.y*-b + a.x*b)); }

template<typename T>
__host__ __device__ __forceinline__ T iHaveToImplementTimes(const T& a, const T& b) { return a*b; }

template <typename RT, typename PT, typename AT, typename QT>
__host__ __device__ __forceinline__ RT DebyeContribution(const PT& posi, const PT& posj, const AT& affi, const AT& affj, const QT& q)
{
	return
		iHaveToImplementTimes(
			multConjRealOrComplex(affi, affj),
			AT(sinc((q) * sqrtf(
				sq(posi.x - posj.x) +
				sq(posi.y - posj.y) +
				sq(posi.z - posj.z)
				))
			)
		);
}

template<typename FLOAT_TYPE, typename AFF_TYPE, int BW, int CHUNK_SIZE, int IPX = 1>
__global__ void DebyeMapsV4(
	maps::ReductiveStaticOutput<FLOAT_TYPE,
		/* the number of output values */1,
		BW, IPX> result,
		maps::Window1DSingleGPU<float4, BW, 0, IPX> my_pos,
		maps::Block1DSingleGPU<float4, BW, IPX, 1, 1, maps::NoBoundaries, CHUNK_SIZE> positions,
		maps::Window1DSingleGPU<AFF_TYPE, BW, 0, IPX> my_aff,
		maps::Block1DSingleGPU<AFF_TYPE, BW, IPX, 1, 1, maps::NoBoundaries, CHUNK_SIZE> affs,
		float q
	)
{
	MAPS_INIT(result, my_aff, my_pos, positions, affs);

	// Needed if affs and positions are not padded with zeros.
//	if (my_pos.Items() == 0) return;

	//printf("%d\n", positions.chunks());

	float4 cur_pos[IPX];
	AFF_TYPE cur_aff[IPX];
	AFF_TYPE results[IPX] = { 0 };

	#pragma unroll
	MAPS_FOREACH(oiter, result)
	{
		cur_pos[oiter.index()] = *my_pos.align(oiter);
		cur_aff[oiter.index()] = *my_aff.align(oiter);
	}

	// Perform the multiplication
	for (int j = 0; j < positions.chunks(); ++j)
	{
#pragma unroll
		MAPS_FOREACH(oiter, result)
		{
			// Initialize B's iterator as well
			auto aff_iter = affs.align(oiter);

			//const auto& cur_pos = *my_pos.align(oiter);
			//const auto& cur_aff = *my_aff.align(oiter);

			//#pragma unroll
			MAPS_FOREACH_ALIGNED(pos_iter, positions, oiter)
			{
				results[oiter.index()] += DebyeContribution<FLOAT_TYPE>(
					cur_pos[oiter.index()], *pos_iter, cur_aff[oiter.index()], *aff_iter, q);
				++aff_iter;
			}
		}

		// Advance chunks efficiently
		maps::NextChunkAll(positions, affs);
	}

	FLOAT_TYPE resss = 0.0f;

	#pragma unroll
	MAPS_FOREACH(oiter, result)
	{
		resss += results[oiter.index()];
//		if (threadIdx.x + blockIdx.x * blockDim.x == 0)
//			printf("results[%d] = %f\n", oiter.index(), results[oiter.index()]);
	}

	result.begin()[0] += resss;

	// Write out results
	result.commit();
}

template<typename FLOAT_TYPE>
int GPUCalcDebyeV3MAPSTemplate(
	int numQValues, float qMin, float qMax, FLOAT_TYPE *outData, int numAtoms,
	const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	cudaError_t err = cudaSuccess;
	
	cudaStream_t calcstream, memoryStream;
	CHKERR(cudaStreamCreate(&calcstream));
	CHKERR(cudaStreamCreate(&memoryStream));

	int numUnIons, totalAtoms = 0;
	for(numUnIons = 0; totalAtoms < numAtoms && atomsPerIon[numUnIons] > 0; numUnIons++)
		totalAtoms += atomsPerIon[numUnIons];	// Count the number of unique ions

	float stepSize = (qMax - qMin) / float(numQValues-1);

	int comb = bSolOnly ? 0x00 : 0x01;
	comb |= ((bSol && solvED != 0.0) ? 0x02 : 0x00);

	float *d_affCoeffs = NULL;
	float4 *d_pdbLocs = NULL;
	float *d_atmRad = NULL;
	float *d_BFactors = NULL;
	float *d_affs = NULL;
	char  *d_ionIdx = NULL;
	FLOAT_TYPE *d_Ires = NULL;

	CHKERR(cudaMalloc(&d_affCoeffs,	sizeof(float)	* 9 * numUnIons));
	CHKERR(cudaMalloc(&d_pdbLocs,	sizeof(float4)		* numAtoms));
	CHKERR(cudaMalloc(&d_affs,		sizeof(float)	* numUnIons * numQValues));
	CHKERR(cudaMalloc(&d_ionIdx,	sizeof(char)	* numAtoms));
	CHKERR(cudaMalloc(&d_Ires,		sizeof(FLOAT_TYPE)	* numQValues));

	if(comb & 0x02) { // Fraser/dummy atom solvent
		CHKERR(cudaMalloc(&d_atmRad,		sizeof(float) * numUnIons));
	}

	CHKERR(cudaMemcpyAsync(d_affCoeffs, coeffs, sizeof(float) * 9 * numUnIons, cudaMemcpyHostToDevice, memoryStream));

	float *ionRads = NULL;
	if(comb & 0x02) {
		ionRads = new float[numUnIons];
		int off = 0;
		for(int i = 0; i < numUnIons; i++) {
			ionRads[i] = atmRad[atmInd[off]];
			off += atomsPerIon[i];
		}
		CHKERR(cudaMemcpyAsync(d_atmRad, ionRads , sizeof(float) * numUnIons, cudaMemcpyHostToDevice, memoryStream));
	}

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = 1;
	dimGrid.x = ((numQValues % BLOCK_WIDTH == 0) ? (numQValues / BLOCK_WIDTH) : 
					(numQValues / BLOCK_WIDTH + 1));
	dimGrid.y = 1;

	CHKERR(cudaStreamSynchronize(memoryStream));

	switch(comb) {
	case 1:	// 0x01 In Vac
		TRACE_KERNEL("AtomicFormFactorKernel<1, true>");
		AtomicFormFactorKernel<1, true><<<dimGrid, dimBlock, 0, calcstream>>>
					(qMin, stepSize, numQValues, d_affCoeffs, d_affs, numUnIons, NULL, 0.f);
		break;
	case 2:	// 0x02 Solvent only
		TRACE_KERNEL("AtomicFormFactorKernel<2. true>");
		AtomicFormFactorKernel<2, true><<<dimGrid, dimBlock, 0, calcstream>>>
					(qMin, stepSize, numQValues, NULL, d_affs, numUnIons, d_atmRad, solvED);
		break;
	case 3:	// 0x03 == 0x01 + 0x02 (Vac - solvent)
		TRACE_KERNEL("AtomicFormFactorKernel<3, true>");
		AtomicFormFactorKernel<3, true><<<dimGrid, dimBlock, 0, calcstream>>>
					(qMin, stepSize, numQValues, d_affCoeffs, d_affs, numUnIons, d_atmRad, solvED);
		break;
	}	// switch
	freePointer(ionRads);

	CHKERR(cudaMemcpyAsync(d_ionIdx, ionInd, sizeof(char) * numAtoms, cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(d_pdbLocs, loc, sizeof(float4) * numAtoms, cudaMemcpyHostToDevice, memoryStream));

	if(bBfactors)
	{
		CHKERR(cudaMalloc(&d_BFactors, sizeof(float) * numAtoms));
		CHKERR(cudaMemcpyAsync(d_BFactors, BFactors, sizeof(float) * numAtoms, cudaMemcpyHostToDevice, memoryStream));
	}


	FLOAT_TYPE *d_triangularMatrices[2];
//	int tmSz = ( (numAtoms * (numAtoms - 1)) / 2) + numAtoms;
	int numBlocksX = ((numAtoms % BLOCK_WIDTH == 0) ? (numAtoms / BLOCK_WIDTH) : 
					(numAtoms / BLOCK_WIDTH + 1));
	int numBlocks = ( (numBlocksX * (numBlocksX + 1)));

	CHKERR(cudaMalloc(&d_triangularMatrices[0], sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMemset(d_triangularMatrices[0], 0, sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMalloc(&d_triangularMatrices[1], sizeof(FLOAT_TYPE) * numBlocks) );
	CHKERR(cudaMemset(d_triangularMatrices[1], 0, sizeof(FLOAT_TYPE) * numBlocks) );
	
	comb |= (bBfactors ? 0x16 : 0x00);

	CHKERR(cudaStreamSynchronize(calcstream));
	
	for(int i = 0; i < numQValues; i++) 
	{
		/*
		if(bBfactors)
		{
			TRACE_KERNEL("Debug32x32");
			Debye32x32<FLOAT_TYPE, true><<<numBlocks, BLOCK_WIDTH, 0, calcstream>>>
				(numAtoms, d_pdbLocs, d_BFactors, d_affs + i * numUnIons, d_ionIdx,
				d_triangularMatrices[i%2], (qMin + i * stepSize) / 10.);
		}
		else
		{
			TRACE_KERNEL("Debye32x32");
			Debye32x32<FLOAT_TYPE, false><<<numBlocks, BLOCK_WIDTH, 0, calcstream>>>
				(numAtoms, d_pdbLocs, NULL, d_affs + i * numUnIons, d_ionIdx,
				d_triangularMatrices[i%2], (qMin + i * stepSize) / 10.);
		}
		*/
		CHKERR(cudaStreamSynchronize(calcstream));
		CHKERR(cudaStreamSynchronize(memoryStream));

		cubReduce<FLOAT_TYPE>(d_triangularMatrices[i%2], d_Ires + i, numBlocks, memoryStream);
	} // for i


	CHKERR(cudaStreamSynchronize(memoryStream));

	CHKERR(cudaMemcpy(outData, d_Ires, sizeof(FLOAT_TYPE) * numQValues, cudaMemcpyDeviceToHost));

	deviceFreePointer(d_affCoeffs);
	deviceFreePointer(d_pdbLocs);
	deviceFreePointer(d_atmRad);
	deviceFreePointer(d_affs);
	deviceFreePointer(d_BFactors);
	deviceFreePointer(d_Ires);
	deviceFreePointer(d_triangularMatrices[0]);
	deviceFreePointer(d_triangularMatrices[1]);

	CHKERR(cudaStreamDestroy(calcstream));
	CHKERR(cudaStreamDestroy(memoryStream));

#ifdef STANDALONE_DEBYE
	cudaDeviceReset();
#endif
	
	return 0;
}

int GPUCalcDebyeV3MAPS(
	int numQValues, float qMin, float qMax, float *outData, int numAtoms,
	const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	return GPUCalcDebyeV3MAPSTemplate(
	numQValues, qMin, qMax, outData, numAtoms, atomsPerIon, loc, ionInd, bBfactors,
	BFactors, coeffs, bSol, bSolOnly, atmInd, atmRad, solvED,
	progfunc, progargs, progmin, progmax, pStop);
}

int GPUCalcDebyeV3MAPS(
	int numQValues, float qMin, float qMax, double *outData, int numAtoms,
	const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	return GPUCalcDebyeV3MAPSTemplate(
	numQValues, qMin, qMax, outData, numAtoms, atomsPerIon, loc, ionInd, bBfactors,
	BFactors, coeffs, bSol, bSolOnly, atmInd, atmRad, solvED,
	progfunc, progargs, progmin, progmax, pStop);
}

template<typename FLOAT_TYPE, typename AFF_TYPE, int BLOCK_WIDTH_T, int CHUNK_SIZE, int ITEMS_PER_THREAD>
int GPUCalcDebyeV4MAPSTemplate(
	int numQValues, float qMin, float qMax, FLOAT_TYPE *outData, int numAtoms,
	const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	cudaError_t err = cudaSuccess;

	cudaStream_t computeStream, memoryStream;
	CHKERR(cudaStreamCreate(&computeStream));
	CHKERR(cudaStreamCreate(&memoryStream));

	int numUnIons, totalAtoms = 0;
	for (numUnIons = 0; totalAtoms < numAtoms && atomsPerIon[numUnIons] > 0; numUnIons++)
		totalAtoms += atomsPerIon[numUnIons];	// Count the number of unique ions

	int comb = bSolOnly ? 0x00 : CALC_ATOMIC_FORMFACTORS;
	comb |= ((bSol && solvED != 0.0) ? CALC_DUMMY_SOLVENT : 0x00);
	comb |= (anomalousVals ? CALC_ANOMALOUS : 0x00);

	atomicFFCalculator affCalculator(comb, numAtoms, numUnIons, coeffs, atomsPerIon);

	if (comb & CALC_DUMMY_SOLVENT)
	{
		std::vector<float> ionRads(numUnIons);
		int off = 0;
		for (int i = 0; i < numUnIons; i++) {
			ionRads[i] = atmRad[atmInd[off]];
			off += atomsPerIon[i];
		}
		affCalculator.SetSolventED(solvED, c1, ionRads.data(), bSolOnly);
	}

	float stepSize = (qMax - qMin) / float(numQValues - 1);

	float4 *d_pdbLocs = NULL;
	AFF_TYPE *d_affsA = NULL;
	AFF_TYPE *d_affsB = NULL;

	AFF_TYPE * h_pinned_affs;
	FLOAT_TYPE *d_kernelResults;

	int roundedNumAtoms = maps::RoundUp((unsigned int)(numAtoms), BLOCK_WIDTH_T) * BLOCK_WIDTH_T;

	CHKERR(cudaMalloc(&d_pdbLocs, sizeof(float4) * roundedNumAtoms));
	CHKERR(cudaMemset(d_pdbLocs, 0, sizeof(float4) * roundedNumAtoms));
	CHKERR(cudaMemcpyAsync(d_pdbLocs, atomLocations, sizeof(float4) * numAtoms, cudaMemcpyHostToDevice, computeStream));

	CHKERR(cudaMalloc(&d_kernelResults, sizeof(FLOAT_TYPE) * numQValues));
	CHKERR(cudaMemset(d_kernelResults, 0, sizeof(FLOAT_TYPE) * numQValues));

	///////////////////////////////////////////////////
	// Deal with calculating atomic form factors
	///////////////////////////////////////////////////
	CHKERR(cudaMallocHost(&h_pinned_affs, sizeof(AFF_TYPE) * roundedNumAtoms));
	affCalculator.GetAllAFFs(h_pinned_affs, qMin, (void*)anomalousVals);
	
	CHKERR(cudaMalloc(&d_affsA, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMalloc(&d_affsB, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMemset(d_affsA, 0, sizeof(AFF_TYPE) * roundedNumAtoms));
	CHKERR(cudaMemset(d_affsB, 0, sizeof(AFF_TYPE) * roundedNumAtoms));

	maps::ReductiveStaticOutput<FLOAT_TYPE, /* the number of output values */1, BLOCK_WIDTH_T, ITEMS_PER_THREAD> rs_kernel_res;

	maps::Window1DSingleGPU<float4, BLOCK_WIDTH_T, 0, ITEMS_PER_THREAD> w1d_pos;
	w1d_pos.m_ptr = d_pdbLocs;
	w1d_pos.m_dimensions[0] = roundedNumAtoms;
	w1d_pos.m_stride = roundedNumAtoms;
	maps::Block1DSingleGPU<float4, BLOCK_WIDTH_T, ITEMS_PER_THREAD, 1, 1, maps::NoBoundaries, CHUNK_SIZE> b1d_pos;
	b1d_pos.m_ptr = d_pdbLocs;
	b1d_pos.m_dimensions[0] = roundedNumAtoms;
	b1d_pos.m_stride = roundedNumAtoms;

	maps::Window1DSingleGPU<AFF_TYPE, BLOCK_WIDTH_T, 0, ITEMS_PER_THREAD> w1d_affs;
	w1d_affs.m_ptr = d_affsA;
	w1d_affs.m_dimensions[0] = roundedNumAtoms;
	w1d_affs.m_stride = roundedNumAtoms;
	maps::Block1DSingleGPU<AFF_TYPE, BLOCK_WIDTH_T, ITEMS_PER_THREAD, 1, 1, maps::NoBoundaries, CHUNK_SIZE> b1d_affs;
	b1d_affs.m_ptr = d_affsA;
	b1d_affs.m_dimensions[0] = roundedNumAtoms;
	b1d_affs.m_stride = roundedNumAtoms;

	for (size_t i = 0; i < numQValues; i++)
	{
		float qqq = qMin + i * stepSize;
		
		rs_kernel_res.m_ptr = d_kernelResults + i;

		// Compute next affs
		AFF_TYPE * d_nextAffs = (i % 2 == 0) ? d_affsA : d_affsB;
		affCalculator.GetAllAFFs(h_pinned_affs, qqq + stepSize);

		CHKERR(cudaMemcpyAsync(d_nextAffs, h_pinned_affs, sizeof(float) * numAtoms, cudaMemcpyHostToDevice, memoryStream));

		w1d_affs.m_ptr = d_nextAffs;
		b1d_affs.m_ptr = d_nextAffs;
		
/*		printf("Items = %d, Blocks = %d, block size = %d\n", 
			roundedNumAtoms,
			roundedNumAtoms / BLOCK_WIDTH_T / ITEMS_PER_THREAD,
			BLOCK_WIDTH_T);
			*/
		CHKERR(cudaStreamSynchronize(memoryStream));
		CHKERR(cudaStreamSynchronize(computeStream));
		TRACE_KERNEL("DebyeMapsV4");
		DebyeMapsV4<FLOAT_TYPE, AFF_TYPE, BLOCK_WIDTH_T, CHUNK_SIZE, ITEMS_PER_THREAD> <<<roundedNumAtoms / BLOCK_WIDTH_T / ITEMS_PER_THREAD, BLOCK_WIDTH_T, 0, computeStream >>>(
			rs_kernel_res,
			w1d_pos, b1d_pos,
			w1d_affs, b1d_affs,
			qqq
			);


		// Report progress
		if (progfunc)
			progfunc(progargs, (double(i + 1) / double(numQValues + 1)));

	} // for i

	CHKERR(cudaStreamSynchronize(computeStream));

	CHKERR(cudaMemcpy(outData, d_kernelResults, sizeof(FLOAT_TYPE) * numQValues, cudaMemcpyDeviceToHost));

//	std::vector<AFF_TYPE>

	//printf("outdata[0] = %f\n", outData[0]);

	deviceFreePointer(d_affsA);
	deviceFreePointer(d_affsB);
	deviceFreePointer(d_pdbLocs);
	deviceFreePointer(d_kernelResults);
	
	if (h_pinned_affs)
		cudaFreeHost(h_pinned_affs);
	h_pinned_affs = NULL;

	CHKERR(cudaStreamDestroy(computeStream));
	CHKERR(cudaStreamDestroy(memoryStream));

	return 0;

}

int GPUCalcDebyeV4MAPS(
	int numQValues, float qMin, float qMax, float *outData, int numAtoms,
	const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	if (anomalousVals == NULL)
	{
		return GPUCalcDebyeV4MAPSTemplate<float, float, 128, 128, 1>(
			numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, atomLocations, NULL, bBfactors, BFactors,
			coeffs, bSol, bSolOnly, atmInd, atmRad, solvED, c1,
			progfunc, progargs, progmin, progmax, pStop);
	}
	else
	{
/*		return GPUCalcDebyeV4MAPSTemplate<float, float2, 128, 128, 1>(
			numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, atomLocations, anomalousVals, bBfactors, BFactors,
			coeffs, bSol, bSolOnly, atmInd, atmRad, solvED,
			progfunc, progargs, progmin, progmax, pStop);
			*/
			fprintf(stderr, "Houston, we have a problem. The reduction requires float*2*...\n");
		return -3648;
	}
}

int GPUCalcDebyeV4MAPS(
	int numQValues, float qMin, float qMax, double *outData, int numAtoms,
	const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	if (anomalousVals == NULL)
	{
		return GPUCalcDebyeV4MAPSTemplate<double, float, 32, 32, 1>(
			numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, atomLocations, anomalousVals, bBfactors, BFactors,
			coeffs, bSol, bSolOnly, atmInd, atmRad, solvED, c1,
			progfunc, progargs, progmin, progmax, pStop);
	}
	else
	{
/*		return GPUCalcDebyeV4MAPSTemplate<double, float2, 128, 128, 1>(
			numQValues, qMin, qMax, outData, numAtoms,
			atomsPerIon, atomLocations, anomalousVals, bBfactors, BFactors,
			coeffs, bSol, bSolOnly, atmInd, atmRad, solvED,
			progfunc, progargs, progmin, progmax, pStop);
			*/
		fprintf(stderr, "Houston, we have a problem. The reduction requires double*2*...\n");
		return -3648;
	}
}
