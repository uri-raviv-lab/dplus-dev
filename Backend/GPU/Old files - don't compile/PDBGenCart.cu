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

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

// Allocates more memory and performs more computations in favor of cancelling the "if"
// SLOWER, hmmph...
//#define ADDITIONAL_ALLOC

#define CONST_COEFFS
//#define SVG_SOL_ONLY
// Constant memory --> 209 * 9 coefficients
__device__ __constant__ double _GCOEFFS_ [1881];


///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of voxel solvent
__global__ void CalcPDBVoxelSolventKernelCart(const int offset, unsigned short dimx, const double stepSize,
							  const double2 *idx, double2 *data, const u64 numVoxels, const double voxelStep,
							  const ulonglong4 *voxDim, const double4 *voxCOM,
							  const double solED, const u64 totalDoubles)
{
	int tid = threadIdx.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid) + offset;
	if(id*2 >= totalDoubles)
		return;

	double2 res = data[id];

	double2 input = idx[id];
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);

 	double qx, qy, qz;
 	qx = (double(in1->w) - (double(dimx - 1) / 2.0)) * stepSize;
 	qy = (double(in1->z) - (double(in1->x - 1) / 2.0)) * stepSize;
 	qz = (double(in1->y) - (double(in2->x - 1) / 2.0)) * stepSize;
 
 	double q = sqrt(qx*qx + qy*qy + qz*qz);

	double va = 0.0, phase = 0.0, prevVAX = 0.0, prevVAY = 0.0, prevVAZ = 0.0;
	ulonglong4 vDims;
	double4 vCOM;
	u64 prevVDX = 0;
	u64 prevVDY = 0;
	u64 prevVDZ = 0;

	for(u64 i = 0; i < numVoxels; i++) {
		vDims = voxDim[i];
		vCOM  = voxCOM[i];
		double vcx = vCOM.x;
		double vcy = vCOM.y;
		double vcz = vCOM.z;
		
		if(vDims.x != prevVDX) {
			double vdx = double(vDims.x);
			prevVDX = vDims.x;
			if(qx == 0.0 || vDims.x == 0)
				prevVAX = voxelStep * vdx;
			else
				prevVAX = (sin(qx * vdx * voxelStep / 2.0) / (qx * vdx * voxelStep / 2.0)) * voxelStep * vdx;
		}
		if(vDims.y != prevVDY) {
			double vdy = double(vDims.y);
			prevVDY = vDims.y;
			if(qy == 0.0 || vDims.y == 0)
				prevVAY = voxelStep * vdy;
			else
				prevVAY = (sin(qy * vdy * voxelStep / 2.0) / (qy * vdy * voxelStep / 2.0)) * voxelStep * vdy;
		}
		if(vDims.z != prevVDZ) {
			double vdz = double(vDims.z);
			prevVDZ = vDims.z;
			if(qz == 0.0 || vDims.z == 0)
				prevVAZ = voxelStep * vdz;
			else
				prevVAZ = (sin(qz * vdz * voxelStep / 2.0) / (qz * vdz * voxelStep / 2.0)) * voxelStep * vdz;
		}

		va = prevVAX * prevVAY * prevVAZ * solED;

		phase = (qx * vcx + qy * vcy + qz * vcz);

		double sn, cs;
		sincos(phase, &sn, &cs);

		res.y += va * sn;
		res.x += va * cs;
	}

 	data[id] = res;
}
// End of voxel solvent kernel
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of dummy atom solvent
__global__ void CalcPDBDummyAtomSolventKernelCart(const int offset, unsigned short dimx, const double stepSize,
							  const double2 *idx, double2 *data, const double4 *locs,
							  const double solED, const float *radii, const u8 *atmInd, const int numAtoms,
							  const u64 totalDoubles, const int atomsInShared, const int atomParts)
{
	int tid = threadIdx.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid) + offset;
	if(id*2 >= totalDoubles)
		return;

	double2 res = data[id];
	
	// Input (in hex form)
	// REAL = [ XXXX YYYY ZZZZ DYDY ]
	// IMAG = [ 0000 0000 0000 DZDZ ]
	double2 input = idx[id];
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);
	double4 loc;

	// TODO::Change when moving to sphereGrid
	unsigned short xi   = in1->w;
	unsigned short yi   = in1->z;
	unsigned short zi   = in1->y;

	unsigned short dimy = in1->x;
	unsigned short dimz = in2->x;

 	double qx, qy, qz;
 	qx = (double(xi) - (double(dimx - 1) / 2.0)) * stepSize;
 	qy = (double(yi) - (double(dimy - 1) / 2.0)) * stepSize;
 	qz = (double(zi) - (double(dimz - 1) / 2.0)) * stepSize;

	double q = sqrt(qx*qx + qy*qy + qz*qz);
	double gi = 0.0;
	u8 prevAtm = 255;
	
	for(int i = 0; i < numAtoms; i++) {
		loc = locs[i];
		double phase = qx * loc.x + qy * loc.y + qz * loc.z;
		if(prevAtm != atmInd[i]) {
			prevAtm = atmInd[i];
			double rad = radii[prevAtm];
		
	 		gi = /*5.56832799683*/-4.1887902047863909846 /*4\pi/3*/ /*5.5683279968317084528 /*pi^1.5*/ * rad * rad * rad * exp( -(rad * q * rad * q / 4.0) ) * solED;
		}

		double sn, cs;
		sincos(phase, &sn, &cs);

		res.x += gi * cs;
 		res.y += gi * sn;
	}

 	res.x = res.x;
 	res.y = res.y;
	data[id] = res;
}
// End of kernel for calculation of dummy atom solvent
///////////////////////////////////////////////////////////////////////////////


//#define USE_KERNEL_AS_CPU
//#define CACHE_ATOM_PARTS	// Doesn't work with large files with a remainder  TODO::Later

///////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
#ifdef USE_KERNEL_AS_CPU
void CalcPDBKernel(const int offset, unsigned short dimx, const double stepSize,
				   double *data, const double *x, const double *y, const double *z,
				   const u8 *ionInd, const int numAtoms, const double *coeffs, const u8 numCoeffs,
				   const unsigned long long totalDoubles, unsigned long long idHalf)
{	
	idHalf += offset;
	unsigned long long id = idHalf * 2;
#else
__global__ void CalcPDBKernelCart(const int offset, unsigned short dimx, const double stepSize,
							  const double2 *idx, double2 *data, const double4 *loc,
							  const u8 *ionInd, const int numAtoms, /*const double *globalCoeffs,*/ const u8 numCoeffs,
							  const unsigned long long totalDoubles, const int atomsInShared, const int atomParts)
{	
	int tid = threadIdx.x;
	//int block = blockDim.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid) + offset;
#ifndef ADDITIONAL_ALLOC
	if(id * 2 >= totalDoubles)
		return;
#endif
	double2 res = data[id];

	//extern __shared__ double coeffs[];
#ifdef CACHE_ATOM_PARTS
	double *cachedAtomX = &coeffs[numCoeffs];
	double *cachedAtomY = &cachedAtomX[atomsInShared];
	double *cachedAtomZ = &cachedAtomY[atomsInShared];
	u8 *cachedIons = (u8 *)&cachedAtomZ[atomsInShared];
#endif
	
#endif
	// NOTE: "coeffs" must be row-major
 
	// Input (in hex form)
	// REAL = [ XXXX YYYY ZZZZ DYDY ]
	// IMAG = [ 0000 0000 0000 DZDZ ]

	double2 input = idx[id];
	double4 coord;
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);

	// TODO::Change when moving to sphereGrid
// 	unsigned short xi   = in1->w;
// 	unsigned short yi   = in1->z;
// 	unsigned short zi   = in1->y;
// 
// 	unsigned short dimy = in1->x;
// 	unsigned short dimz = in2->x;

 	double qx, qy, qz;
 	qx = (double(in1->w) - (double(dimx - 1) / 2.0)) * stepSize;
 	qy = (double(in1->z) - (double(in1->x - 1) / 2.0)) * stepSize;
 	qz = (double(in1->y) - (double(in2->x - 1) / 2.0)) * stepSize;
 
 	double q = sqrt(qx*qx + qy*qy + qz*qz), aff = 0.0;
 	double q10 = q / 10.0;
 	// The number is (4pi)^2
 	double sqq = q10 * q10 / (157.913670417429737901351855998);

#ifndef USE_KERNEL_AS_CPU
#ifndef CONST_COEFFS
	// t0 reads 0,n,2n,...
	// t1 reads 1,n+1,...
	if((tid) < numCoeffs)
		coeffs[        tid] = globalCoeffs[        tid];
	if((block + tid) < numCoeffs)
		coeffs[  block+tid] = globalCoeffs[  block+tid];
	if((2*block + tid) < numCoeffs)
		coeffs[2*block+tid] = globalCoeffs[2*block+tid];
#endif
#endif

#ifdef CACHE_ATOM_PARTS
	for(int part = 0; part < atomParts; part++) {
		int atomsInThisPart = (part == atomParts - 1) ? (numAtoms % atomsInShared) : atomsInShared;
		//const int atomsInThisPart = atomsInShared;
#ifdef _DEBUG
		if(id == 0)
			printf("Kernel:\npart = %d\nnumAtoms = %d\natomsInShared = %d\nnumAtoms %% atomsInShared = %d\natomsInThisPart = %d\n",
							part, numAtoms, atomsInShared, numAtoms % atomsInShared, atomsInThisPart);
#endif
		if(atomsInThisPart == 0)
			break;
		if(tid < atomsInThisPart) {
			int poffset = atomsInShared * part + tid;
			cachedAtomX[tid] = x[poffset];
			cachedAtomY[tid] = y[poffset];
			cachedAtomZ[tid] = z[poffset];
			cachedIons [tid] = ionInd[poffset];
		}

		#ifndef USE_KERNEL_AS_CPU
		// Sync threads in the block before using coeffs and cached atoms
		__syncthreads();
		#endif

 		for(int i = 0; i < atomsInThisPart; i++) {
 			u8 ion = cachedIons[i];

			double phase = qx * cachedAtomX[i] + qy * cachedAtomY[i] + qz * cachedAtomZ[i];
#else
		#if !defined(USE_KERNEL_AS_CPU) && !defined(CONST_COEFFS)
		// Sync threads in the block before using coeffs
		__syncthreads();
		#endif

#define a1 _GCOEFFS_[ion * 9 + 0]
#define b1 _GCOEFFS_[ion * 9 + 1]
#define a2 _GCOEFFS_[ion * 9 + 2]
#define b2 _GCOEFFS_[ion * 9 + 3]
#define a3 _GCOEFFS_[ion * 9 + 4]
#define b3 _GCOEFFS_[ion * 9 + 5]
#define a4 _GCOEFFS_[ion * 9 + 6]
#define b4 _GCOEFFS_[ion * 9 + 7]
#define c  _GCOEFFS_[ion * 9 + 8]

#define LOOP_DENOM 5
		u8 lastIonInd = 255;
		int reps = LOOP_DENOM * int(numAtoms / LOOP_DENOM);
		for(int i = 0; i < reps; ) {
			u8 ion;
			double phase;
#endif
			
#pragma unroll 5
			for(int ii = 0; ii < LOOP_DENOM; ii++) {
				//// REP
				ion = ionInd[i];
				coord = loc[i++];
				phase = qx * coord.x + qy * coord.y + qz * coord.z;
				if(lastIonInd != ion) {
  					aff = a1 * exp(-b1 * sqq) + a2 * exp(-b2 * sqq)
 						+ a3 * exp(-b3 * sqq) + a4 * exp(-b4 * sqq) + c;
					lastIonInd = ion;
				}
						
				double sn, cs;
				sincos(phase, &sn, &cs);

 				res.x += aff * cs;
 				res.y += aff * sn;
			}

		}

		for(int i = reps; i < numAtoms; i++) {
			u8 ion = ionInd[i];
			coord = loc[i];
			double phase = qx * coord.x + qy * coord.y + qz * coord.z;
			if(lastIonInd != ion) {
  				aff = a1 * exp(-b1 * sqq) + a2 * exp(-b2 * sqq)
 					+ a3 * exp(-b3 * sqq) + a4 * exp(-b4 * sqq) + c;
				lastIonInd = ion;
			}
		
			double sn, cs;
			sincos(phase, &sn, &cs);

 			res.x += aff * cs;
 			res.y += aff * sn;
		}

#ifdef CACHE_ATOM_PARTS
 	}
#endif
 	data[id] = res;
}
// 

////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////
int
GPUCalcPDBCart(u64 voxels, unsigned short dimx, double qmax, unsigned short sections, double *outData,
		   double *loc, u8 *ionInd,
		   int numAtoms, double *coeffs, int numCoeffs, bool bSolOnly,
		   u8 * atmInd, float *atmRad, double solvED,	u8 solventType,// FOR DUMMY ATOM SOLVENT
		   double *solCOM, u64 *solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
		   double *outSolCOM, u64 *outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
		   progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	// Number of (complex) voxels (the size of the data is 2 * voxels * sizeof(double)
	//                                                  OR voxels * sizeof(complex<double>)

	// TODO (A LOT LATER)::OneStepAhead: Multi-GPU support?
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

	

#ifdef CONST_COEFFS
	err = cudaMemcpyToSymbol(_GCOEFFS_, coeffs, sizeof(double) * numCoeffs);
	if(err != cudaSuccess) {
		return -3000 - err;
	}
#else
	u64 usedSharedMemory = 0;
	u64 freeSharedMemory = devProp.sharedMemPerBlock;

	if((devProp.sharedMemPerBlock - usedSharedMemory) < numCoeffs * sizeof(double)) {
		printf("(For now...) Not enough shared memory for coeffs! (%llu out of %llu bytes)\n", 
			   (u64)(numCoeffs * sizeof(double)), (u64)devProp.sharedMemPerBlock);
		return -2001;
	}
	usedSharedMemory += numCoeffs * sizeof(double);
	freeSharedMemory -= numCoeffs * sizeof(double);
#endif

	const int PARTS = 1;
	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;

	u64 totalSize = voxels;
#ifdef ADDITIONAL_ALLOC
	int remainder = voxels % maxThreadsPerBlock;
	totalSize += (u64)remainder;
	const u64 N = (totalSize / maxThreadsPerBlock);
#else
	const u64 N = (voxels / maxThreadsPerBlock) + 1;
#endif
	//const u64 voxelsPerPart = int(0.5 + (double)N / (double)PARTS);
	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

	double stepSize = 2.0 * qmax / (double)sections;

#ifdef CACHE_ATOM_PARTS
	// The exact amount of atoms we can cram into the shared memory
	int atomsInShared = MIN((int)freeSharedMemory / (sizeof(double) * 3 + sizeof(u8)),
									maxThreadsPerBlock);
	int atomParts = 1 + numAtoms / atomsInShared;
	usedSharedMemory += atomsInShared * (sizeof(double) * 3 + sizeof(u8));
	freeSharedMemory -= atomsInShared * (sizeof(double) * 3 + sizeof(u8));

#ifdef _DEBUG
	printf("\n\n------------\natomsInShared = %d\natomParts = %d\nremainder = %d\nnumAtoms = %d\n", atomsInShared, atomParts, numAtoms % atomsInShared, numAtoms);
#endif

#endif


#ifdef USE_KERNEL_AS_CPU
	for(unsigned long long i = 0 ; i < voxels; i++ ) {
		CalcPDBKernel(0, dimx, stepSize, outData, 
					  x, y, z, ionInd,
					  numAtoms, coeffs, numCoeffs, (voxels * 2) - 1, 
					  i);

		if(pStop && *pStop)
			break;
	}

		progfunc(progargs, progmin + (progmax - progmin) * ((i + 1) / double(PARTS)));
	}
#else

	// "Assertion" to verify that we can load coeffs to shared memory
	if(numCoeffs > voxels)
		return -2000;



	double *dCoeffs = NULL;
	float *dAtmRadii;
	u8  *dIonInd, *dAtmInd;
	double4 *dLoc;

	double2 *dData;
	double2 *dIdx;

	double4 *dSolCOM, *dOsolCOM;
	ulonglong4 *dSolDims, *dOsolDims;

	err = cudaMalloc(&dIdx,			sizeof(double2) * totalSize);
	err = cudaMalloc(&dData,		sizeof(double2) * totalSize);
	err = cudaMalloc(&dCoeffs,		sizeof(double) * numCoeffs);
	err = cudaMalloc(&dAtmRadii,	sizeof(float) * 119);
	err = cudaMalloc(&dLoc,			sizeof(double4)* numAtoms);
	err = cudaMalloc(&dIonInd,		sizeof(u8) * numAtoms);
	err = cudaMalloc(&dAtmInd,		sizeof(u8) * numAtoms);
	err = cudaMalloc(&dSolDims,		sizeof(ulonglong4) * solDimLen);
	err = cudaMalloc(&dSolCOM,		sizeof(double4) * solDimLen);
	err = cudaMalloc(&dOsolDims,	sizeof(ulonglong4) * outSolDimLen);
	err = cudaMalloc(&dOsolCOM,		sizeof(double4) * outSolDimLen);
	
	if ( cudaSuccess != err ) {
		printf("Error allocating data: %d", (int)err);
		cudaFree(dLoc);
		cudaFree(dIonInd);
		cudaFree(dData);
		cudaFree(dIdx);
		cudaFree(dCoeffs);
		cudaFree(dAtmInd);
		cudaFree(dAtmRadii);
		cudaFree(dSolDims);
		cudaFree(dSolCOM);
		cudaFree(dOsolDims);
		cudaFree(dOsolCOM);

		return -1000 - (int)err;
	}

#ifdef ADDITIONAL_ALLOC
	err = cudaMemset(dIdx,		0,			sizeof(double2) * totalSize							  );
#endif
	err = cudaMemcpy(dIdx,		outData,	sizeof(double2) * voxels,		cudaMemcpyHostToDevice);
	err = cudaMemset(dData,		0,			sizeof(double2) * totalSize							  );
	err = cudaMemcpy(dCoeffs,	coeffs,		sizeof(double) * numCoeffs,		cudaMemcpyHostToDevice);
	err = cudaMemcpy(dLoc,		loc,		sizeof(double4)* numAtoms,		cudaMemcpyHostToDevice);
	err = cudaMemcpy(dIonInd,	ionInd,		sizeof(u8) * numAtoms,			cudaMemcpyHostToDevice);
	err = cudaMemcpy(dAtmInd,	atmInd,		sizeof(u8) * numAtoms,			cudaMemcpyHostToDevice);
	err = cudaMemcpy(dAtmRadii,	atmRad,		sizeof(float) * 119,			cudaMemcpyHostToDevice);
	err = cudaMemcpy(dSolDims,	solDims,	sizeof(ulonglong4) * solDimLen,	cudaMemcpyHostToDevice);
	err = cudaMemcpy(dSolCOM,	solCOM,		sizeof(double4) * solDimLen,	cudaMemcpyHostToDevice);
	err = cudaMemcpy(dOsolDims,	outSolDims,	sizeof(ulonglong4) * outSolDimLen,	cudaMemcpyHostToDevice);
	err = cudaMemcpy(dOsolCOM,	outSolCOM,	sizeof(double4) * outSolDimLen,	cudaMemcpyHostToDevice);
	
	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dLoc);
		cudaFree(dIonInd);
		cudaFree(dData);
		cudaFree(dIdx);
		cudaFree(dCoeffs);
		cudaFree(dAtmInd);
		cudaFree(dAtmRadii);
		cudaFree(dSolDims);
		cudaFree(dSolCOM);
		cudaFree(dOsolDims);
		cudaFree(dOsolCOM);

		return -1000 - (int)err;
	}
	endMem = clock();

	for(int i = 0; i < PARTS; i++) {
#ifndef SVG_SOL_ONLY
		if(!bSolOnly)
 			CalcPDBKernelCart<<<grid, threads/*, usedSharedMemory*/>>>(
										 0/*i * voxelsPerPart*/, dimx, stepSize, 
										 dIdx, dData, 
 										 dLoc, dIonInd,
 										 numAtoms, /*dCoeffs,*/ numCoeffs,(voxels * 2) - 1
#ifdef CACHE_ATOM_PARTS
										 , atomsInShared, atomParts);
#else
										 ,0, 0);
#endif

#endif // SVG_SOL_ONLY

		if(solvED > 0.0 && solventType == 4)
			CalcPDBDummyAtomSolventKernelCart <<<grid, threads/*, usedSharedMemory*/ >>>(
  										 0/*i * voxelsPerPart*/, dimx, stepSize, 
  										 dIdx, dData, dLoc,
  										 solvED, dAtmRadii, dAtmInd,
   										 numAtoms, (voxels * 2) - 1
  #ifdef CACHE_ATOM_PARTS
  										 , atomsInShared, atomParts);
  #else
  										 ,0, 0);
  #endif
		// Voxel Based Solvent 
		if(solvED > 0.0 && solDimLen > 0)
			CalcPDBVoxelSolventKernelCart<<<grid, threads/*, usedSharedMemory*/>>>(
										 0/*i * voxelsPerPart*/, dimx, stepSize, 
										 dIdx, dData, solDimLen, voxStep,
 										 dSolDims, dSolCOM, -solvED, (voxels * 2) - 1);

		// Voxel Based Outer Solvent 
		if(outerSolED > 0.0 && outSolDimLen > 0)
			CalcPDBVoxelSolventKernelCart<<<grid, threads/*, usedSharedMemory*/>>>(
										 0/*i * voxelsPerPart*/, dimx, stepSize, 
										 dIdx, dData, outSolDimLen, voxStep,
 										 dOsolDims, dOsolCOM, outerSolED, (voxels * 2) - 1);

		// Check for launch errors
		err = cudaGetLastError();
		if ( cudaSuccess != err ) {
			cudaFree(dLoc);
			cudaFree(dIonInd);
			cudaFree(dData);
			cudaFree(dIdx);
			cudaFree(dCoeffs);
			cudaFree(dAtmInd);
			cudaFree(dAtmRadii);
			cudaFree(dSolDims);
			cudaFree(dSolCOM);
			cudaFree(dOsolDims);
			cudaFree(dOsolCOM);

			return -1000 - (int)err;
		}

		cudaThreadSynchronize();
		endKern = clock();
		// Check for execution errors
		err = cudaGetLastError();	
		if ( cudaSuccess != err )
		{
			cudaFree(dLoc);
			cudaFree(dIonInd);
			cudaFree(dData);
			cudaFree(dIdx);
			cudaFree(dCoeffs);
			cudaFree(dAtmInd);
			cudaFree(dAtmRadii);
			cudaFree(dSolDims);
			cudaFree(dSolCOM);
			cudaFree(dOsolDims);
			cudaFree(dOsolCOM);
			
			return -1000 - (int)err;
		}

		if(pStop && *pStop)
			break;
		endAll = clock();
		printf("CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
			double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);
		progfunc(progargs, progmin + (progmax - progmin) * ((i + 1) / double(PARTS)));
	}


	//err = cudaMemcpy(outData, dData, sizeof(double2) * voxels, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(outData, dData, sizeof(double) * voxels * 2, cudaMemcpyDeviceToHost);
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);
	
	cudaFree(dLoc);
	cudaFree(dIonInd);
	cudaFree(dData);
	cudaFree(dIdx);
	cudaFree(dCoeffs);
	cudaFree(dAtmInd);
	cudaFree(dAtmRadii);
	cudaFree(dSolDims);
	cudaFree(dSolCOM);
	cudaFree(dOsolDims);
	cudaFree(dOsolCOM);

		
	if ( cudaSuccess != err )
		return -1000 - (int)err;

#endif

	return 0;
}
