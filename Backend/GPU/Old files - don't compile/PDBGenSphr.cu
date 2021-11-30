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

// Allocates more memory and performs more computations in favor of cancelling the "if"
// SLOWER, hmmph...
//#define ADDITIONAL_ALLOC

#define CONST_COEFFS
//#define SVG_SOL_ONLY
// Constant memory --> 209 * 9 coefficients
__device__ __constant__ double _GCOEFFS_ [1881];


///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of voxel solvent
__global__ void CalcPDBVoxelSolventKernel(const int offset, unsigned short dimx, const double stepSize,
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

//  	unsigned short thetaInd  = in1->w;
//  	unsigned short maxTheta  = in1->z;
//  	unsigned short phiIndex  = in1->y;
//  
//  	unsigned short maxPhi = in1->x;
//  	unsigned short rIndex = in2->x;

	// Can we think of a way to do this NOT cartesean?
 	double qx, qy, qz, q = double(in2->x) * stepSize;
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

	double va = 0.0, phase = 0.0, prevVAX = 0.0, prevVAY = 0.0, prevVAZ = 0.0;
	ulonglong4 vDims;
	double4 vCOM;
	u64 prevVDX = 0;
	u64 prevVDY = 0;
	u64 prevVDZ = 0;
	double sn, cs;

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
				prevVAX = sin(qx * vdx * voxelStep / 2.0) / (qx * vdx * voxelStep / 2.0) * voxelStep * vdx;
		}
		if(vDims.y != prevVDY) {
			double vdy = double(vDims.y);
			prevVDY = vDims.y;
			if(qy == 0.0 || vDims.y == 0)
				prevVAY = voxelStep * vdy;
			else
				prevVAY = sin(qy * vdy * voxelStep / 2.0) / (qy * vdy * voxelStep / 2.0) * voxelStep * vdy;
		}
		if(vDims.z != prevVDZ) {
			double vdz = double(vDims.z);
			prevVDZ = vDims.z;
			if(qz == 0.0 || vDims.z == 0.0)
				prevVAZ = voxelStep * vdz;
			else
				prevVAZ = sin(qz * vdz * voxelStep / 2.0) / (qz * vdz * voxelStep / 2.0) * voxelStep * vdz;
		}

		va = prevVAX * prevVAY * prevVAZ * solED;

		phase = (qx * vcx + qy * vcy + qz * vcz);

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
__global__ void CalcPDBDummyAtomSolventKernel(const int offset, unsigned short dimx, const double stepSize,
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

	// Can we think of a way to do this NOT cartesean?
 	double qx, qy, qz, q = double(in2->x) * stepSize;
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
	double gi = 0.0;
	u8 prevAtm = 255;
	
	q = q * q / 4.0;	// The only place it's used from here on needs the value (q/2)^2
	for(int i = 0; i < numAtoms; i++) {
		loc = locs[i];
		double phase = qx * loc.x + qy * loc.y + qz * loc.z;
		if(prevAtm != atmInd[i]) {
			prevAtm = atmInd[i];
			double rad = radii[prevAtm];
		
	 		gi = /*5.56832799683*/-4.1887902047863909846 /*4\pi/3*/ /*5.5683279968317084528 pi^1.5*/ * rad * rad * rad * exp( -(rad * rad * q) ) * solED;
		}

		double sn, cs;
		sincos(phase, &sn, &cs);

 		res.x += gi * cs;
 		res.y += gi * sn;
	}

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
void CalcPDBKernelSphr(const int offset, unsigned short dimx, const double stepSize,
				   double *data, const double *x, const double *y, const double *z,
				   const u8 *ionInd, const int numAtoms, const double *coeffs, const u8 numCoeffs,
				   const unsigned long long totalDoubles, unsigned long long idHalf)
{	
	idHalf += offset;
	unsigned long long id = idHalf * 2;
#else
__global__ void CalcPDBKernelSphr(const int offset, unsigned short dimx, const double stepSize,
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
 
	// Input (in hex form)	// Numbers are indices and max indices
	// real part = [ THTH DYDY PHPH DZDZ ]
	// imag part = [ 0000 0000 0000 RRRR ]
	// Input (in hex form)
	// REAL = [ XXXX YYYY ZZZZ DYDY ]
	// IMAG = [ 0000 0000 0000 DZDZ ]

	double2 input = idx[id];
	double4 coord;
	ushort4 *in1 = (ushort4 *)&(input.x), *in2 = (ushort4 *)&(input.y);

	// TODO::Change when moving to sphereGrid
//  	unsigned short thetaInd  = in1->w;
//  	unsigned short maxTheta  = in1->z;
//  	unsigned short phiIndex  = in1->y;
//  
//  	unsigned short maxPhi = in1->x;
//  	unsigned short rIndex = in2->x;

	// Can we think of a way to do this NOT cartesean?
 	double qx, qy, qz, q = double(in2->x) * stepSize;
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
 	double aff = 0.0;
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
GPUCalcPDBSphr(u64 voxels, unsigned short dimx, double qmax, unsigned short sections, double *outData,
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
		CalcPDBKernelSphr(0, dimx, stepSize, outData, 
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
 			CalcPDBKernelSphr<<<grid, threads/*, usedSharedMemory*/>>>(
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
			CalcPDBDummyAtomSolventKernel <<<grid, threads/*, usedSharedMemory*/ >>>(
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
		if(solvED > 0.0 && solDimLen == 4)
			CalcPDBVoxelSolventKernel<<<grid, threads/*, usedSharedMemory*/>>>(
										 0/*i * voxelsPerPart*/, dimx, stepSize, 
										 dIdx, dData, solDimLen, voxStep,
 										 dSolDims, dSolCOM, -solvED, (voxels * 2) - 1);

		// Voxel Based Outer Solvent 
		if(outerSolED > 0.0 && outSolDimLen > 4)
			CalcPDBVoxelSolventKernel<<<grid, threads/*, usedSharedMemory*/>>>(
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
		if(progfunc && progargs)
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
	cudaDeviceReset();	// For profiling
	return 0;
}
	

int GetNumGPUs() {
	int num = -1;
	cudaGetDeviceCount(&num);

	return num;
}

void ResetGPU() {
	cudaDeviceReset();
}


/************************************************************************/
/* Debye formula kernel<s>                                              */
/************************************************************************/
template<typename FT>
__global__ void CalcDebyeKernel(int offset, int qValues, FT *qVals, FT *outData,
	FT *loc, u8 *ionInd, int numAtoms, u8 nAtmTypes, u8 *atmTypes, int numCoeffs,
	bool bSol, FT *rad, FT solvED)
{
	int tid = threadIdx.x;
	//int block = blockDim.x;
	unsigned long long id = (blockIdx.x * blockDim.x + tid) + offset;

	if(id >= qValues)
		return;

	FT qq = qVals[id];
#ifdef _DEBUG
	if(id == qValues/2) {
		printf("\nqVals[%llu] = %f", id, qVals[id]);
	}
#endif

	FT res = 0.0;
 	FT q10 = qq / 10.0;
 	// The number is (4pi)^2
 	FT sqq = q10 * q10 / (157.913670417429737901351855998);

#ifdef _DEBUG
	if(id == qValues/2) {
		printf("\nsqq = %f", sqq);
	}
#endif


	// Precalculate according to type, i.e for a PDB with just C and N there should be 2 values
	FT atmAmps[10];	// BAD!! DEBUG!! TODO!! I don't want to dynamically allocate, but I don't want 209 either
	FT *_GLOB_ = (FT*)_GCOEFFS_;

#ifdef _DEBUG
	if(id == qValues/2) {
		int atI = atmTypes[0];
		printf("\natI = %d", atI);
		printf("\n_GLOB_[atI * 9 + 0] = %f", _GLOB_[atI * 9 + 0]);
		printf("\n_GLOB_[atI * 9 + 1] = %f", _GLOB_[atI * 9 + 1]);
		printf("\n_GLOB_[atI * 9 + 2] = %f", _GLOB_[atI * 9 + 2]);
		printf("\n_GLOB_[atI * 9 + 3] = %f", _GLOB_[atI * 9 + 3]);
		printf("\n_GLOB_[atI * 9 + 4] = %f", _GLOB_[atI * 9 + 4]);
		printf("\n_GLOB_[atI * 9 + 5] = %f", _GLOB_[atI * 9 + 5]);
		printf("\n_GLOB_[atI * 9 + 6] = %f", _GLOB_[atI * 9 + 6]);
		printf("\n_GLOB_[atI * 9 + 7] = %f", _GLOB_[atI * 9 + 7]);
		printf("\n_GLOB_[atI * 9 + 8] = %f", _GLOB_[atI * 9 + 8]);
		printf("\n_GLOB_[atI * 9 + 9] = %f", _GLOB_[atI * 9 + 9]);
	}
#endif

	for(int ii = 0; ii < nAtmTypes; ii++) {
		int atI = atmTypes[ii] * 9;
		atmAmps[ii] = _GLOB_[atI + 0] * exp(-sqq * _GLOB_[atI + 1])
					+ _GLOB_[atI + 2] * exp(-sqq * _GLOB_[atI + 3])
					+ _GLOB_[atI + 4] * exp(-sqq * _GLOB_[atI + 5])
					+ _GLOB_[atI + 6] * exp(-sqq * _GLOB_[atI + 7])
					+ _GLOB_[atI + 8];
#ifdef _DEBUG
	if(id == qValues/2) {
		printf("\natmAmps[%d] = %f", ii, atmAmps[ii]);
	}
#endif

	}	// for ii
	
#ifdef _DEBUG
	if(id == qValues/2) {
		printf("\nloc[0] = {%f, %f, %f}", loc[0], loc[1], loc[2]);
		printf("\nloc[0] = {%f, %f, %f}\n", loc[3], loc[4], loc[5]);
		printf("\nnumAtoms = %d\n", numAtoms);
	}
#endif

	// Calculate the 
	for(int ii = 0; ii < numAtoms; ii++) {
		FT aaii = atmAmps[ionInd[ii]];
 #ifdef _DEBUG
 	if(id == qValues/2) {
 		printf("ionInd[%d] = %d\n", (int)ii, (int)ionInd[ii]);
 	}
 #endif
		for(int jj = 0; jj < ii; jj++) {
			FT dx, dy, dz;
			dx = loc[ii * 3 + 0] - loc[jj * 3 + 0]; 
			dy = loc[ii * 3 + 1] - loc[jj * 3 + 1]; 
			dz = loc[ii * 3 + 2] - loc[jj * 3 + 2]; 
			FT qr = qq * sqrt(dx*dx + dy*dy + dz*dz);
			if(qr != 0.0)
				res += 2.0 * aaii * atmAmps[ionInd[jj]] * sin(qr) / qr;
			else
				res += 2.0 * aaii * atmAmps[ionInd[jj]];
		}	// for jj
		res += aaii * aaii;
#ifdef _DEBUG
		if(id == qValues/2) {
			printf("GPU: aaii = %f\n", aaii);
			printf("\tres = %f\n", res);
		}
#endif
	}	// for ii

#ifdef _DEBUG
	if(id == qValues/2) {
		printf("\n");
	}
#endif

	outData[id] = res;
}	// CalcDebyeKernel

template <typename FLOAT_TYPE>
int GPUCalcDebyeTempl(int qValues, FLOAT_TYPE *qVals, FLOAT_TYPE *outData,
	FLOAT_TYPE *loc, u8 *ionInd, int numAtoms, FLOAT_TYPE *coeffs, int numCoeffs, bool bSol,
	FLOAT_TYPE *rad, FLOAT_TYPE solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	// TODO (A LOT LATER)::OneStepAhead: Multi-GPU support?
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;
	clock_t begMem, endMem, endKern, endAll;
	begMem = clock();

#ifdef CONST_COEFFS
	err = cudaMemcpyToSymbol(_GCOEFFS_, coeffs, sizeof(FLOAT_TYPE) * numCoeffs);
	if(err != cudaSuccess) {
		printf("\nERROR! Error allocating global memory (1)");
		return -3000 - err;
	}
#else
	printf("\nERROR! Kernel needs CONST_COEFFS to be defined");
	return -5678;
#endif

	//const int PARTS = 1;
	const u16 maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const u64 N = (qValues / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);

    FLOAT_TYPE *dQVals, *dOutData, *dLoc, *dRad;
	u8 *dIonInd, *dDifIonTypes;
	
	// Find and consolidate unique ion indices
	u8 *newIonInds = new u8[numAtoms];
	u8 numUIons = 1;
	for(int i = 1; i < numAtoms; i++) {
		if(ionInd[i] != ionInd[i-1])
			numUIons++;
	} // for
	printf("\nnumUIons = %d\n",  (int)numUIons);
	u8 *hUIons = new u8[numUIons];
	u8 uInd = 0;
	newIonInds[uInd] = uInd;
	hUIons[uInd] = ionInd[0];
	for(int i = 1; i < numAtoms; i++) {
		if(ionInd[i] != ionInd[i-1])
			hUIons[++uInd] = ionInd[i];
		newIonInds[i] = uInd;
	} // for

	////////////////////////////////////////////////////////////////
	// Mallocs
	err = cudaMalloc(&dQVals,		sizeof(FLOAT_TYPE) * qValues);
	err = cudaMalloc(&dOutData,		sizeof(FLOAT_TYPE) * qValues);
	err = cudaMalloc(&dRad,			sizeof(FLOAT_TYPE) * numAtoms);
	err = cudaMalloc(&dIonInd,		sizeof(u8) * numAtoms);
	err = cudaMalloc(&dDifIonTypes,	sizeof(u8) * numUIons);
	err = cudaMalloc(&dLoc,			sizeof(FLOAT_TYPE) * 3 * numAtoms);
	if ( cudaSuccess != err ) {
		printf("Error allocating memory on device: %d", (int)err);
		cudaFree(dQVals);
		cudaFree(dOutData);
		cudaFree(dIonInd);
		cudaFree(dDifIonTypes);
		cudaFree(dLoc);
		cudaFree(dRad);
		delete newIonInds;	newIonInds = NULL;
		delete hUIons;		hUIons = NULL;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Memcpys
	err = cudaMemcpy(dQVals,		qVals,		sizeof(FLOAT_TYPE) * qValues,		cudaMemcpyHostToDevice);
	err = cudaMemset(dOutData,		0,			sizeof(FLOAT_TYPE) * qValues							  );
	err = cudaMemcpy(dIonInd,		newIonInds,	sizeof(u8) * numAtoms,				cudaMemcpyHostToDevice);
	err = cudaMemcpy(dDifIonTypes,	hUIons,		sizeof(u8) * numUIons,				cudaMemcpyHostToDevice);
	err = cudaMemcpy(dLoc,			loc,		sizeof(FLOAT_TYPE) * 3 * numAtoms,	cudaMemcpyHostToDevice);

	delete newIonInds;	newIonInds = NULL;
	delete hUIons;		hUIons = NULL;

	if ( cudaSuccess != err ) {
		printf("Error copying data to device: %d", (int)err);
		cudaFree(dQVals);
		cudaFree(dOutData);
		cudaFree(dIonInd);
		cudaFree(dDifIonTypes);
		cudaFree(dLoc);
		cudaFree(dRad);
		return -3000 - err;
	}

	endMem = clock();

	////////////////////////////////////////////////////////////////
	// Run the kernel
	CalcDebyeKernel<FLOAT_TYPE><<< grid, threads >>>
		(0, qValues, dQVals, dOutData, dLoc, dIonInd, numAtoms, numUIons, dDifIonTypes,
			numCoeffs,
			false, NULL, 0.0f	// These need to be changed when implementing the solvent...
		);
	// Check for launch errors
	err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		printf("Launch error: %d", (int)err);
		cudaFree(dQVals);
		cudaFree(dOutData);
		cudaFree(dIonInd);
		cudaFree(dDifIonTypes);
		cudaFree(dLoc);
		cudaFree(dRad);
		return -3000 - err;
	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if ( cudaSuccess != err )
		printf("Error in kernel: %d\n", err);

	endKern = clock();

	err = cudaMemcpy(outData, dOutData, sizeof(FLOAT_TYPE) * qValues, cudaMemcpyDeviceToHost);
	if ( cudaSuccess != err )
		printf("Bad cudaMemcpy from device: %d\n", err);
	cudaFree(dQVals);
	cudaFree(dOutData);
	cudaFree(dIonInd);
	cudaFree(dDifIonTypes);
	cudaFree(dLoc);
	cudaFree(dRad);

	endAll = clock();

	printf("CUDA timing:\n\tMemory allocation and copying: %f seconds\n\tKernel: %f seconds\n\tMemory copying and freeing: %f seconds\n",
		double(endMem - begMem)/CLOCKS_PER_SEC, double(endKern - endMem)/CLOCKS_PER_SEC, double(endAll - endKern)/CLOCKS_PER_SEC);
	
	if(progfunc && progargs)
		progfunc(progargs, progmin + (progmax - progmin));

	if ( cudaSuccess != err )
		return -1000 - (int)err;


	return 0;
}
////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////
/*extern "C" */int
GPUCalcDebye(int qValues, float *qVals, float *outData,
	float *loc, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSol,
	float *rad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	return GPUCalcDebyeTempl<float>(qValues, qVals, outData, loc, ionInd, numAtoms,
					coeffs, numCoeffs, bSol, rad, solvED,
					progfunc, progargs, progmin, progmax, pStop);

}
/*extern "C" */int
GPUCalcDebye(int qValues, double *qVals, double *outData,
	double *loc, u8 *ionInd, int numAtoms, double *coeffs, int numCoeffs, bool bSol,
	double *rad, double solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	return GPUCalcDebyeTempl<double>(qValues, qVals, outData, loc, ionInd, numAtoms,
					coeffs, numCoeffs, bSol, rad, solvED,
					progfunc, progargs, progmin, progmax, pStop);

}
