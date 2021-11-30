#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "GPUHybridCalc.cuh"
#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"

#include <cuda_runtime.h>

#include <time.h>

__device__ __constant__ float _GROTMAT_ [9];
__device__ __constant__ float _GTRANSV_ [3];


bool GPUHybrid_SetSymmetry(GridWorkspace& work, float4 *locs, float4 *rots, int numLocs)
{
	cudaError_t err = cudaSuccess;

	CHKERR(cudaGetLastError());

	CHKERR(cudaMalloc(&work.d_amp,			sizeof(double) * work.totalsz));
	CHKERR(cudaMemset(work.d_amp,		 0, sizeof(double) * work.totalsz));

	work.symSize = numLocs;
	CHKERR(cudaMalloc(&work.d_symLocs, sizeof(float4) * work.symSize));
	CHKERR(cudaMalloc(&work.d_symRots, sizeof(float4) * work.symSize));
	CHKERR(cudaMemcpy(work.d_symLocs,	locs, sizeof(float4) * work.symSize, cudaMemcpyHostToDevice));
	CHKERR(cudaMemcpy(work.d_symRots,	rots, sizeof(float4) * work.symSize, cudaMemcpyHostToDevice));

	if(!work.computeStream)
	{
		cudaStream_t calcstream;
		CHKERR(cudaStreamCreate(&calcstream));
		work.computeStream = calcstream;
	}

	return err == cudaSuccess;
}

__global__ void ManualSymmetryKernel(const float stepSize,
										const int thetaDivs, const int phiDivs,
										int numOfCopies, double2 *inAmp,
										double2 *ds, double2 *outAmp,
										float4 *locs, float4 *rots,
										const long long voxels)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x);

	if(id >= voxels)
		return;

	float q, qx, qy, qz;
	long long bot;
	int lqi;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &lqi, &bot);

	double2 res, tmp;
	res.x = 0.0;
	res.y = 0.0;

	// This is so the pointers point to the first element on the plane and not to the last 
	// element on the previous plane
	bot++;
	
	for(int i = 0; i < numOfCopies; i++) {
		double newTheta, newPhi;
		double csA, csB, csC;
		double snA, snB, snC;
		float4 rot = rots[i];
		sincos(rot.x, &snA, &csA);
		sincos(rot.y, &snB, &csB);
		sincos(rot.z, &snC, &csC);
		
 		float preRotX = qx, preRotY = qy, preRotZ = qz;
 		
// 		// Rotate the vector by the Manual Symmetry rotations first
 		float rMat[9];	// This is a prime target for shared memory...
 		rMat[0] = csB * csC;			rMat[1] = -csB*snC;				rMat[2] = snB;
 		rMat[3] = csC*snA*snB+csA*snC;	rMat[4] = csA*csC-snA*snB*snC;	rMat[5] = -csB*snA;
 		rMat[6] = snA*snC-csA*csC*snB;	rMat[7] = csC*snA+csA*snB*snC;	rMat[8] = csA*csB;

  		qx = rMat[0] * preRotX + rMat[3] * preRotY + rMat[6] * preRotZ;
  		qy = rMat[1] * preRotX + rMat[4] * preRotY + rMat[7] * preRotZ;
  		qz = rMat[2] * preRotX + rMat[5] * preRotY + rMat[8] * preRotZ;

		// Rotate the vector by the inner objects rotations last
 		float tmpx = qx, tmpy = qy, tmpz = qz;
  	 	qx = _GROTMAT_[0] * tmpx + _GROTMAT_[3] * tmpy + _GROTMAT_[6] * tmpz;
  	 	qy = _GROTMAT_[1] * tmpx + _GROTMAT_[4] * tmpy + _GROTMAT_[7] * tmpz;
  	 	qz = _GROTMAT_[2] * tmpx + _GROTMAT_[5] * tmpy + _GROTMAT_[8] * tmpz;


		// Convert to polar
		
		if (fabs(qz) > q) q = fabs(qz);
		newTheta = acos(qz / q);
		newPhi	 = atan2f(qy, qx);
/*
		if(newTheta != newTheta && id != 0) {
			printf("[%lld, %d] --> %f\n", id, i, qz / q);
		}
*/
		qx = preRotX; qy = preRotY; qz = preRotZ;

		if(newPhi < 0.0)
 			newPhi += M_2PI;

		double2 tmpRes;
		if(id == 0) {
			tmpRes = inAmp[0];
		} else {
			tmpRes = GetAmpAtPointInPlaneJacob<double, double2, double2>
				(lqi, newTheta, newPhi, thetaDivs, phiDivs, (double*)(inAmp+bot), ds+(bot));
		}


		sincos(tmpx * _GTRANSV_[0] + tmpy * _GTRANSV_[1] + tmpz * _GTRANSV_[2], &snA, &csA);

 		tmp.x = tmpRes.x * csA - tmpRes.y * snA;
 		tmp.y = tmpRes.y * csA + tmpRes.x * snA;

		float4 loc = locs[i];

 		sincos(preRotX * loc.x + preRotY * loc.y + preRotZ * loc.z, &snA, &csA);
 		res.x += tmp.x * csA - tmp.y * snA;
 		res.y += tmp.y * csA + tmp.x * snA;
/*
		if(
			(res.x != res.x || res.y != res.y)
			)
		{
			printf("[%lld, %d] {%d, %f, %f} = (%f, %f)--> {%f, %f}\n",
				id, i,
				lqi, newTheta, newPhi,
				tmpRes.x, tmpRes.y, res.x, res.y);
		}
*/
	}

/*
	if(res.x != res.x ||
		res.y != res.y)
	{
		printf("[%lld] --> {%f, %f}\n", id, res.x, res.y);
	}
*/

	outAmp[id].x += res.x;
	outAmp[id].y += res.y;

}

bool GPUHybrid_ManSymmetryAmplitude(GridWorkspace& work, GridWorkspace& child, float4 trans, float4 rot)
{
	cudaError err = cudaSuccess;
	long long voxels = work.totalsz/2;
	int outerLayer;
	float dummyFloat, q;

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, work.gpuID);

	const int maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const int N = int(voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);

	GetQVectorFromIndex(voxels, work.thetaDivs, work.phiDivs, work.stepSize, 
						&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);

	float snA, snB, snC, csA, csB, csC;
	snA = sin(rot.x); snB = sin(rot.y); snC = sin(rot.z);
	csA = cos(rot.x); csB = cos(rot.y); csC = cos(rot.z);
	float rotmat[9];
	rotmat[0] = csB * csC;				rotmat[1] = -csB*snC;				rotmat[2] = snB;
	rotmat[3] = csC*snA*snB+csA*snC;	rotmat[4] = csA*csC-snA*snB*snC;	rotmat[5] = -csB*snA;
	rotmat[6] = snA*snC-csA*csC*snB;	rotmat[7] = csC*snA+csA*snB*snC;	rotmat[8] = csA*csB;


	CHKERR(cudaMemcpyToSymbol(_GROTMAT_,	&rotmat[0],	sizeof(float) * 9));
	CHKERR(cudaMemcpyToSymbol(_GTRANSV_,	&trans,		sizeof(float) * 3));

	CHKERR(cudaStreamSynchronize((cudaStream_t)work.memoryStream));
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));


	TRACE_KERNEL("ManualSymmetryKernel");
	ManualSymmetryKernel<<<grid, threads, 0, cudaStream_t(work.computeStream)>>>
		(work.stepSize, work.thetaDivs, work.phiDivs, work.symSize, (double2*)child.d_amp,
		(double2*)child.d_int, (double2*)work.d_amp, work.d_symLocs, work.d_symRots, voxels);

	// Sync threads for last time
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));

	return err == cudaSuccess;
}

bool GPUSemiHybrid_ManSymmetryAmplitude(GridWorkspace& work, double* amp, float4 trans, float4 rot)
{
	cudaError err = cudaSuccess;
	int voxels = int(work.totalsz/2);
	int outerLayer;
	float dummyFloat, q;

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, work.gpuID);

	const int maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const int N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);

	GetQVectorFromIndex(voxels, work.thetaDivs, work.phiDivs, work.stepSize, 
						&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);


	if(amp == NULL)	// caluclate splines
	{
		// Allocate the memory for the interpolants
		CHKERR(cudaMalloc(&work.d_int, sizeof(double) * work.totalsz));

		// Scale first
		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));
#define XPERTHREAD 8
		int newGrid = 2 * ((voxels / maxThreadsPerBlock) / XPERTHREAD) + 1;
		if(work.scale != 1.0)
		{
			TRACE_KERNEL("ScaleKernel");
			ScaleKernel<double, XPERTHREAD><<< newGrid, maxThreadsPerBlock, 0, (cudaStream_t)work.computeStream >>>
				((double*)work.d_amp, double(work.scale), 2*voxels);
		}
#undef XPERTHREAD

		// Sync threads for last time
		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));

		CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
			(work.thetaDivs, work.phiDivs, outerLayer-1, (double2*)work.d_amp, (double2*)work.d_int /*,
			*(cudaStream_t*)work.memoryStream, *(cudaStream_t*)work.computeStream*/);

		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));
		return err == cudaSuccess;
	}

	// Create a new temporary child workspace
	GridWorkspace child;
	CHKERR(cudaMalloc(&child.d_amp, sizeof(double) * work.totalsz));	
	CHKERR(cudaMemcpy(child.d_amp, amp, sizeof(double) * work.totalsz, cudaMemcpyHostToDevice));	
	// Allocate the memory for the interpolants
	CHKERR(cudaMalloc(&child.d_int, sizeof(double) * work.totalsz));

	CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
		(work.thetaDivs, work.phiDivs, outerLayer-1, (double2*)child.d_amp, (double2*)child.d_int /*,
		*(cudaStream_t*)work.memoryStream, *(cudaStream_t*)work.computeStream*/);

	bool res = GPUHybrid_ManSymmetryAmplitude(work, child, trans, rot);

	{if(child.d_amp) CHKERR(cudaFree(child.d_amp)); child.d_amp = NULL;}
	{if(child.d_int) CHKERR(cudaFree(child.d_int)); child.d_int = NULL;}


	return res && err == cudaSuccess;
}

