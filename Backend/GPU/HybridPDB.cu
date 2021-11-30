#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "GPUHybridCalc.cuh"
#include "CommonJacobGPUMethods.cu"
#include "CalculateJacobianSplines.cuh"

#include "CommonPDB.cuh"
#include "CommonCUDA.cuh"

#include <cuda_runtime.h>

#include <time.h>


/// Enough room to hold the coefficients for 10 ions. For more ions, do multiple copies.
__device__ __constant__ float affCoeffs[90];

#define THREADXDIM 256

///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of both the Atomic form factors and dummy atom
//  solvent contributions using the JacobianSphere grid
template <typename resFType>
__global__ void CombinedPDBandDummyAtomJacobForSingleIonKernel(const int offset, const float stepSize,
					const int thetaDivs, const int phiDivs, resFType *data, const float4 *gLoc,
					// For atomic form factors
					float *atmff,
					int firstAtomInd, const int lastAtomInd,
					const long long voxels)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x) + offset;

	float q;
	float qx, qy, qz;
	int qlayer;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &qlayer);

	resFType resR = 0.0;
	resFType resI = 0.0;

	resFType sn, cs;
	#define INNER_LOOP_SIZE THREADXDIM
	__shared__ float4 sLoc[INNER_LOOP_SIZE];

	while(firstAtomInd < lastAtomInd) {
		// Load locations into shared memory
		if(threadIdx.x < lastAtomInd - firstAtomInd) {
			sLoc[threadIdx.x] = gLoc[firstAtomInd + threadIdx.x];
		}	// threadIdx.x < INNER_LOOP_SIZE
		__syncthreads();

		int loopEnd = min(INNER_LOOP_SIZE, (lastAtomInd - firstAtomInd));

		for(int j = 0; j < loopEnd; j++)
		{
 			resFType phase = qx * sLoc[j].x + qy * sLoc[j].y + qz * sLoc[j].z;
			sincos(phase, &sn, &cs);

			resR += cs;
			resI += sn;
		}
		
		// IMPORTANT!!! Otherwise, the first threads to finish will modify the shared memory
		// for the for the subsequent threads!!!!
		__syncthreads();
		
		firstAtomInd += loopEnd;
	}
	if(id >= voxels)
		return;

	id *= 2;

	resR *= atmff[qlayer];
	resI *= atmff[qlayer];
	
	data[id  ] += resR;
	data[id+1] += resI;

}


///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of voxel based solvent contributions using the 
//  JacobianSphere grid
template <typename fType, typename resFType>
__global__ void CalcPDBVoxelSolventKernelJacobSphr(const int offset, const fType stepSize,
					const int thetaDivs, const int phiDivs, resFType *data,
					long long numSolVoxels, const fType voxelStep,
					/*voxel dimensions*/
					int4 *sVo,
					/*voxel COMs*/
					float4 *sCOM,
					const fType solED, const long long voxels)
{
	int tid = threadIdx.x;
	//int block = blockDim.x;
	long long id = (blockIdx.x * blockDim.x + tid) + offset;
	if(id >= voxels)
		return;
	
	fType q;
	fType qx, qy, qz;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz);
	
	id *= 2;

	resFType resR = data[id  ];
	resFType resI = data[id+1];

	fType sn, cs;

	fType va = 0.0, phase = 0.0, prevVAX = 0.0;
	int4 vDim;
	float4 vc;
	int prevVDX = 0;

	for(long long i = 0; i < numSolVoxels; i++) {
		vDim = sVo[i];
		vc = sCOM[i];
		
		if(vDim.x != prevVDX) {
			fType vdx = fType(vDim.x);
			prevVDX = vDim.x;
			if(qx == 0.0)
				prevVAX = voxelStep * vdx;
			else
				prevVAX = sin(qx * vdx * voxelStep / 2.0) / (qx * vdx * voxelStep / 2.0) * voxelStep * vdx;
		}
		va = prevVAX;

		fType vdy = fType(vDim.y);
		if(qy == 0.0)
			va *= voxelStep * vdy;
		else
			va *= sin(qy * vdy * voxelStep / 2.0) / (qy * vdy * voxelStep / 2.0) * voxelStep * vdy;
		
		fType vdz = fType(vDim.z);
		if(qz == 0.0)
			va *= voxelStep * vdz;
		else
			va *= sin(qz * vdz * voxelStep / 2.0) / (qz * vdz * voxelStep / 2.0) * voxelStep * vdz;


		va *= solED;

		phase = (qx * vc.x + qy * vc.y + qz * vc.z);

		sincos(phase, &sn, &cs);
		resR += va * cs;
		resI += va * sn;
	}

	if (id == 0)
		printf(""); // For some reason, without this, the results never make it to `*data`. WTF?!?!?
	data[id  ] = resR;
 	data[id+1] = resI;
}
// End of CalcPDBKernelJacobSphr
///////////////////////////////////////////////////////////////////////////////

#include <fstream>
bool GPUHybrid_PDBAmplitude(GridWorkspace& work)
{
	cudaError err = cudaSuccess;
	long long voxels = work.totalsz/2;

	int outerLayer;
	float dummyFloat, q;
	const int maxThreadsPerBlock = THREADXDIM/*devProp.maxThreadsPerBlock / 4*/;
	const long long N = (voxels / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);
	clock_t beg, endKern;

	beg = clock();

	GetQVectorFromIndex(voxels, work.thetaDivs, work.phiDivs, work.stepSize,
						&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);

	if(work.kernelComb & 0x01 || work.kernelComb & 0x02) { // Run aff based kernel
		int curIonInd, lastIonInd = 0;

		int firstIonPos = 0, lastIonPos = 0;
	
		// Make sure Atomic form factors finished calculating
		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));

		// Run kernel(s)
		for (curIonInd = lastIonInd; curIonInd < work.numUniqueIons; curIonInd++) {
			lastIonPos += work.atomsPerIon[curIonInd];

			// Break down into smaller kernels to try and minimize hitting watchdog timers
			int totalNumberofIonType = work.atomsPerIon[curIonInd];
			int offset_stepsize = std::min(1000000.0 * 1024 * 8 / double(totalNumberofIonType), double(voxels));
				
			offset_stepsize = ((offset_stepsize + BLOCK_WIDTH - 1) / BLOCK_WIDTH) * BLOCK_WIDTH; // Nearest multiple of 32
				
			const long long N_insteps = (offset_stepsize / maxThreadsPerBlock) + 1;
			dim3 gridInsteps(N_insteps, 1, 1);
			for (int offset = 0; offset < voxels; offset += offset_stepsize)
			{
				TRACE_KERNEL("CombinedPDBandDummyAtomJacobForSingleIonKernel");
				CombinedPDBandDummyAtomJacobForSingleIonKernel<double> <<<gridInsteps, threads, 0, (cudaStream_t)work.computeStream >>>
					(offset, work.stepSize, work.thetaDivs, work.phiDivs, work.d_amp, work.d_pdbLocs, work.d_affs + (curIonInd * outerLayer),
					firstIonPos, lastIonPos, std::min(static_cast<long long>(offset + offset_stepsize), voxels));
			}
			firstIonPos = lastIonPos;
		} // for (curIonInd = lastIonInd...

	} // if(work.kernelComb & 0x01 || work.kernelComb & 0x02) {


	if (work.kernelComb & 0x04) { // Voxelized solvent
		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));

		int offset_stepsize = std::min(1.0e10 * 8 * 1024 / double(work.numSolVoxels), double(voxels));
		offset_stepsize = ((offset_stepsize + BLOCK_WIDTH - 1) / BLOCK_WIDTH) * BLOCK_WIDTH; // Nearest multiple of 32
		const long long N_insteps = (offset_stepsize / maxThreadsPerBlock) + 1;
		dim3 gridInsteps(N_insteps, 1, 1);

		for (int offset = 0; offset < voxels; offset += offset_stepsize)
		{
			TRACE_KERNEL("CalcPDBVoxelSolventKernelJacobSphr");
			CalcPDBVoxelSolventKernelJacobSphr<double, double> <<<gridInsteps, threads, 0, (cudaStream_t)work.computeStream >>>(
				offset, work.stepSize, work.thetaDivs, work.phiDivs, work.d_amp,
				work.numSolVoxels, work.voxSize,
				work.d_SolDims, work.d_SolCOM,
				-work.solventED, std::min(static_cast<long long>(offset + offset_stepsize), voxels));
		}
	}

	if (work.kernelComb & 0x08) { // Voxelized solvation layer
		CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));
		
		int offset_stepsize = std::min(1.0e10 * 8 * 1024 / double(work.numOSolVoxels), double(voxels));
		offset_stepsize = ((offset_stepsize + BLOCK_WIDTH - 1) / BLOCK_WIDTH) * BLOCK_WIDTH; // Nearest multiple of 32
		const long long N_insteps = (offset_stepsize / maxThreadsPerBlock) + 1;
		dim3 gridInsteps(N_insteps, 1, 1);

		for (int offset = 0; offset < voxels; offset += offset_stepsize)
		{
			TRACE_KERNEL("CalcPDBVoxelSolventKernelJacobSphr");
			CalcPDBVoxelSolventKernelJacobSphr<double, double> <<<gridInsteps, threads, 0, (cudaStream_t)work.computeStream >>>(
				offset, work.stepSize, work.thetaDivs, work.phiDivs, work.d_amp,
				work.numOSolVoxels, work.voxSize,
				work.d_OSolDims, work.d_OSolCOM,
				work.outSolED - work.solventED, std::min(static_cast<long long>(offset + offset_stepsize), voxels));
		}
	}

	// Allocate the memory for the interpolants
	CHKERR(cudaMalloc(&work.d_int, sizeof(double) * work.totalsz));

	CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));
#define XPERTHREAD 8
	int newGrid = 2 * int((voxels / maxThreadsPerBlock) / XPERTHREAD) + 1;
	if(work.scale != 1.0)
	{
		TRACE_KERNEL("ScaleKernel");
		ScaleKernel<double, XPERTHREAD><<< newGrid, maxThreadsPerBlock, 0, (cudaStream_t)work.computeStream >>>
			((double*)work.d_amp, double(work.scale), 2*voxels);
	}

#undef XPERTHREAD

	// Sync threads for last time
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.computeStream));
	endKern = clock();

	CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
		(work.thetaDivs, work.phiDivs, outerLayer-1, (double2*)work.d_amp, (double2*)work.d_int /*,
		*(cudaStream_t*)work.memoryStream, *(cudaStream_t*)work.computeStream*/);

	CHKERR(cudaDeviceSynchronize());
	printf("Hybrid PDB CUDA timing:\n\tKernel: %f seconds\n", double(endKern - beg)/CLOCKS_PER_SEC);

	return err == cudaSuccess;
}




bool GPUHybrid_SetPDB(GridWorkspace& work, const std::vector<float4>& atomLocs,
					  const std::vector<float>& atomicFormFactors_q_major_ordering,
//					  const std::vector<unsigned char>& ionInd,
//					  const std::vector<float>& coeffs,
//					  const std::vector<int>& atomsPerIon,
//					  int solventType,
//					  std::vector<unsigned char>& atmInd, std::vector<float>& atmRad, double solvED,
					  double solventRad, // For dummy atom solvent
					  float4 *solCOM, int4 *solDims, int solDimLen, float voxStep, // For voxel based solvent
					  float4 *outSolCOM, int4 *outSolDims, int outSolDimLen, float outerSolED // For outer solvent layer
					  )
{
    cudaError_t err = cudaSuccess;
	
	CHKERR(cudaGetLastError());

	work.numAtoms = int(atomLocs.size());
	work.numSolVoxels = solDimLen;
	work.numOSolVoxels = outSolDimLen;
	work.voxSize = voxStep;
	work.outSolED = outerSolED;
	work.numSolVoxels = solDimLen;
	work.numOSolVoxels = outSolDimLen;
	
	int numUnIons = int(work.numUniqueIons);
	work.numUniqueIons = numUnIons;

	work.kernelComb |= ((work.solventED != 0.0 && work.solventType != 4 && work.solventType > 0) ? 0x04 : 0x00);
	work.kernelComb |= ((work.solventED != outerSolED && outerSolED != 0.0 && solventRad > 0.0) ? 0x08 : 0x00);

	CHKERR(cudaSetDevice(work.gpuID));

	CHKERR(cudaMalloc(&work.d_amp,			sizeof(double) * work.totalsz));
	CHKERR(cudaMemset(work.d_amp,		 0, sizeof(double) * work.totalsz));

	CHKERR(cudaMalloc(&work.d_affs,			sizeof(float) * numUnIons * work.qLayers));
	CHKERR(cudaMalloc(&work.d_pdbLocs,		sizeof(float4) * work.numAtoms));
	
	if(work.kernelComb & 0x02) { // Dummy atom solvent
		CHKERR(cudaMalloc(&work.d_atmRad,		sizeof(float) * numUnIons));
	}
	if(work.kernelComb & 0x04) { // Voxelized solvent
		CHKERR(cudaMalloc(&work.d_SolCOM,	sizeof(float4) * solDimLen));
		CHKERR(cudaMalloc(&work.d_SolDims,	sizeof(int4  ) * solDimLen));
	}
	if(work.kernelComb & 0x08) { // Voxelized solvation layer
		CHKERR(cudaMalloc(&work.d_OSolCOM,	sizeof(float4) * outSolDimLen));
		CHKERR(cudaMalloc(&work.d_OSolDims,	sizeof(int4  ) * outSolDimLen));
	}

	CHKERR(cudaMemcpyAsync(work.d_affs, atomicFormFactors_q_major_ordering.data(), sizeof(float) * numUnIons * work.qLayers, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));

	if(!work.computeStream)
	{
		cudaStream_t calcstream;
		CHKERR(cudaStreamCreate(&calcstream));
		work.computeStream = calcstream;
	}

	dim3 dimGrid, dimBlock;
	dimBlock.x = BLOCK_WIDTH;
	dimBlock.y = 1;
	dimGrid.x = ((work.qLayers % BLOCK_WIDTH == 0) ? (work.qLayers / BLOCK_WIDTH) : 
					(work.qLayers / BLOCK_WIDTH + 1));
	dimGrid.y = 1;
	CHKERR(cudaStreamSynchronize((cudaStream_t)work.memoryStream));

#ifdef _DEBUG
	CHKERR(cudaStreamSynchronize( (cudaStream_t)work.computeStream ));
	TRACE_KERNEL("PrintFloatDeviceMemory");
	PrintFloatDeviceMemory<<<5, 32>>>(work.d_affs, 130);
	CHKERR(cudaDeviceSynchronize());

#endif


	CHKERR(cudaMemcpyAsync(work.d_pdbLocs,	atomLocs.data(), sizeof(float4) * work.numAtoms, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));

	if(work.kernelComb & 0x04) { // Voxelized solvent
		CHKERR(cudaMemcpyAsync(work.d_SolCOM, solCOM,	sizeof(float4) * solDimLen, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));
		CHKERR(cudaMemcpyAsync(work.d_SolDims, solDims,	sizeof(int4  ) * solDimLen, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));
	}
	if(work.kernelComb & 0x08) { // Voxelized solvation layer
		CHKERR(cudaMemcpyAsync(work.d_OSolCOM, outSolCOM,	sizeof(float4) * outSolDimLen, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));
		CHKERR(cudaMemcpyAsync(work.d_OSolDims, outSolDims,	sizeof(int4  ) * outSolDimLen, cudaMemcpyHostToDevice, (cudaStream_t)work.memoryStream));
	}

	CHKERR(cudaStreamSynchronize((cudaStream_t)work.memoryStream));
	return err == cudaSuccess;
}
