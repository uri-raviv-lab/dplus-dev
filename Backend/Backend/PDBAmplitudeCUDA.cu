#include "PDBAmplitudeCUDA.h"

#include "../GPU/CommonJacobGPUMethods.cu"

#include <cuda_runtime.h>

#include "maps/maps.cuh"

#define THREADXDIM 256


///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of voxel based solvent contributions using the 
//  JacobianSphere grid
template <typename fType, typename resFType>
__global__ void CalcPDBVoxelSolventKernelJacobSphr(const int offset, const fType stepSize,
	const int thetaDivs, const int phiDivs, resFType *data,
	u64 numSolVoxels, const fType voxelStep,
	/*voxel dimensions*/
	int4 *sVo,
	/*voxel COMs*/
	float4 *sCOM,
	const fType solED, const long long voxels)
{
	int tid = threadIdx.x;
	//int block = blockDim.x;
	long long id = (blockIdx.x * blockDim.x + tid) + offset;
	if (id >= voxels)
		return;

	fType q;
	fType qx, qy, qz;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz);

	id *= 2;

	resFType resR = 0;
	resFType resI = 0;

	fType sn, cs;

	fType va = 0.0, phase = 0.0, prevVAX = 0.0;
	int4 vDim;
	float4 vc;
	int prevVDX = 0;

	for (long long i = 0; i < numSolVoxels; i++) {
		vDim = sVo[i];
		vc = sCOM[i];

		if (vDim.x != prevVDX) {
			fType vdx = fType(vDim.x);
			prevVDX = vDim.x;
			if (qx == 0.0 || vDim.x == 0)
				prevVAX = voxelStep * vdx;
			else
				prevVAX = sin(qx * vdx * voxelStep / 2.0) / (qx * vdx * voxelStep / 2.0) * voxelStep * vdx;
		}
		va = prevVAX;

		fType vdy = fType(vDim.y);
		if (qy == 0.0 || vDim.y == 0)
			va *= voxelStep * vdy;
		else
			va *= sin(qy * vdy * voxelStep / 2.0) / (qy * vdy * voxelStep / 2.0) * voxelStep * vdy;

		fType vdz = fType(vDim.z);
		if (qz == 0.0 || vDim.z == 0.0)
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
	{
		printf(""); // For some reason, the calculation is not used on my computer without this pointless line...
	}
	
	data[id] += resR;
	data[id + 1] += resI;
}
// End of CalcPDBKernelJacobSphr
///////////////////////////////////////////////////////////////////////////////

template <typename resFType>
__global__ void AddAnomalousContributionKernel(const float stepSize,
	const int thetaDivs, const int phiDivs, resFType *data,
	const float4 * __restrict__ gLoc, const float2 * __restrict__ gAnomFactors,
	const int numAnomAtoms
	)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x);

	float q;
	float qx, qy, qz;
	int qlayer;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &qlayer);

	resFType resR = 0.0;
	resFType resI = 0.0;

	float sn, cs, phase;

	for (int j = 0; j < numAnomAtoms; j++) {
		phase = qx * gLoc[j].x + qy * gLoc[j].y + qz * gLoc[j].z;

		sincos(phase, &sn, &cs);
		resR += (cs * gAnomFactors[j].x - sn * gAnomFactors[j].y);
		resI += (cs * gAnomFactors[j].y + sn * gAnomFactors[j].x);
	}

	id *= 2;

	data[id] += resR;
	data[id + 1] += resI;

}

///////////////////////////////////////////////////////////////////////////////
// Kernel for calculation of both the Atomic form factors and dummy atom solvent
//  contributions using the JacobianSphere grid
template <typename resFType, typename AFF_TYPE, int INNER_LOOP_SIZE>
__global__ void CombinedPDBandDummyAtomJacobForSingleIonKernel(const float stepSize,
	const int thetaDivs, const int phiDivs, resFType *data, const float4 * __restrict__ gLoc,
	// For atomic form factors
	int firstAtomInd, const int lastAtomInd, const AFF_TYPE* __restrict__ affs)
{
	long long id = (blockIdx.x * blockDim.x + threadIdx.x);

	float q;
	float qx, qy, qz;
	int qlayer;

	GetQVectorFromIndex(id, thetaDivs, phiDivs, stepSize, &q, &qx, &qy, &qz, &qlayer);

	resFType resR = 0.0;
	resFType resI = 0.0;

	float sn, cs, phase;

	__shared__ float4 sLoc[INNER_LOOP_SIZE];
	while (firstAtomInd < lastAtomInd)
	{
		// Load locations into shared memory
		if (threadIdx.x < lastAtomInd - firstAtomInd)
		{
			sLoc[threadIdx.x] = gLoc[firstAtomInd + threadIdx.x];
		}	// threadIdx.x < INNER_LOOP_SIZE
		__syncthreads();

		int loopEnd = min(INNER_LOOP_SIZE, (lastAtomInd - firstAtomInd));

		for (int j = 0; j < loopEnd; j++) {
			phase = qx * sLoc[j].x + qy * sLoc[j].y + qz * sLoc[j].z;

			sincos(phase, &sn, &cs);
			resR += cs;
			resI += sn;
		}

		// IMPORTANT!!! Otherwise, the first threads to finish will modify the shared memory
		// for the for the subsequent threads!!!!
		__syncthreads();

		firstAtomInd += loopEnd;
	}

	resR *= (affs[qlayer]); // Requires that the memory be layed out as "ionType major"
	resI *= (affs[qlayer]); // TODO: Rewrite to make complex multiplication an option

	id *= 2;

	data[id] += resR;
	data[id + 1] += resI;
}

#define FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY	\
	cudaFree(dOutData);								\
	cudaFree(dLoc);									\
	cudaFree(dAffs);								\
	cudaFree(dAnomLocs);							\
	cudaFree(dAnomFactors);							\
	cudaFree(dSolCOM);								\
	cudaFree(dOsolCOM);								\
	cudaFree(dSolDims);								\
	cudaFree(dOsolDims);							\
	cudaStreamDestroy(kernelStream);				\
	cudaStreamDestroy(memoryStream);



template <typename RES_FLOAT_TYPE, typename AFF_TYPE, int BLOCK_WIDTH_T = THREADXDIM>
int PDBJacobianGridAmplitudeCalculationCUDA(
	u64 voxels, int thDivs, int phDivs, float stepSize, RES_FLOAT_TYPE *outData,
	std::vector<float4> atomLocations,
	atomicFFCalculator &affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, float voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, float outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;

	size_t numAtoms = atomLocations.size();

	////////////////////////////
	// Device arrays
	RES_FLOAT_TYPE *dOutData;
	float4 *dLoc, *dAnomLocs = NULL;
	float2 *dAnomFactors = NULL;
	float *dAffs = NULL;

	float q, qx, qy, qz;
	int numQLayers;
	GetQVectorFromIndex(voxels-1, thDivs, phDivs, stepSize, &q, &qx, &qy, &qz, &numQLayers);
	std::vector<float> affMatrix;
	affMatrix.resize((numQLayers + 1) * affCalculator.GetNumUniqueIon());

	float4 *dSolCOM, *dOsolCOM;
	int4 *dSolDims, *dOsolDims;

	////////////////////////////
	// Streams
	cudaStream_t kernelStream, memoryStream;
	CHKERR(cudaStreamCreate(&kernelStream));
	CHKERR(cudaStreamCreate(&memoryStream));

	//const int PARTS = 1;
	const u16 maxThreadsPerBlock = BLOCK_WIDTH_T;
	int roundedNumberVoxels = maps::RoundUp((u64)(voxels), maxThreadsPerBlock) * maxThreadsPerBlock;
	int roundedNumAtoms = maps::RoundUp((u64)(numAtoms), maxThreadsPerBlock) * maxThreadsPerBlock;
	const unsigned int N = (roundedNumberVoxels / maxThreadsPerBlock);
	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);

	int kernJobs = (
		(affCalculator.HasSomethingToCalculate()) ? 1 : 0) +
		((solvED != 0.0 && !(affCalculator.GetBitCombination() & CALC_DUMMY_SOLVENT) && solDims.size() > 4) ? 1 : 0) +
		((outerSolED != 0.0 && outSolDims.size() > 4) ? 1 : 0);
	float kernFrac = 1.f / kernJobs;
	int kernDone = 0;

	// Find and consolidate unique ion and atom indices

	////////////////////////////////////////////////////////////////
	// Mallocs
	CHKERR(cudaMalloc(&dOutData, sizeof(RES_FLOAT_TYPE) * 2 * roundedNumberVoxels));
	CHKERR(cudaMalloc(&dLoc, sizeof(float4) * roundedNumAtoms));
	CHKERR(cudaMalloc(&dAffs, sizeof(float) * affMatrix.size()));

	CHKERR(cudaMalloc(&dSolCOM, sizeof(float4) * solDims.size()));
	CHKERR(cudaMalloc(&dSolDims, sizeof(int4) * solDims.size()));

	CHKERR(cudaMalloc(&dOsolDims, sizeof(float4) * outSolDims.size()));
	CHKERR(cudaMalloc(&dOsolCOM, sizeof(int4) * outSolDims.size()));
	
	
	if (cudaSuccess != err) {
		printf("Error allocating memory on device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Memcpys
	CHKERR(cudaMemsetAsync(dAffs, 0, sizeof(float) * affMatrix.size(), memoryStream));
	CHKERR(cudaMemsetAsync(dLoc, 0, sizeof(float4) * roundedNumAtoms, memoryStream));
	CHKERR(cudaMemsetAsync(dOutData, 0, sizeof(RES_FLOAT_TYPE) * 2 * roundedNumberVoxels, memoryStream));
	CHKERR(cudaMemcpyAsync(dLoc, atomLocations.data(), sizeof(float4) * numAtoms, cudaMemcpyHostToDevice, memoryStream));

	affCalculator.GetQMajorAFFMatrix(affMatrix.data(), numQLayers+1, stepSize);
	CHKERR(cudaMemcpyAsync(dAffs, affMatrix.data(), sizeof(float) * affMatrix.size(), cudaMemcpyHostToDevice, memoryStream));

	if (cudaSuccess != err) {
		printf("Error copying data to device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	CHKERR(cudaStreamSynchronize(memoryStream));

	int currentIonPosition = 0;
	for (size_t i = 0; i < affCalculator.GetNumUniqueIon(); i++)
	{
		CHKERR(cudaStreamSynchronize(kernelStream));

		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * (progmax - progmin) * (float(i) / float(affCalculator.GetNumUniqueIon())));

		TRACE_KERNEL("CombinedPDBandDummyAtomJacobForSingleIonKernel");
		CombinedPDBandDummyAtomJacobForSingleIonKernel<RES_FLOAT_TYPE, AFF_TYPE, BLOCK_WIDTH_T> <<<grid, threads, 0, kernelStream >>>
			(stepSize, thDivs, phDivs, dOutData, dLoc,
			currentIonPosition, currentIonPosition + affCalculator.GetNumAtomsPerIon(i),
			dAffs + i * (numQLayers + 1)
			);
		currentIonPosition += affCalculator.GetNumAtomsPerIon(i);

	}
	if (affCalculator.GetAnomalousIndices())
	{
		int numAnomAtoms = affCalculator.GetAnomalousIndices();
		std::vector<int> anomIndices(numAnomAtoms);
		affCalculator.GetAnomalousIndices(anomIndices.data());

		std::vector<float4> anomLocs(numAnomAtoms);

		for (size_t i = 0; i < numAnomAtoms; i++)
			anomLocs[i] = atomLocations[anomIndices[i]];

		CHKERR(cudaMalloc(&dAnomLocs, sizeof(float4) * numAnomAtoms));
		CHKERR(cudaMemcpyAsync(dAnomLocs, anomLocs.data(), sizeof(float4) * numAnomAtoms, cudaMemcpyHostToDevice, memoryStream));

		std::vector<float2> anomFactors(numAnomAtoms);
		affCalculator.GetSparseAnomalousFactors(anomFactors.data());

		CHKERR(cudaMalloc(&dAnomFactors, sizeof(float2) * numAnomAtoms));
		CHKERR(cudaMemcpyAsync(dAnomFactors, anomFactors.data(), sizeof(float2) * numAnomAtoms, cudaMemcpyHostToDevice, memoryStream));

		CHKERR(cudaStreamSynchronize(memoryStream));
		CHKERR(cudaStreamSynchronize(kernelStream));

		TRACE_KERNEL("AddAnomalousContributionKernel");
		AddAnomalousContributionKernel<<<grid, threads, 0, kernelStream >>>
			(stepSize, thDivs, phDivs, dOutData, dAnomLocs, dAnomFactors, numAnomAtoms);
	}

	CHKERR(cudaStreamSynchronize(kernelStream));

	if (progfunc && progargs)
		progfunc(progargs, progmin + kernFrac * (progmax - progmin));
	kernDone++;


	////////////////////////////////////////////////////////////////
	// Memcpys
	CHKERR(cudaMemcpyAsync(dSolCOM, solCOM.data(), sizeof(float4) * solDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dSolDims, solDims.data(), sizeof(int4) * solDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dOsolDims, outSolDims.data(), sizeof(float4) * outSolDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dOsolCOM, outSolCOM.data(), sizeof(int4) * outSolDims.size(), cudaMemcpyHostToDevice, memoryStream));


	if (cudaSuccess != err) {
		printf("Error copying data to device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Run the voxel based kernel(s)

	// Voxel Based Solvent 
	if (solvED != 0.0 && (affCalculator.GetBitCombination() & CALC_VOXELIZED_SOLVENT) && solDims.size() > 4) {
		CHKERR(cudaStreamSynchronize(kernelStream));
		TRACE_KERNEL("CalcPDBVoxelSolventKenrelJacobSphr");
		CalcPDBVoxelSolventKernelJacobSphr<float, RES_FLOAT_TYPE> << <grid, threads, 0, kernelStream >> >(
			0, stepSize, thDivs, phDivs, dOutData,
			solDims.size(), voxStep,
			dSolDims, dSolCOM,
			-solvED, voxels);
		kernDone++;
		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * kernDone * (progmax - progmin));
	}

	// Voxel Based Outer Solvent 
	if (outerSolED != 0.0 && outSolDims.size() > 4) {
		CHKERR(cudaStreamSynchronize(kernelStream));
		TRACE_KERNEL("CalcPDBVoxelSolventKernelJacobSphr");
		CalcPDBVoxelSolventKernelJacobSphr<float, RES_FLOAT_TYPE> <<<grid, threads, 0, kernelStream >>>(
			0, stepSize, thDivs, phDivs, dOutData,
			outSolDims.size(), voxStep,
			dOsolDims, dOsolCOM,
			outerSolED - solvED, voxels);
		kernDone++;
		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * kernDone * (progmax - progmin));
	}

	// Apply scale
	CHKERR(cudaStreamSynchronize(kernelStream));
#define XPERTHREAD 8
	int newGrid = 2 * int((voxels / maxThreadsPerBlock) / XPERTHREAD) + 1;
	if (scale != 1.0)
	{
		TRACE_KERNEL("ScaleKernel");
		ScaleKernel<double, XPERTHREAD> <<< newGrid, threads, 0, kernelStream >>>((double*)dOutData, scale, 2 * voxels);
	}
#undef XPERTHREAD

	// End of kernels
	////////////////////////////////////////////////////////////////

	// Check for launch errors
	err = cudaPeekAtLastError();
	if (cudaSuccess != err) {
		printf("Launch error: %d", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	CHKERR(cudaStreamSynchronize(kernelStream));
	err = cudaPeekAtLastError();
	if (cudaSuccess != err)
		printf("Error in kernel: %d\n", err);


	err = cudaMemcpy(outData, dOutData, sizeof(RES_FLOAT_TYPE) * 2 * voxels, cudaMemcpyDeviceToHost);
	if (cudaSuccess != err)
		printf("Bad cudaMemcpy from device: %d\n", err);

	FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;

	if (progfunc && progargs)
		progfunc(progargs, progmin + (progmax - progmin));

	if (cudaSuccess != err)
		return -1000 - (int)err;

	//cudaDeviceReset();	// For profiling

	return 0;

}

template <typename RES_FLOAT_TYPE, typename AFF_TYPE, int BLOCK_WIDTH_T = THREADXDIM>
int PDBJacobianGridAmplitudeCalculationCUDA(
	u64 voxels, int thDivs, int phDivs, float stepSize, RES_FLOAT_TYPE *outData,
	std::vector<float4> atomLocations,
	electronAtomicFFCalculator& affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, float voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, float outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop)
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	cudaError err = cudaSuccess;

	size_t numAtoms = atomLocations.size();

	////////////////////////////
	// Device arrays
	RES_FLOAT_TYPE* dOutData;
	float4* dLoc, * dAnomLocs = NULL;
	float2* dAnomFactors = NULL;
	float* dAffs = NULL;

	float q, qx, qy, qz;
	int numQLayers;
	GetQVectorFromIndex(voxels - 1, thDivs, phDivs, stepSize, &q, &qx, &qy, &qz, &numQLayers);
	std::vector<float> affMatrix;
	affMatrix.resize((numQLayers + 1) * affCalculator.GetNumUniqueIon());

	float4* dSolCOM, * dOsolCOM;
	int4* dSolDims, * dOsolDims;

	////////////////////////////
	// Streams
	cudaStream_t kernelStream, memoryStream;
	CHKERR(cudaStreamCreate(&kernelStream));
	CHKERR(cudaStreamCreate(&memoryStream));

	//const int PARTS = 1;
	const u16 maxThreadsPerBlock = BLOCK_WIDTH_T;
	int roundedNumberVoxels = maps::RoundUp((u64)(voxels), maxThreadsPerBlock) * maxThreadsPerBlock;
	int roundedNumAtoms = maps::RoundUp((u64)(numAtoms), maxThreadsPerBlock) * maxThreadsPerBlock;
	const unsigned int N = (roundedNumberVoxels / maxThreadsPerBlock);
	dim3 grid(N, 1, 1);
	dim3 threads(maxThreadsPerBlock, 1, 1);

	int kernJobs = (
		(affCalculator.HasSomethingToCalculate()) ? 1 : 0) +
		((solvED != 0.0 && !(affCalculator.GetBitCombination() & CALC_DUMMY_SOLVENT) && solDims.size() > 4) ? 1 : 0) +
		((outerSolED != 0.0 && outSolDims.size() > 4) ? 1 : 0);
	float kernFrac = 1.f / kernJobs;
	int kernDone = 0;

	// Find and consolidate unique ion and atom indices

	////////////////////////////////////////////////////////////////
	// Mallocs
	CHKERR(cudaMalloc(&dOutData, sizeof(RES_FLOAT_TYPE) * 2 * roundedNumberVoxels));
	CHKERR(cudaMalloc(&dLoc, sizeof(float4) * roundedNumAtoms));
	CHKERR(cudaMalloc(&dAffs, sizeof(float) * affMatrix.size()));

	CHKERR(cudaMalloc(&dSolCOM, sizeof(float4) * solDims.size()));
	CHKERR(cudaMalloc(&dSolDims, sizeof(int4) * solDims.size()));

	CHKERR(cudaMalloc(&dOsolDims, sizeof(float4) * outSolDims.size()));
	CHKERR(cudaMalloc(&dOsolCOM, sizeof(int4) * outSolDims.size()));


	if (cudaSuccess != err) {
		printf("Error allocating memory on device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Memcpys
	CHKERR(cudaMemsetAsync(dAffs, 0, sizeof(float) * affMatrix.size(), memoryStream));
	CHKERR(cudaMemsetAsync(dLoc, 0, sizeof(float4) * roundedNumAtoms, memoryStream));
	CHKERR(cudaMemsetAsync(dOutData, 0, sizeof(RES_FLOAT_TYPE) * 2 * roundedNumberVoxels, memoryStream));
	CHKERR(cudaMemcpyAsync(dLoc, atomLocations.data(), sizeof(float4) * numAtoms, cudaMemcpyHostToDevice, memoryStream));

	affCalculator.GetQMajorAFFMatrix(affMatrix.data(), numQLayers + 1, stepSize);
	CHKERR(cudaMemcpyAsync(dAffs, affMatrix.data(), sizeof(float) * affMatrix.size(), cudaMemcpyHostToDevice, memoryStream));

	if (cudaSuccess != err) {
		printf("Error copying data to device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	CHKERR(cudaStreamSynchronize(memoryStream));

	int currentIonPosition = 0;
	for (size_t i = 0; i < affCalculator.GetNumUniqueIon(); i++)
	{
		CHKERR(cudaStreamSynchronize(kernelStream));

		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * (progmax - progmin) * (float(i) / float(affCalculator.GetNumUniqueIon())));

		TRACE_KERNEL("CombinedPDBandDummyAtomJacobForSingleIonKernel");
		CombinedPDBandDummyAtomJacobForSingleIonKernel<RES_FLOAT_TYPE, AFF_TYPE, BLOCK_WIDTH_T> << <grid, threads, 0, kernelStream >> >
			(stepSize, thDivs, phDivs, dOutData, dLoc,
				currentIonPosition, currentIonPosition + affCalculator.GetNumAtomsPerIon(i),
				dAffs + i * (numQLayers + 1)
				);
		currentIonPosition += affCalculator.GetNumAtomsPerIon(i);

	}
	if (affCalculator.GetAnomalousIndices())
	{
		int numAnomAtoms = affCalculator.GetAnomalousIndices();
		std::vector<int> anomIndices(numAnomAtoms);
		affCalculator.GetAnomalousIndices(anomIndices.data());

		std::vector<float4> anomLocs(numAnomAtoms);

		for (size_t i = 0; i < numAnomAtoms; i++)
			anomLocs[i] = atomLocations[anomIndices[i]];

		CHKERR(cudaMalloc(&dAnomLocs, sizeof(float4) * numAnomAtoms));
		CHKERR(cudaMemcpyAsync(dAnomLocs, anomLocs.data(), sizeof(float4) * numAnomAtoms, cudaMemcpyHostToDevice, memoryStream));

		std::vector<float2> anomFactors(numAnomAtoms);
		affCalculator.GetSparseAnomalousFactors(anomFactors.data());

		CHKERR(cudaMalloc(&dAnomFactors, sizeof(float2) * numAnomAtoms));
		CHKERR(cudaMemcpyAsync(dAnomFactors, anomFactors.data(), sizeof(float2) * numAnomAtoms, cudaMemcpyHostToDevice, memoryStream));

		CHKERR(cudaStreamSynchronize(memoryStream));
		CHKERR(cudaStreamSynchronize(kernelStream));

		TRACE_KERNEL("AddAnomalousContributionKernel");
		AddAnomalousContributionKernel << <grid, threads, 0, kernelStream >> >
			(stepSize, thDivs, phDivs, dOutData, dAnomLocs, dAnomFactors, numAnomAtoms);
	}

	CHKERR(cudaStreamSynchronize(kernelStream));

	if (progfunc && progargs)
		progfunc(progargs, progmin + kernFrac * (progmax - progmin));
	kernDone++;


	////////////////////////////////////////////////////////////////
	// Memcpys
	CHKERR(cudaMemcpyAsync(dSolCOM, solCOM.data(), sizeof(float4) * solDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dSolDims, solDims.data(), sizeof(int4) * solDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dOsolDims, outSolDims.data(), sizeof(float4) * outSolDims.size(), cudaMemcpyHostToDevice, memoryStream));
	CHKERR(cudaMemcpyAsync(dOsolCOM, outSolCOM.data(), sizeof(int4) * outSolDims.size(), cudaMemcpyHostToDevice, memoryStream));


	if (cudaSuccess != err) {
		printf("Error copying data to device: %d\n", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	////////////////////////////////////////////////////////////////
	// Run the voxel based kernel(s)

	// Voxel Based Solvent 
	if (solvED != 0.0 && (affCalculator.GetBitCombination() & CALC_VOXELIZED_SOLVENT) && solDims.size() > 4) {
		CHKERR(cudaStreamSynchronize(kernelStream));
		TRACE_KERNEL("CalcPDBVoxelSolventKenrelJacobSphr");
		CalcPDBVoxelSolventKernelJacobSphr<float, RES_FLOAT_TYPE> << <grid, threads, 0, kernelStream >> > (
			0, stepSize, thDivs, phDivs, dOutData,
			solDims.size(), voxStep,
			dSolDims, dSolCOM,
			-solvED, voxels);
		kernDone++;
		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * kernDone * (progmax - progmin));
	}

	// Voxel Based Outer Solvent 
	if (outerSolED != 0.0 && outSolDims.size() > 4) {
		CHKERR(cudaStreamSynchronize(kernelStream));
		TRACE_KERNEL("CalcPDBVoxelSolventKernelJacobSphr");
		CalcPDBVoxelSolventKernelJacobSphr<float, RES_FLOAT_TYPE> << <grid, threads, 0, kernelStream >> > (
			0, stepSize, thDivs, phDivs, dOutData,
			outSolDims.size(), voxStep,
			dOsolDims, dOsolCOM,
			outerSolED - solvED, voxels);
		kernDone++;
		if (progfunc && progargs)
			progfunc(progargs, progmin + kernFrac * kernDone * (progmax - progmin));
	}

	// Apply scale
	CHKERR(cudaStreamSynchronize(kernelStream));
#define XPERTHREAD 8
	int newGrid = 2 * int((voxels / maxThreadsPerBlock) / XPERTHREAD) + 1;
	if (scale != 1.0)
	{
		TRACE_KERNEL("ScaleKernel");
		ScaleKernel<double, XPERTHREAD> << < newGrid, threads, 0, kernelStream >> > ((double*)dOutData, scale, 2 * voxels);
	}
#undef XPERTHREAD

	// End of kernels
	////////////////////////////////////////////////////////////////

	// Check for launch errors
	err = cudaPeekAtLastError();
	if (cudaSuccess != err) {
		printf("Launch error: %d", (int)err);
		FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;
		return -3000 - err;
	}

	CHKERR(cudaStreamSynchronize(kernelStream));
	err = cudaPeekAtLastError();
	if (cudaSuccess != err)
		printf("Error in kernel: %d\n", err);


	err = cudaMemcpy(outData, dOutData, sizeof(RES_FLOAT_TYPE) * 2 * voxels, cudaMemcpyDeviceToHost);
	if (cudaSuccess != err)
		printf("Bad cudaMemcpy from device: %d\n", err);

	FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY;

	if (progfunc && progargs)
		progfunc(progargs, progmin + (progmax - progmin));

	if (cudaSuccess != err)
		return -1000 - (int)err;

	//cudaDeviceReset();	// For profiling

	return 0;

}

#undef FREE_GPUCalcPDBKernelJacobSphrTempl_MEMORY

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, double *outData,
	std::vector<float4> atomLocations,
	atomicFFCalculator &affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	return PDBJacobianGridAmplitudeCalculationCUDA<double, float, THREADXDIM>(
		voxels, thDivs, phDivs, stepSize, outData,
		atomLocations,
		affCalculator,
		solCOM, solDims, solvED, float(voxStep),
		outSolCOM, outSolDims, float(outerSolED),
		scale, progfunc, progargs, progmin, progmax, pStop);
}

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, double* outData,
	std::vector<float4> atomLocations,
	electronAtomicFFCalculator& affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop)
{
	return PDBJacobianGridAmplitudeCalculationCUDA<double, float, THREADXDIM>(
		voxels, thDivs, phDivs, stepSize, outData,
		atomLocations,
		affCalculator,
		solCOM, solDims, solvED, float(voxStep),
		outSolCOM, outSolDims, float(outerSolED),
		scale, progfunc, progargs, progmin, progmax, pStop);
}

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, float *outData,
	std::vector<float4> atomLocations,
	atomicFFCalculator &affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	return PDBJacobianGridAmplitudeCalculationCUDA<float, float, THREADXDIM>(
		voxels, thDivs, phDivs, stepSize, outData,
		atomLocations,
		affCalculator,
		solCOM, solDims, solvED, float(voxStep),
		outSolCOM, outSolDims, float(outerSolED),
		scale, progfunc, progargs, progmin, progmax, pStop);
}

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, float* outData,
	std::vector<float4> atomLocations,
	electronAtomicFFCalculator& affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop)
{
	return PDBJacobianGridAmplitudeCalculationCUDA<float, float, THREADXDIM>(
		voxels, thDivs, phDivs, stepSize, outData,
		atomLocations,
		affCalculator,
		solCOM, solDims, solvED, float(voxStep),
		outSolCOM, outSolDims, float(outerSolED),
		scale, progfunc, progargs, progmin, progmax, pStop);
}
