#include <cstdio>
#include <cstdlib>

#include "GPUHybridCalc.cuh"
#include "CalculateJacobianSplines.cuh"
#include "HybridOA.cu"

#include <cuda_runtime.h>
#include <stack>
#include <vector>

#include <vector_functions.h>

// For integration
/*
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
*/
#include <curand.h>

//#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>


#include "CommonCUDA.cuh"
#define THETA_BINS 32
#define PHI_BINS (THETA_BINS)

bool GPUHybridCalculator::Initialize(int gpuID, const std::vector<float>& qPoints,
		long long totalSize, int thetaDivisions, int phiDivisions, int qLayers,
		double qMax, double stepSize, GridWorkspace& res)
{
	cudaError_t err = cudaSuccess;
    
	res.calculator = this;
	res.phiDivs = phiDivisions;
	res.thetaDivs = thetaDivisions;
	res.qMax = float(qMax);
	res.totalsz = totalSize;
	res.stepSize = float(stepSize);
	res.qLayers = qLayers;

	res.numQ = int(qPoints.size());
	freePointer(res.qVec);
	res.qVec = new float[res.numQ];
	memcpy(res.qVec, qPoints.data(), sizeof(float) * res.numQ);
		
	if(res.parent == NULL)
	{
		// This is mostly just to clear any memory someone may have forgotten about.
		CHKERR(cudaDeviceReset());
		CHKERR(cudaSetDevice(gpuID));

		cudaStream_t memstream;

		CHKERR(cudaStreamCreate(&memstream));
		res.computeStream = NULL;
	
		res.gpuID = gpuID;
		res.memoryStream = memstream;


		CHKERR(cudaGetLastError());
	}

	return true;
}

bool GPUHybridCalculator::FreeWorkspace(GridWorkspace& workspace)
{
	cudaError_t err = cudaSuccess;
	CHKERR(cudaGetLastError());

	if(workspace.numChildren > 0)
	{
		for(int i = 0; i < workspace.numChildren; i++)
			FreeWorkspace(workspace.children[i]);
		delete [] workspace.children;
		workspace.children = NULL;
		workspace.numChildren = 0;
	}

	if(workspace.parent == NULL)	// Only let the top nodes release the streams
	{
		if(workspace.memoryStream)
			CHKERR(cudaStreamDestroy((cudaStream_t)workspace.memoryStream));
		workspace.memoryStream = NULL;
		if(workspace.computeStream)
			CHKERR(cudaStreamDestroy((cudaStream_t)workspace.computeStream));
		workspace.computeStream = NULL;
	}

	// PDB device parameters
	deviceFreePointer(workspace.d_amp);
	deviceFreePointer(workspace.d_int);
	deviceFreePointer(workspace.d_pdbLocs);
	deviceFreePointer(workspace.d_affCoeffs);
	deviceFreePointer(workspace.d_affs);
	deviceFreePointer(workspace.d_atmRad);
	deviceFreePointer(workspace.d_SolCOM);
	deviceFreePointer(workspace.d_SolDims);
	deviceFreePointer(workspace.d_OSolCOM);
	deviceFreePointer(workspace.d_OSolDims);
	deviceFreePointer(workspace.d_rots);
	deviceFreePointer(workspace.d_nTrans);
	deviceFreePointer(workspace.d_trns);
	deviceFreePointer(workspace.d_params);
	deviceFreePointer(workspace.d_extraPrm);
//	deviceFreePointer(workspace.d_constantMemory);
	deviceFreePointer(workspace.d_symLocs);
	deviceFreePointer(workspace.d_symRots);
	
	freePointer(workspace.numTrans);
	for(int i = 0; i < workspace.numRotations; i++)
	{
		freePointer(workspace.trans[i]);
	}

	freePointer(workspace.qVec);
	
	return err == cudaSuccess;
}

int GPUHybridCalculator::AssembleAmplitudeGrid(GridWorkspace& workspace, double **subAmp,
		double **subInt, double **transRot, int numSubAmps)
{
	printf("GPUHybridCalculator::AssembleAmplitudeGrid not yet implemented\n");
	return -1;
}

int GPUHybridCalculator::OrientationAverageMC(GridWorkspace& workspace, long long maxIters,
						double convergence,  double *qVals, double *iValsOut)
{
	printf("GPUHybridCalculator::OrientationAverageMC not yet implemented\n");
	return -1;
}

int GPUHybridCalculator::CalculateSplines(GridWorkspace& workspace)
{
	cudaError_t err = cudaSuccess;
	float dummyFloat, q;
	int outerLayer;
	long long voxels = workspace.totalsz/2;

	// Allocate the memory for the interpolants
	if(!workspace.d_int) {
		CHKERR(cudaMalloc(&workspace.d_int, sizeof(double) * workspace.totalsz));
	}

	GetQVectorFromIndex(voxels, workspace.thetaDivs, workspace.phiDivs, workspace.stepSize, 
					&q, &dummyFloat, &dummyFloat, &dummyFloat, &outerLayer, NULL);

	CudaJacobianSplines::GPUCalculateSplinesJacobSphrOnCardTempl<double2, double2>
		(workspace.thetaDivs, workspace.phiDivs, outerLayer - 1, (double2*)workspace.d_amp, (double2*)workspace.d_int/*,
		*(cudaStream_t*)workspace.memoryStream, *(cudaStream_t*)workspace.computeStream*/);

	return 0;
}

bool GPUHybridCalculator::ComputeIntensity(std::vector<GridWorkspace>& workspaces,
										   double *outData, double epsi, long long iterations,
										   progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	cudaError_t err = cudaSuccess;
	curandStatus_t ranErr = CURAND_STATUS_SUCCESS;

	if(workspaces.size() == 0)
		return true;

	if(!outData)
		return false;

	// Make sure all children have finished calculating. Probably NOT good for multiple devices/multiple jobs
	CHKERR(cudaDeviceSynchronize());

	GridWorkspace &master = workspaces[0];
	int computeGPU = master.gpuID;

	//size_t numAngles = master.numAngles, numQ = master.numQ, ampPitch = master.ampPitch;
	cudaStream_t dummyStream;
	if(!master.computeStream) {
		CHKERR(cudaStreamCreate( &dummyStream ) );
		master.computeStream = dummyStream;
	}
	cudaStream_t masterComStream = (cudaStream_t)master.computeStream;	
	cudaStream_t masterMemStream = (cudaStream_t)master.memoryStream;	

    curandGenerator_t d_prngGPU;
	cudaStream_t randomStream;
	CHKERR(cudaStreamCreate(&randomStream));

	CHKRANDERR(curandCreateGenerator(&d_prngGPU, CURAND_RNG_PSEUDO_MTGP32));
    CHKRANDERR(curandSetStream(d_prngGPU, randomStream));
    CHKRANDERR(curandSetPseudoRandomGeneratorSeed(d_prngGPU, clock()));
	
	// Can be double if we change curandGenerateUniform to curandGenerateUniformDouble
	bool flip = false;
	double* d_rndNumbers1;
	double* d_rndNumbers2;
	#define ITERS_PER_KERNEL (1024*4)
	size_t numNumbers = 2 * ITERS_PER_KERNEL;
	CHKERR(cudaMalloc(&d_rndNumbers1, sizeof(double) * numNumbers));
	CHKERR(cudaMalloc(&d_rndNumbers2, sizeof(double) * numNumbers));
	double *h_pinned_thetaPhiDivs = NULL;
	double *d_thetaPhiDivs = NULL;
	double *d_thetaPhiBinResults = NULL;
	double *d_integrationResults = NULL;

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, computeGPU);

	const int maxThreadsPerBlock = devProp.maxThreadsPerBlock / 4;
	const long long N = (ITERS_PER_KERNEL / maxThreadsPerBlock) + 1;

	dim3 grid(N, 1, 1);
    dim3 threads(maxThreadsPerBlock, 1, 1);
	dim3 randGrid((numNumbers/maxThreadsPerBlock) + 1, 1, 1);

	int baseLen = -(1 + int(master.numQ * ( master.stepSize / master.qVec[master.numQ-1]) ) );

	float* d_qvec = NULL;

	double** d_amps = NULL;
	double2** d_ints = NULL;
	float4** d_TranslationAddresses = NULL;
	float4** d_RotationAddresses = NULL;
	int** d_NumberOfTranslationAddresses = NULL;
	int* d_NumberOfRotationAddresses = NULL;

	//////////////////
	// Collect the Grid device pointers to a host array to be copied to device
	// To collect: Grid, interpolation coeffs, rotations, #rotations, #translations, # #translations, translations
	std::vector<double*> h_amps;
	std::vector<double2*> h_ints;

	// Count number of amplitudes
	std::vector<float4**> h_TranslationAddresses;
	std::vector<float4*> h_RotationAddresses;
	std::vector<int**> h_NumberOfTranslationAddresses;
	std::vector<int> h_NumberOfRotationAddresses;

	std::stack<GridWorkspace*> wss;

	std::vector<GridWorkspace*> wsv; // NEED?

	wss.push(&workspaces[0]);
	while(!wss.empty())
	{
		GridWorkspace *ws = wss.top();
		wss.pop();

		if(ws->numChildren > 0)
		{
			for(int i = 0; i < ws->numChildren; i++)
			{
				wss.push(&ws->children[i]);
			} // for i
		}
		else
		{
			//////////////////////////////
			// First collect the amplitude and interpolation coeffs.
			h_amps.push_back(ws->d_amp);
			h_ints.push_back((double2*)(ws->d_int));
			//////////////////////////////
			// Prepare [rotations, #rotations, #translations, # #translations, translations] for collection
			// Count translations
			int totTrans = 0;
			std::vector<float4> wsTrnsv;
			std::vector<int> numTrnsv;

			// Collect rotations and #rotations
			h_RotationAddresses.push_back(ws->d_rots);
			h_NumberOfRotationAddresses.push_back(ws->numRotations);

			for(int i = 0; i < ws->numRotations; i++)
			{
				totTrans += ws->numTrans[i];
				numTrnsv.push_back(ws->numTrans[i]);
				wsTrnsv.insert(wsTrnsv.end(), (ws->trans[i]), (ws->trans[i])+ws->numTrans[i]);
			} // for int i < ws->numRotations

			// Malloc and copy the translations
			CPU_VECTOR_TO_DEVICE(ws->d_trns, wsTrnsv.data(), sizeof(float4) * totTrans);
			// d_trns contains the **translations, need one more layer of *
			// Collect **translations
			h_TranslationAddresses.push_back((float4**)ws->d_trns);	// done. Need to copy to host

			// Effectively # translations
			CPU_VECTOR_TO_DEVICE(ws->d_nTrans, numTrnsv.data(), sizeof(int)*numTrnsv.size());

			// Collect # #translations
			h_NumberOfTranslationAddresses.push_back((int**)(ws->d_nTrans));	// This is the map used for determining which and how many translations there are for each amplitude

			wsv.push_back(ws);

		} // if/else ws->numChildren > 0
	} // while wss stack not empty

	// Allocate and copy the h_ vectors to device
	// h_amps
	// h_ints
	// h_RotationAddresses
	// h_NumberOfRotationAddresses
	// h_TranslationAddresses
	// h_NumberOfTranslationAddresses
	

	CPU_VECTOR_TO_DEVICE(d_amps, h_amps.data(), h_amps.size() * sizeof(double*));
	CPU_VECTOR_TO_DEVICE(d_ints, h_ints.data(), h_ints.size() * sizeof(double2*));
	CPU_VECTOR_TO_DEVICE(d_RotationAddresses, h_RotationAddresses.data(), h_RotationAddresses.size() * sizeof(float4*));
	CPU_VECTOR_TO_DEVICE(d_NumberOfRotationAddresses, h_NumberOfRotationAddresses.data(), h_NumberOfRotationAddresses.size() * sizeof(int));
	CPU_VECTOR_TO_DEVICE(d_TranslationAddresses, h_TranslationAddresses.data(), h_TranslationAddresses.size() * sizeof(float4**));
	CPU_VECTOR_TO_DEVICE(d_NumberOfTranslationAddresses, h_NumberOfTranslationAddresses.data(), h_NumberOfTranslationAddresses.size() * sizeof(int**));

	int maxIters = iterations;

	double* h_IntenConv = NULL;

	if(master.intMethod == OA_ADAPTIVE_MC_VEGAS)
	{
		CHKERR(cudaMalloc(&d_thetaPhiDivs, sizeof(double) * (THETA_BINS + PHI_BINS + 2) ) );
		CHKERR(cudaMallocHost(&h_pinned_thetaPhiDivs, sizeof(double) * (THETA_BINS + PHI_BINS + 2) ) );
		memset(h_pinned_thetaPhiDivs, 0, sizeof(double) * (THETA_BINS + PHI_BINS + 2) );
		for(int i = 0; i <= THETA_BINS; i++)
			h_pinned_thetaPhiDivs[i] = double(i) / THETA_BINS;
		for(int i = 0; i <= PHI_BINS; i++)
			h_pinned_thetaPhiDivs[1 + THETA_BINS + i] = double(i) / PHI_BINS;
		CHKERR(cudaMemcpy(d_thetaPhiDivs, h_pinned_thetaPhiDivs, sizeof(double) * (THETA_BINS + PHI_BINS + 2), cudaMemcpyHostToDevice) );
		// The first (THETA_BINS + PHI_BINS) are for the sum, the second for the sum of squares
		CHKERR(cudaMalloc(&d_thetaPhiBinResults, sizeof(double) * 2 * (THETA_BINS + PHI_BINS)));
		CHKERR(cudaMemsetAsync(d_thetaPhiBinResults, 0, sizeof(double) * 2 * (THETA_BINS + PHI_BINS), masterMemStream));
	}
	
	/////////////////////////////////////////
	// Integration loops
	int qInd = 0;
	int lowerLayer = 0;
	CPU_VECTOR_TO_DEVICE(d_qvec, master.qVec, sizeof(float) * master.numQ);
	double2* d_resA = NULL;
	double * d_resI = NULL;
	int *d_resKeys = NULL;
	cudaEvent_t finishedUsingDIntegrationResults, finishedUsingDThetaPhiBinResults;
	CHKERR(cudaEventCreate(&finishedUsingDIntegrationResults));
	CHKERR(cudaEventCreate(&finishedUsingDThetaPhiBinResults));

	do // Loop over grid layers
	{
		// Find the points that need to be integrated
		while(master.qVec[qInd] > double(lowerLayer+1) * master.stepSize) {
			lowerLayer++;
		}

		int qEnd;
		for(qEnd = qInd + 1; qEnd < master.numQ; qEnd++) {
			if(master.qVec[qEnd] > double(lowerLayer+1) * master.stepSize)
				break;
		}
		qEnd--;
		int len = qEnd - qInd + 1;

		// Not enough allocated memory
		if(len > baseLen) {
			baseLen = len + 3;	// The 3 is to hopefully refrain from reaching here more than once
			reallocateMemoryOA(err, d_resA, d_resI, d_resKeys, d_integrationResults, baseLen, masterMemStream, h_IntenConv, maxIters);
		}

		CHKERR(cudaStreamSynchronize(masterMemStream));
		CHKRANDERR(curandGenerateUniformDouble(d_prngGPU, (flip ? d_rndNumbers2 : d_rndNumbers1), numNumbers));
		// Kernel that takes the (0,1] range and turns it to theta phi pairs
		CHKERR(cudaStreamSynchronize(randomStream));
		if(master.intMethod == OA_MC)
		{
			//step 1
			TRACE_KERNEL("UniformRandomToSphericalPoints");
			UniformRandomToSphericalPoints<double2> <<< randGrid, threads, 0, randomStream >>>
				((double2*)(flip ? d_rndNumbers2 : d_rndNumbers1), ITERS_PER_KERNEL);
		
			err = cudaPeekAtLastError();
			if ( cudaSuccess != err ) {
				printf("Error in UniformRandomToSphericalPoints kernel: %d", (int)err);
				break;
			}

			//step 2
			MonteCarloOrientationFunc(err, randomStream, masterMemStream,
				master, finishedUsingDIntegrationResults, d_integrationResults,
				grid, threads, randGrid,
				d_amps, d_ints, d_TranslationAddresses, d_RotationAddresses, d_NumberOfTranslationAddresses, d_NumberOfRotationAddresses,
				h_amps, d_qvec, d_thetaPhiDivs,
				baseLen, finishedUsingDThetaPhiBinResults, d_thetaPhiBinResults, flip,
				ranErr, d_prngGPU, d_rndNumbers2, d_rndNumbers1,
				numNumbers, masterComStream, maxThreadsPerBlock, len,
				devProp, lowerLayer, d_resI, h_IntenConv, d_resA, epsi, maxIters, outData, qInd);
		}
		else if(master.intMethod == OA_ADAPTIVE_MC_VEGAS)
		{
			//nonexistent step 1
			// Do nothing, the [0..1) distribution is dealt with internally by the calculation kernel;
			//step 2:
			VegasOrientationFunc(err, randomStream, masterMemStream,
				master, finishedUsingDIntegrationResults, d_integrationResults,
				grid, threads, randGrid, 
				d_amps,	d_ints, d_TranslationAddresses, d_RotationAddresses, d_NumberOfTranslationAddresses, d_NumberOfRotationAddresses,
				h_amps, d_qvec, d_thetaPhiDivs,
				baseLen, finishedUsingDThetaPhiBinResults, d_thetaPhiBinResults, flip,
				ranErr, d_prngGPU, d_rndNumbers2, d_rndNumbers1,
				numNumbers, masterComStream, maxThreadsPerBlock, len,
				devProp, lowerLayer, d_resI, h_IntenConv, d_resA, epsi, maxIters, outData, qInd);
		}
		else
		{
			printf("ERROR!!! Integration method is not set!!\n");
			break;
		}

		qInd = qEnd + 1;	// TODO Check the +1
		lowerLayer++;

		//TODO::HYBRID
		if(progfunc && progargs)
			progfunc(progargs, progmin + (progmax - progmin) * double(qEnd) / double(master.numQ));

		if(pStop && *pStop)
			break;

	} while(qInd < master.numQ); // Until all q values are integrated

	CHKERR(cudaEventDestroy(finishedUsingDIntegrationResults));

	CHKRANDERR(curandDestroyGenerator(d_prngGPU));
	CHKERR(cudaStreamDestroy(randomStream));

	// Clean up
	cleanUp(d_rndNumbers1, err, d_rndNumbers2, d_resA, d_resI, d_amps, d_ints, d_RotationAddresses, d_NumberOfRotationAddresses, d_TranslationAddresses, d_NumberOfTranslationAddresses, d_thetaPhiDivs, d_thetaPhiBinResults, h_IntenConv, h_pinned_thetaPhiDivs);

	return err == cudaSuccess;	
}

void GPUHybridCalculator::reallocateMemoryOA(cudaError_t &err, double2 * &d_resA, double * &d_resI, int * &d_resKeys, double * &d_integrationResults, int baseLen, const cudaStream_t &masterMemStream, double * &h_IntenConv, int &maxIters)
{
	CHKERR(cudaFree(d_resA));
	CHKERR(cudaFree(d_resI));
	CHKERR(cudaFree(d_resKeys));
	CHKERR(cudaFree(d_integrationResults));

	CHKERR(cudaMalloc(&d_resA, sizeof(double2) * baseLen * ITERS_PER_KERNEL));
	CHKERR(cudaMalloc(&d_resI, sizeof(double) * baseLen * ITERS_PER_KERNEL));
	CHKERR(cudaMalloc(&d_resKeys, sizeof(int) * baseLen * ITERS_PER_KERNEL));

	CHKERR(cudaMalloc(&d_integrationResults, sizeof(double) * baseLen));

	cudaMemsetAsync(d_resA, 0, sizeof(double2) * baseLen * ITERS_PER_KERNEL, masterMemStream);
	cudaMemsetAsync(d_resI, 0, sizeof(double) * baseLen * ITERS_PER_KERNEL, masterMemStream);
	cudaMemsetAsync(d_integrationResults, 0, sizeof(double) * baseLen, masterMemStream);

	if (h_IntenConv)
		cudaFreeHost(h_IntenConv);
	CHKERR(cudaMallocHost(&h_IntenConv, sizeof(double) * baseLen * ((maxIters / ITERS_PER_KERNEL) + 1)));
	memset(h_IntenConv, 0, sizeof(double) * baseLen * ((maxIters / ITERS_PER_KERNEL) + 1));
}

void GPUHybridCalculator::MonteCarloOrientationFunc(cudaError_t &err, const cudaStream_t &randomStream, const cudaStream_t &masterMemStream,
	GridWorkspace & master, const cudaEvent_t &finishedUsingDIntegrationResults, double * d_integrationResults,
	dim3 grid, dim3 threads, dim3 randGrid,
	double** d_amps, double2** d_ints, float4** d_TranslationAddresses, float4** d_RotationAddresses, int** d_NumberOfTranslationAddresses, int* d_NumberOfRotationAddresses,
	std::vector<double*> h_amps, float* d_qvec, double *d_thetaPhiDivs,
	int baseLen, const cudaEvent_t &finishedUsingDThetaPhiBinResults, double * d_thetaPhiBinResults, bool &flip,
	curandStatus_t &ranErr, const curandGenerator_t &d_prngGPU, double * d_rndNumbers2, double * d_rndNumbers1,
	const size_t &numNumbers, const cudaStream_t &masterComStream, const int &maxThreadsPerBlock, int len,
	cudaDeviceProp &devProp, int lowerLayer, double * d_resI, double * h_IntenConv, double2 * d_resA, double epsi, int maxIters, double * outData, int qInd)
{
	bool converged = false;
	int loopCtr = 0;

	do // Loop until convergence
	{
		// Wait for random numbers and previous memory copies to be ready
		CHKERR(cudaStreamSynchronize(randomStream));
		CHKERR(cudaStreamSynchronize(masterMemStream));

		if (err != cudaSuccess) {
			printf("LoopCtr: %d\n", loopCtr);
		}
		////////////////
		// RUN KERNEL!!

			//cudaDeviceSynchronize();
			TRACE_KERNEL("HybridMCOAJacobianKernel");
			HybridMCOAJacobianKernel<double, double2, double2, ITERS_PER_KERNEL> << <grid, threads, 0, masterComStream >> >
				(d_amps, d_ints, h_amps.size(),
					master.thetaDivs, master.phiDivs, master.stepSize,
					d_RotationAddresses, d_NumberOfRotationAddresses,
					d_TranslationAddresses, d_NumberOfTranslationAddresses,
					lowerLayer, len, d_qvec + qInd,
					(double2*)(flip ? d_rndNumbers2 : d_rndNumbers1),
					d_resA
					);
		

		flip = !flip;
		// Run new random numbers concurrently
		CHKRANDERR(curandGenerateUniformDouble(d_prngGPU, (flip ? d_rndNumbers2 : d_rndNumbers1), numNumbers));
		// Kernel that takes the (0,1] range and turns it to theta phi pairs
		CHKERR(cudaStreamSynchronize(randomStream));

			TRACE_KERNEL("UniformRandomToSphericalPoints");
			UniformRandomToSphericalPoints<double2> << < randGrid, threads, 0, randomStream >> >
				((double2*)(flip ? d_rndNumbers2 : d_rndNumbers1), ITERS_PER_KERNEL);
		

		CHKERR(cudaStreamSynchronize(masterComStream));
		// Convert to intensity
		dim3 cgrid((ITERS_PER_KERNEL / maxThreadsPerBlock) * 2 + 1, (len / (devProp.maxThreadsPerBlock / maxThreadsPerBlock)) + 1, 1);
		dim3 cthreads(maxThreadsPerBlock / 2, devProp.maxThreadsPerBlock / maxThreadsPerBlock, 1);
		TRACE_KERNEL("ConvertAmplitudeToIntensity");
		ConvertAmplitudeToIntensity<double, double2, ITERS_PER_KERNEL> << <cgrid, cthreads, 0, masterComStream >> >(
			d_resA, d_resI, len);

		CHKERR(cudaStreamSynchronize(masterComStream));
		err = cudaPeekAtLastError();
		if (cudaSuccess != err) {
			printf("Error in kernel: %d\t[lowerLayer = %d\tloopCtr = %d]\n", (int)err, loopCtr, lowerLayer);
		}


		// Collect results


#if defined(WIN32) || defined(_WIN32)
#pragma warning( push, 0) // Disabling 4305 (name/decoration too long)
#endif
			thrust::device_vector<int> dptr_intind(len);
			thrust::device_vector<double> dptr_intensity(len);

			thrust::reduce_by_key
			(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(ITERS_PER_KERNEL)),	//keys_first
				thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(ITERS_PER_KERNEL)) + (ITERS_PER_KERNEL*len),	// keys_last
				thrust::device_ptr<double>(d_resI),	// values_first
				dptr_intind.data(),					// keys_output
				dptr_intensity.data(),				// values_output
				thrust::equal_to<int>(),			// binary_pred
				thrust::plus<double>());			// binary_op

			CHKERR(cudaMemcpy((h_IntenConv + loopCtr * len),
				thrust::raw_pointer_cast(dptr_intensity.data()), sizeof(double) * len, cudaMemcpyDeviceToHost));
#if defined(WIN32) || defined(_WIN32)
#pragma warning( pop )
#endif
		
		CHKERR(cudaMemsetAsync(d_resA, 0, sizeof(double2) * len * ITERS_PER_KERNEL, masterMemStream));


			if (loopCtr == 0) {
				for (int i = 0; i < len; i++) {
					h_IntenConv[i] /= double(ITERS_PER_KERNEL);
				}
			}
			else {
				for (int i = 0; i < len; i++) {
					h_IntenConv[loopCtr*len + i] = (h_IntenConv[(loopCtr - 1)*len + i] * double(loopCtr * ITERS_PER_KERNEL) +
						h_IntenConv[loopCtr*len + i]) /
						double((loopCtr + 1) * ITERS_PER_KERNEL);
				}
			}	// if/else

		// Convergence Place TODO FIXME
		// Check convergence
		checkConvergenceOA(loopCtr, epsi, len, h_IntenConv, converged);
	} while (++loopCtr * ITERS_PER_KERNEL < maxIters && !converged);	// Until converged

																		// Copy the results
	memcpy(outData + qInd, h_IntenConv + (loopCtr - 1)*len, sizeof(double)*len);
}


void GPUHybridCalculator::VegasOrientationFunc(cudaError_t &err, const cudaStream_t &randomStream, const cudaStream_t &masterMemStream,
	GridWorkspace & master, const cudaEvent_t &finishedUsingDIntegrationResults, double * d_integrationResults,
	dim3 grid, dim3 threads, dim3 randGrid,
	double** d_amps, double2** d_ints, float4** d_TranslationAddresses, float4** d_RotationAddresses, int** d_NumberOfTranslationAddresses, int* d_NumberOfRotationAddresses,
	std::vector<double*> h_amps, float* d_qvec, double *d_thetaPhiDivs,
	int baseLen, const cudaEvent_t &finishedUsingDThetaPhiBinResults, double * d_thetaPhiBinResults, bool &flip,
	curandStatus_t &ranErr, const curandGenerator_t &d_prngGPU, double * d_rndNumbers2, double * d_rndNumbers1,
	const size_t &numNumbers, const cudaStream_t &masterComStream, const int &maxThreadsPerBlock, int len,
	cudaDeviceProp &devProp, int lowerLayer, double * d_resI, double * h_IntenConv, double2 * d_resA, double epsi, int maxIters, double * outData, int qInd)
{
	bool converged = false;
	int loopCtr = 0;

	do // Loop until convergence
	{
		// Wait for random numbers and previous memory copies to be ready
		CHKERR(cudaStreamSynchronize(randomStream));
		CHKERR(cudaStreamSynchronize(masterMemStream));

		if (err != cudaSuccess) {
			printf("LoopCtr: %d\n", loopCtr);
		}
		////////////////
		// RUN KERNEL!!

			TRACE_KERNEL("HybridVEGASMCOAJacobianKernel");
			HybridVEGASMCOAJacobianKernel<double, double2, double2, ITERS_PER_KERNEL,
				THETA_BINS, PHI_BINS, ITERS_PER_KERNEL / (THETA_BINS * PHI_BINS)>
				<< < grid, threads, 0, masterComStream >> >
				(d_thetaPhiDivs,
					d_amps, d_ints, h_amps.size(),
					master.thetaDivs, master.phiDivs, master.stepSize,
					d_RotationAddresses, d_NumberOfRotationAddresses,
					d_TranslationAddresses, d_NumberOfTranslationAddresses,
					lowerLayer, len, d_qvec + qInd,
					(double2*)(flip ? d_rndNumbers2 : d_rndNumbers1),
					d_resA
					);
			CHKERR(cudaEventSynchronize(finishedUsingDIntegrationResults));
			CHKERR(cudaMemsetAsync(d_integrationResults, 0, sizeof(double) * baseLen, masterMemStream));
			CHKERR(cudaEventRecord(finishedUsingDIntegrationResults, masterMemStream));
			CHKERR(cudaEventSynchronize(finishedUsingDThetaPhiBinResults));
			CHKERR(cudaMemsetAsync(d_thetaPhiBinResults, 0, sizeof(double) * 2 * (THETA_BINS + PHI_BINS), masterMemStream));
			CHKERR(cudaEventRecord(finishedUsingDThetaPhiBinResults, masterMemStream));
		

		flip = !flip;
		// Run new random numbers concurrently
		CHKRANDERR(curandGenerateUniformDouble(d_prngGPU, (flip ? d_rndNumbers2 : d_rndNumbers1), numNumbers));
		// Kernel that takes the (0,1] range and turns it to theta phi pairs - as far as i can tell, this only actually does something in monte carlo
		CHKERR(cudaStreamSynchronize(randomStream)); //can probably be commented out
		//if (master.intMethod == OA_ADAPTIVE_MC_VEGAS)
		//{ //I'm only keeping this comment here to make it obvious that in monte carlo something happens here, even though in vegas nothing does (for refactoring clarity purposes)
		//	;
		//}

		CHKERR(cudaStreamSynchronize(masterComStream));
		// Convert to intensity
		dim3 cgrid((ITERS_PER_KERNEL / maxThreadsPerBlock) * 2 + 1, (len / (devProp.maxThreadsPerBlock / maxThreadsPerBlock)) + 1, 1);
		dim3 cthreads(maxThreadsPerBlock / 2, devProp.maxThreadsPerBlock / maxThreadsPerBlock, 1);
		TRACE_KERNEL("ConvertAmplitudeToIntensity");
		ConvertAmplitudeToIntensity<double, double2, ITERS_PER_KERNEL> << <cgrid, cthreads, 0, masterComStream >> > (
			d_resA, d_resI, len);

		CHKERR(cudaStreamSynchronize(masterComStream));
		err = cudaPeekAtLastError();
		if (cudaSuccess != err) {
			printf("Error in kernel: %d\t[lowerLayer = %d\tloopCtr = %d]\n", (int)err, loopCtr, lowerLayer);
		}

		CHKERR(cudaMemsetAsync(d_resA, 0, sizeof(double2) * len * ITERS_PER_KERNEL, masterMemStream));


			CHKERR(cudaEventSynchronize(finishedUsingDIntegrationResults));
			TRACE_KERNEL("HybridVEGASBinReduceToIKernel");
			HybridVEGASBinReduceToIKernel<double, ITERS_PER_KERNEL, THETA_BINS, PHI_BINS, (ITERS_PER_KERNEL / (THETA_BINS * PHI_BINS))>
				<< < 1, THETA_BINS*PHI_BINS, 0, masterComStream >> >
				(d_resI, d_thetaPhiDivs, len, d_integrationResults);
			CHKERR(cudaEventRecord(finishedUsingDIntegrationResults, masterComStream));


			//printf("TODO: reduce by key in both dimensions.\n");
			CHKERR(cudaEventSynchronize(finishedUsingDThetaPhiBinResults));
			TRACE_KERNEL("HybridVEGASBinReduceKernel");
			HybridVEGASBinReduceKernel<double, ITERS_PER_KERNEL, THETA_BINS, PHI_BINS, (ITERS_PER_KERNEL / (THETA_BINS * PHI_BINS))>
				<< < 1, THETA_BINS*PHI_BINS, 0, masterComStream >> >
				(d_resI, d_thetaPhiDivs, len, d_thetaPhiBinResults);
			CHKERR(cudaEventRecord(finishedUsingDThetaPhiBinResults, masterComStream));

			CHKERR(cudaEventSynchronize(finishedUsingDIntegrationResults));
			CHKERR(cudaMemcpyAsync((h_IntenConv + loopCtr * len),
				d_integrationResults, sizeof(double) * len, cudaMemcpyDeviceToHost, masterMemStream));
			CHKERR(cudaEventRecord(finishedUsingDIntegrationResults, masterMemStream));


			TRACE_KERNEL("HybridVEGASResizeGridKernel");
			HybridVEGASResizeGridKernel<double, THETA_BINS, PHI_BINS>
				<< <1, THETA_BINS + PHI_BINS, 0, masterComStream >> >
				(d_thetaPhiDivs, d_thetaPhiBinResults);



			if (loopCtr > 0) {
				CHKERR(cudaEventSynchronize(finishedUsingDIntegrationResults));

				for (int i = 0; i < len; i++) {
					h_IntenConv[loopCtr*len + i] = (h_IntenConv[(loopCtr - 1)*len + i] * double(loopCtr) +
						h_IntenConv[loopCtr*len + i]) / double(loopCtr + 1);

				}
			}
			else
			{
				CHKERR(cudaEventSynchronize(finishedUsingDIntegrationResults));
			}
		
		// Check convergence
		checkConvergenceOA(loopCtr, epsi, len, h_IntenConv, converged);
	} while (++loopCtr * ITERS_PER_KERNEL < maxIters && !converged);	// Until converged

																		// Copy the results
	memcpy(outData + qInd, h_IntenConv + (loopCtr - 1)*len, sizeof(double)*len);
}

void GPUHybridCalculator::checkConvergenceOA(int loopCtr, double epsi, int len, double * h_IntenConv, bool &converged)
{
	// Convergence Place TODO FIXME -- this comment is from Avi
	if (loopCtr > 2 && epsi > 0.0) {
		bool tmp = true;
		for (int i = 0; i < len; i++) {
			if ((fabs(1.0 - (h_IntenConv[loopCtr*len + i] / h_IntenConv[(loopCtr - 1)*len + i])) > epsi) ||
				(fabs(1.0 - (h_IntenConv[loopCtr*len + i] / h_IntenConv[(loopCtr - 2)*len + i])) > epsi) ||
				(fabs(1.0 - (h_IntenConv[loopCtr*len + i] / h_IntenConv[(loopCtr - 3)*len + i])) > epsi)) {
				tmp = false;
				break;
			} // if
		} // for i
		converged = tmp;
	} // if loopCtr > 2
}

void GPUHybridCalculator::cleanUp(double * &d_rndNumbers1, cudaError_t &err, double * &d_rndNumbers2, double2 * &d_resA, double * &d_resI, double ** &d_amps, double2 ** &d_ints, float4 ** &d_RotationAddresses, int * &d_NumberOfRotationAddresses, float4 ** &d_TranslationAddresses, int ** &d_NumberOfTranslationAddresses, double * &d_thetaPhiDivs, double * &d_thetaPhiBinResults, double * &h_IntenConv, double * &h_pinned_thetaPhiDivs)
{
	deviceFreePointer(d_rndNumbers1);
	deviceFreePointer(d_rndNumbers2);
	deviceFreePointer(d_resA);
	deviceFreePointer(d_resI);
	deviceFreePointer(d_amps);
	deviceFreePointer(d_ints);
	deviceFreePointer(d_RotationAddresses);
	deviceFreePointer(d_NumberOfRotationAddresses);
	deviceFreePointer(d_TranslationAddresses);
	deviceFreePointer(d_NumberOfTranslationAddresses);
	deviceFreePointer(d_thetaPhiDivs);
	deviceFreePointer(d_thetaPhiBinResults);

	if (h_IntenConv)
		cudaFreeHost(h_IntenConv);
	h_IntenConv = NULL;

	if (h_pinned_thetaPhiDivs)
		cudaFreeHost(h_pinned_thetaPhiDivs);
	h_pinned_thetaPhiDivs = NULL;
}

bool GPUHybridCalculator::SetNumChildren(GridWorkspace &workspace, int numChildren)
{
	if(workspace.children)
	{
		for(int i = 0; i < workspace.numChildren; i++)
		{
			FreeWorkspace(workspace.children[i]);
		}
		delete [] workspace.children;
		workspace.children = NULL;
	}

	workspace.numChildren = numChildren;

	if(numChildren > 0) {
		workspace.children = new GridWorkspace[numChildren];
		for(int i = 0; i < workspace.numChildren; i++)
		{
			workspace.children[i].parent = &workspace;

			workspace.children[i].calculator	= this;
			workspace.children[i].phiDivs		= workspace.phiDivs;
			workspace.children[i].thetaDivs		= workspace.thetaDivs;
			workspace.children[i].qMax			= workspace.qMax;
			workspace.children[i].totalsz		= workspace.totalsz;
			workspace.children[i].stepSize		= workspace.stepSize;
			workspace.children[i].qLayers		= workspace.qLayers;
		
			workspace.children[i].computeStream	= workspace.computeStream;
			workspace.children[i].memoryStream	= workspace.memoryStream;
			workspace.children[i].gpuID			= workspace.gpuID;
		}
	}

	return true;

}

bool GPUHybridCalculator::AddRotations(GridWorkspace &workspace, std::vector<float4>& rotations)
{
	cudaError_t err = cudaSuccess;
	if(workspace.trans) 
	{
		for(int cnt = 0; cnt < workspace.numRotations; cnt++)
		{
			freePointer(workspace.trans[cnt]);
		}
		freePointer(workspace.trans);
	}
	workspace.trans = new float4*[rotations.size()];
	for(int cnt = 0; cnt < rotations.size(); cnt++)
		workspace.trans[cnt] = NULL;


	workspace.numRotations = int(rotations.size());
	deviceFreePointer(workspace.d_rots);
	cudaMalloc(&workspace.d_rots, sizeof(float4) * rotations.size());
	// If the following is asynchronous, *rotations may be deleted before finished (it's a temp variable)!!
	CHKERR(cudaMemcpyAsync(workspace.d_rots, rotations.data(), sizeof(float4) * workspace.numRotations, cudaMemcpyHostToDevice, (cudaStream_t)(workspace.memoryStream)));

#ifdef _DEBUG
	printf("[%d] = \n", workspace.numRotations);
	for (int i = 0; i < workspace.numRotations; i++)
	{
		printf("\t[%d] = {%f, %f, %f}\n", i, rotations[i].x, rotations[i].y, rotations[i].z);
	}
#endif

	freePointer(workspace.numTrans);
	workspace.numTrans = new int[workspace.numRotations];

	for(int i = 0; i < workspace.numRotations; i++) {
		workspace.numTrans[i] = 0;
	}
	return true;
}

bool GPUHybridCalculator::AddTranslations(GridWorkspace &workspace, int rotationIndex, std::vector<float4>& translations)
{
	workspace.numTrans[rotationIndex] = int(translations.size());

	freePointer(workspace.trans[rotationIndex]);

	workspace.trans[rotationIndex] = new float4[translations.size()];

	memcpy(workspace.trans[rotationIndex], translations.data(), sizeof(float4) * translations.size());

	return true;
}
