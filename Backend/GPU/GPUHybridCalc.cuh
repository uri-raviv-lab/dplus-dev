#pragma once

#include "GPUInterface.h"
#include <cuda_runtime.h>
#include <curand.h>
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16


class GPUHybridCalculator : public IGPUGridCalculator
{
public:
		
	virtual bool Initialize(int gpuID, const std::vector<float>& qPoints,
		long long totalSize, int thetaDivisions, int phiDivisions, int qLayers,
		double qMax, double stepSize, GridWorkspace& res);

	virtual bool FreeWorkspace(GridWorkspace& workspace);

	virtual bool ComputeIntensity(std::vector<GridWorkspace>& workspaces,
		double *outData, double epsi, long long iterations,
		progressFunc progfunc = NULL, void *progargs = NULL, float progmin = 0., float progmax = 0., int *pStop = NULL);

	void reallocateMemoryOA(cudaError_t &err, double2 * &d_resA, double * &d_resI, int * &d_resKeys, double * &d_integrationResults, int baseLen, const cudaStream_t &masterMemStream, double * &h_IntenConv, int &maxIters);

	void MonteCarloOrientationFunc(cudaError_t &err, const cudaStream_t &randomStream, const cudaStream_t &masterMemStream, 
		GridWorkspace & master, const cudaEvent_t &finishedUsingDIntegrationResults, double * d_integrationResults, 
		dim3 grid, dim3 threads,
		dim3 randGrid,
		double** d_amps,
		double2** d_ints,
		float4** d_TranslationAddresses,
		float4** d_RotationAddresses,
		int** d_NumberOfTranslationAddresses,
		int* d_NumberOfRotationAddresses,
		std::vector<double*> h_amps,
		float* d_qvec,
		double *d_thetaPhiDivs,
		int baseLen, const cudaEvent_t &finishedUsingDThetaPhiBinResults, double * d_thetaPhiBinResults, bool &flip, curandStatus_t &ranErr, const curandGenerator_t &d_prngGPU, double * d_rndNumbers2, double * d_rndNumbers1, const size_t &numNumbers, const cudaStream_t &masterComStream, const int &maxThreadsPerBlock, int len, cudaDeviceProp &devProp, int lowerLayer, double * d_resI, double * h_IntenConv, double2 * d_resA, double epsi, int maxIters, double * outData, int qInd);

	void VegasOrientationFunc(cudaError_t &err, const cudaStream_t &randomStream, const cudaStream_t &masterMemStream,
		GridWorkspace & master, const cudaEvent_t &finishedUsingDIntegrationResults, double * d_integrationResults,
		dim3 grid, dim3 threads,
		dim3 randGrid,
		double** d_amps,
		double2** d_ints,
		float4** d_TranslationAddresses,
		float4** d_RotationAddresses,
		int** d_NumberOfTranslationAddresses,
		int* d_NumberOfRotationAddresses,
		std::vector<double*> h_amps,
		float* d_qvec,
		double *d_thetaPhiDivs,
		int baseLen, const cudaEvent_t &finishedUsingDThetaPhiBinResults, double * d_thetaPhiBinResults, bool &flip, curandStatus_t &ranErr, const curandGenerator_t &d_prngGPU, double * d_rndNumbers2, double * d_rndNumbers1, const size_t &numNumbers, const cudaStream_t &masterComStream, const int &maxThreadsPerBlock, int len, cudaDeviceProp &devProp, int lowerLayer, double * d_resI, double * h_IntenConv, double2 * d_resA, double epsi, int maxIters, double * outData, int qInd);

	void checkConvergenceOA(int loopCtr, double epsi, int len, double * h_IntenConv, bool &converged);


	void cleanUp(double * &d_rndNumbers1, cudaError_t &err, double * &d_rndNumbers2, double2 * &d_resA, double * &d_resI, double ** &d_amps, double2 ** &d_ints, float4 ** &d_RotationAddresses, int * &d_NumberOfRotationAddresses, float4 ** &d_TranslationAddresses, int ** &d_NumberOfTranslationAddresses, double * &d_thetaPhiDivs, double * &d_thetaPhiBinResults, double * &h_IntenConv, double * &h_pinned_thetaPhiDivs);

	virtual bool SetNumChildren(GridWorkspace &workspace, int numChildren);

	virtual bool AddRotations(GridWorkspace &workspace, std::vector<float4>& rotations);

	virtual bool AddTranslations(GridWorkspace &workspace, int rotationIndex, std::vector<float4>& translations);

	int AssembleAmplitudeGrid(GridWorkspace& workspace, double **subAmp,
		double **subInt, double **transRot, int numSubAmps);
	
	int CalculateSplines(GridWorkspace& workspace);

	int OrientationAverageMC(GridWorkspace& workspace, long long maxIters,
						double convergence,  double *qVals, double *iValsOut);

};
