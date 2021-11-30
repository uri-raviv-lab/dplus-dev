#include "GPUHeader.h"
#include "InternalGPUHeaders.h"
#include "Common.h"
#include "GPUInterface.h"
#include "GPUDirectCalc.cuh"
#include "GPUHybridCalc.cuh"
#include <cuda_runtime_api.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__ ((visibility ("default")))
#endif
  
/************************************************************************/
/* Calculate the intensity from a PDB directly using the GPU.           */
/************************************************************************/
IGPUCalculator *GPUCreateCalculatorDirect()
{
	return new GPUDirectCalculator();
}



bool GPUDirect_SetPDBDLL(Workspace& work, const float4 *atomLocs,
	const unsigned char *ionInd,
	size_t numAtoms, const float *coeffs,
	const unsigned int *atomsPerIon,
	size_t numCoeffs)
{
	return GPUDirect_SetPDB(work, atomLocs, ionInd, numAtoms, coeffs, atomsPerIon, numCoeffs);
}

bool GPUDirect_PDBAmplitudeDLL(Workspace& work, float3 rotation)
{
	return GPUDirect_PDBAmplitude(work, rotation);
}

IGPUGridCalculator *GPUCreateCalculatorHybrid()
{
	return new GPUHybridCalculator();
}


bool GPUHybrid_PDBAmplitudeDLL(GridWorkspace& work)
{
	return GPUHybrid_PDBAmplitude(work);
}



bool GPUHybrid_USphereAmplitudeDLL(GridWorkspace& work)
{
	return GPUHybrid_USphereAmplitude(work);
}


bool GPUHybrid_SetUSphereDLL(GridWorkspace& work, float2 *params, int numLayers, float* extras, int nExtras)
{
	return GPUHybrid_SetUSphere(work, params, numLayers, extras, nExtras);
}




bool GPUDirect_SetSphereParamsDLL(Workspace& work, const float2 *params, int numLayers)
{
	return GPUDirect_SetSphereParams(work, params, numLayers);
}



bool GPUHybrid_AmpGridAmplitudeDLL(GridWorkspace& work, double* amp)
{
	return GPUHybrid_AmpGridAmplitude(work, amp);
}

bool GPUHybrid_SetSymmetryDLL(GridWorkspace& work, float4 *locs, float4 *rots, int numLocs)
{
	return GPUHybrid_SetSymmetry(work, locs, rots, numLocs);
}

bool GPUHybrid_ManSymmetryAmplitudeDLL(GridWorkspace& work, GridWorkspace& child, float4 trans, float4 rot)
{
	return GPUHybrid_ManSymmetryAmplitude(work, child, trans, rot);
}


bool GPUSemiHybrid_ManSymmetryAmplitudeDLL(GridWorkspace& work, double* amp, float4 trans, float4 rot)
{
	return GPUSemiHybrid_ManSymmetryAmplitude(work, amp, trans, rot);
}


// int GPUCalculateDebyeD(int qValues, double *qVals, double *outData,
// 	double *loc, u8 *ionInd, int numAtoms, double *coeffs, int numCoeffs, bool bSol,
// 	u8 * atmInd, double *rad, double solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop) {
// 	return GPUCalcDebye(qValues, qVals, outData, loc, ionInd, numAtoms, coeffs, numCoeffs,
// 		bSol, rad, solvED,
// 		progfunc, progargs, progmin, progmax, pStop);
// }
// 
// 
// int GPUCalculateDebyeF(int qValues, float *qVals, float *outData,
// 	float *loc, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSol,
// 	float *rad, float solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop) {
// 	return GPUCalcDebye(qValues, qVals, outData, loc, ionInd, numAtoms, coeffs, numCoeffs,
// 		bSol, rad, solvED,
// 		progfunc, progargs, progmin, progmax, pStop);
// }




/************************************************************************/
/* General dll functions                                                */
/************************************************************************/

void ResetGPU() {
	cudaDeviceReset();
}

void GPUReset()
{
	ResetGPU();
}


#ifdef _WIN32_DONT_USE_THIS_ONCE_THE_GPU_BACKEND_IS_MERGED_INTO_THE_MAIN_BACKEND
int _stdcall DllMain(_In_ void * _HDllHandle, _In_ unsigned _Reason, _In_opt_ void * _Reserved) {
	switch (_Reason) {
	case 1: // DLL_PROCESS_ATTACH
	case 2: // DLL_THREAD_ATTACH
		if (GetNumGPUs() <= 0)
			return 0; // Don't let the DLL load if there are no CUDA-capable devices
		break;
	}

	return 1;
}
#endif



#pragma region disgusting, horrible stuff

//////////////////////////////////////////////////////////////////////////
// Version 2.0 of the Debye formula
#define GET_EXPORTED_GPUCalcDebyeV2_MACRO(SUFFIX, DFT)													\
int GPUCalcDebyeV2(int numQValues, float qMin, float qMax, DFT *outData, int numAtoms, 					\
	const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors,		\
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,		\
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);				\
int GPUCalculateDebye##SUFFIX##v2(int qValues, float qMin, float qMax,				\
	DFT *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals,	\
	bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd,				\
	float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs,									\
	double progmin, double progmax, int *pStop)															\
{																										\
	return GPUCalcDebyeV2(qValues, qMin, qMax, outData, numAtoms,										\
		atomsPerIon, loc, ionInd, anomalousVals, bBfactors, BFactors, coeffs, bSol, bSolOnly,			\
		atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);						\
}																										

int GPUCalculateDebyeDv2(int qValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	\
		return GPUCalcDebyeV2(qValues, qMin, qMax, outData, numAtoms, \
		atomsPerIon, loc, ionInd, anomalousVals, bBfactors, BFactors, coeffs, bSol, bSolOnly, \
		atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);							\
}
// int GPUCalculateDebyeFv2(int qValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
// {
// 	\
// 		return GPUCalcDebyeV2(qValues, qMin, qMax, outData, numAtoms, \
// 		atomsPerIon, loc, ionInd, anomalousVals, bBfactors, BFactors, coeffs, bSol, bSolOnly, \
// 		atmInd, atmRad, solvED, c1, progfunc, progargs, progmin, progmax, pStop);							\
// }


#undef GET_EXPORTED_GPUCalcDebyeV2_MACRO

/************************************************************************/
/* Version 3.0 of the Debye formula                                     */
/************************************************************************/
template<typename FLOAT_TYPE>
int GPUCalcDebyeV3MAPS(
	int numQValues, float qMin, float qMax, FLOAT_TYPE *outData, int numAtoms,
	const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors,
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

#define GET_EXPORTED_GPUCalcDebyeV3MAPS_MACRO(SUFFIX,T1)									\
int GPUCalcDebyeV3MAPS##SUFFIX(										\
	int numQValues, float qMin, float qMax,	T1 *outData, int numAtoms,						\
	const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors,		\
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED,		\
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)		\
	{																						\
		return GPUCalcDebyeV3MAPS<T1>(numQValues, qMin, qMax, outData, numAtoms, atomsPerIon,	\
								  loc, ionInd, bBfactors, BFactors, coeffs, bSol, bSolOnly,	\
								  atmInd, atmRad, solvED, progfunc, progargs, progmin,		\
								  progmax, pStop);											\
	}


#undef GET_EXPORTED_GPUCalcDebyeV3MAPS_MACRO



/************************************************************************/
/* Calculate the amplitude from a PDB using the Jacobian grids.         */
/************************************************************************/
#define GET_EXPORTED_GPUCalcPDBJacobSphr_MACRO(SUFFIX, T1, T2)											\
	int GPUCalcPDBJacobSphr(u64 voxels, int thDivs, int phDivs, T1 stepSize,							\
	T2 *outData, float *locX, float *locY, float *locZ,													\
	u8	*ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly,								\
	u8	* atmInd, float *atmRad, T1 solvED,	u8 solventType/* FOR DUMMY ATOM SOLVENT*/,					\
	float4 *solCOM, int4 *solDims, u64 solDimLen, T1 voxStep/*For voxel based solvent*/,				\
	float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, T1 outerSolED /*For outer solvent layer*/,	\
	T1 scale,																							\
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);					\
int GPUCalcPDBJacobSphr##SUFFIX(													\
	u64 voxels, int thDivs, int phDivs, T1 stepSize, T2 *outData,										\
	float *locX, float *locY, float *locZ,																\
	u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly,								\
	u8 * atmInd, float *atmRad, T1 solvED,	u8 solventType,												\
	float4 *solCOM, int4 *solDims, u64 solDimLen, T1 voxStep,											\
	float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, T1 outerSolED, T1 scale,						\
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)					\
{																										\
		return GPUCalcPDBJacobSphr(voxels, thDivs, phDivs, stepSize, outData, locX, locY, locZ,			\
			ionInd, numAtoms, coeffs, numCoeffs, bSolOnly,												\
			atmInd, atmRad, solvED, solventType,														\
			solCOM, solDims, solDimLen, voxStep,														\
			outSolCOM, outSolDims, outSolDimLen, outerSolED, scale,										\
			progfunc, progargs, progmin, progmax, pStop);												\
}


#undef GET_EXPORTED_GPUCalcPDBJacobSphr_MACRO


/***************************************************************************/
/* Calculate the amplitude from of a manual symmetry using Jacobian grids. */
/***************************************************************************/
#define GET_EXPORTED_GPUCalcManSymJacobSphr_MACRO(SUFFIX, T1, T2)										\
int GPUCalcManSymJacobSphr(long long voxels, int thDivs, int phDivs, int numCopies, T1 stepSize,		\
		T1 *inModel, T1 *outData, T2 *ds, T1 *locX, T1 *locY, T1 *locZ,									\
		T1 *locA, T1 *locB, T1 *locC, T2 scale,															\
		progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);				\
int GPUCalcManSymJacobSphr##SUFFIX(												\
		long long voxels, int thDivs, int phDivs, int numCopies, T1 stepSize,							\
		T1 *inModel, T1 *outData, T2 *ds, T1 *locX, T1 *locY, T1 *locZ,									\
		T1 *locA, T1 *locB, T1 *locC, T2 scale,															\
		progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)				\
{																										\
	return GPUCalcManSymJacobSphr(voxels, thDivs, phDivs, numCopies, stepSize, inModel,					\
		outData, ds, locX, locY, locZ, locA, locB, locC, scale,											\
		progfunc, progargs, progmin, progmax, pStop);													\
}

int GPUCalcManSymJacobSphrDD(long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, double *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcManSymJacobSphr(voxels, thDivs, phDivs, numCopies, stepSize, inModel, \
		outData, ds, locX, locY, locZ, locA, locB, locC, scale, \
		progfunc, progargs, progmin, progmax, pStop);													\
}
/* Disable single precision
int GPUCalcManSymJacobSphrDF(long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, float *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcManSymJacobSphr(voxels, thDivs, phDivs, numCopies, stepSize, inModel, \
		outData, ds, locX, locY, locZ, locA, locB, locC, scale, \
		progfunc, progargs, progmin, progmax, pStop);													\
}

int GPUCalcManSymJacobSphrFF(long long voxels, int thDivs, int phDivs, int numCopies, float stepSize, float *inModel, float *outData, float *ds, float *locX, float *locY, float *locZ, float *locA, float *locB, float *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcManSymJacobSphr(voxels, thDivs, phDivs, numCopies, stepSize, inModel, \
		outData, ds, locX, locY, locZ, locA, locB, locC, scale, \
		progfunc, progargs, progmin, progmax, pStop);													\
}
*/

#undef GET_EXPORTED_GPUCalcManSymJacobSphr_MACRO

/**********************************************************************************/
/* Calculate the amplitude from of a space filling symmetry using Jacobian grids. */
/**********************************************************************************/
#define GET_EXPORTED_GPUCalcSpcFillSymJacobSphr_MACRO(SUFFIX, dataFType, interpFType)								\
int GPUCalcSpcFillSymJacobSphr(long long voxels, int thDivs, int phDivs, dataFType stepSize,						\
							   dataFType *inModel, dataFType *outData, interpFType *ds,								\
							   dataFType *vectorMatrix /*This is the three unit cell vectors*/,						\
							   dataFType *repeats /*This is the repeats in the dimensions*/,						\
							   dataFType *innerRots /*This is the three angles of the inner objects rotations*/,	\
							   dataFType *innerTrans /*This is the translation of the inner object*/,				\
							   dataFType scale,																		\
							   progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);	\
int GPUCalcSpcFillSymJacobSphr##SUFFIX(														\
	long long voxels, int thDivs, int phDivs, dataFType stepSize,													\
	dataFType *inModel, dataFType *outData, interpFType *ds,														\
	dataFType *vectorMatrix /*This is the three unit cell vectors*/,												\
	dataFType *repeats /*This is the repeats in the dimensions*/,													\
	dataFType *innerRots /*This is the three angles of the inner objects rotations*/,								\
	dataFType *innerTrans /*This is the translation of the inner object*/,											\
	dataFType scale,																								\
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)								\
{																													\
	return GPUCalcSpcFillSymJacobSphr(voxels, thDivs, phDivs, stepSize, inModel, outData, ds,						\
		  vectorMatrix, repeats, innerRots, innerTrans, scale, progfunc, progargs, progmin, progmax, pStop);		\
}

int GPUCalcSpcFillSymJacobSphrDD(long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, double *ds, double *vectorMatrix, double *repeats, double *innerRots, double *innerTrans, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{																													\
return GPUCalcSpcFillSymJacobSphr(voxels, thDivs, phDivs, stepSize, inModel, outData, ds, \
vectorMatrix, repeats, innerRots, innerTrans, scale, progfunc, progargs, progmin, progmax, pStop);		\
}
/* Disable single precision
int GPUCalcSpcFillSymJacobSphrDF(long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, float *ds, double *vectorMatrix, double *repeats, double *innerRots, double *innerTrans, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcSpcFillSymJacobSphr(voxels, thDivs, phDivs, stepSize, inModel, outData, ds, \
		vectorMatrix, repeats, innerRots, innerTrans, scale, progfunc, progargs, progmin, progmax, pStop);		\
}
int GPUCalcSpcFillSymJacobSphrFF(long long voxels, int thDivs, int phDivs, float stepSize, float *inModel, float *outData, float *ds, float *vectorMatrix, float *repeats, float *innerRots, float *innerTrans, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcSpcFillSymJacobSphr(voxels, thDivs, phDivs, stepSize, inModel, outData, ds, \
		vectorMatrix, repeats, innerRots, innerTrans, scale, progfunc, progargs, progmin, progmax, pStop);		\
}
*/

#undef GET_EXPORTED_GPUCalcSpcFillSymJacobSphr_MACRO


/************************************************************************/
/* Orientation averaging                                                */
/************************************************************************/
#define GET_EXPORTED_GPUCalcMCOAJacobSphr_MACRO(SUFFIX, dataFType, interpFType)							\
	int GPUCalcMCOAJacobSphr(long long voxels, int thDivs, int phDivs, dataFType stepSz,				\
			dataFType *inAmpData, interpFType *inD, dataFType *qs, dataFType *intensities, int qPoints,	\
			long long maxIters, dataFType convergence, progressFunc progfunc, void *progargs,			\
			float progmin, float progmax, int *pStop);													\
int GPUCalcMCOAJacobSphr##SUFFIX(													\
			long long voxels, int thDivs, int phDivs, dataFType stepSz,									\
			dataFType *inAmpData, interpFType *inD, dataFType *qs, dataFType *intensities, int qPoints,	\
			long long maxIters, dataFType convergence, progressFunc progfunc, void *progargs,			\
			float progmin, float progmax, int *pStop)													\
{																										\
	return GPUCalcMCOAJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, qs, intensities,		\
qPoints, maxIters, convergence, progfunc, progargs, progmin, progmax, pStop);							\
}
int GPUCalcMCOAJacobSphrDD(long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, double *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcMCOAJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, qs, intensities, \
		qPoints, maxIters, convergence, progfunc, progargs, progmin, progmax, pStop);							\
}
/* Disable single precision
int GPUCalcMCOAJacobSphrDF(long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, float *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcMCOAJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, qs, intensities, \
		qPoints, maxIters, convergence, progfunc, progargs, progmin, progmax, pStop);							\
}
int GPUCalcMCOAJacobSphrFF(long long voxels, int thDivs, int phDivs, float stepSz, float *inAmpData, float *inD, float *qs, float *intensities, int qPoints, long long maxIters, float convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUCalcMCOAJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, qs, intensities, \
		qPoints, maxIters, convergence, progfunc, progargs, progmin, progmax, pStop);							\
}
*/

#undef GET_EXPORTED_GPUCalcMCOAJacobSphr_MACRO
/************************************************************************/
/* Summation of multiple inner grids using Jacobian grids               */
/************************************************************************/
#define GET_EXPORTED_GPUSumGridsJacobSphr_MACRO(SUFFIX, dataFType, interpFType)							\
	int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, dataFType stepSz,				\
				dataFType **inAmpData,  interpFType **inD, dataFType *trans, dataFType *rots,			\
				int numGrids, dataFType *outAmpData, progressFunc progfunc, void *progargs,				\
				float progmin, float progmax, int *pStop);												\
int GPUSumGridsJacobSphr##SUFFIX(long long voxels, int thDivs, int phDivs,			\
				double stepSz, double **inAmpData,  double **inD, double *trans, double *rots,			\
				int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin,	\
				float progmax, int *pStop)																\
{																										\
	return GPUSumGridsJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, trans, rots,			\
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);								\
}

int GPUSumGridsJacobSphrDD(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, double **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUSumGridsJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, trans, rots, \
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);								\
}
/* Disable single precision
int GPUSumGridsJacobSphrDF(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, float **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUSumGridsJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, trans, rots, \
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);								\
}
int GPUSumGridsJacobSphrFF(long long voxels, int thDivs, int phDivs, double stepSz, float **inAmpData, float **inD, float *trans, float *rots, int numGrids, float *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop)
{
	\
		return GPUSumGridsJacobSphr(voxels, thDivs, phDivs, stepSz, inAmpData, inD, trans, rots, \
		numGrids, outAmpData, progfunc, progargs, progmin, progmax, pStop);								\
}
*/

#undef GET_EXPORTED_GPUSumGridsJacobSphr_MACRO


#define GET_EXPORTED_GPUCalcDebyeV4MAPS_MACRO(SUFFIX, DFT)												\
int GPUCalcDebyeV4MAPS(																					\
	int numQValues, float qMin, float qMax, DFT *outData, int numAtoms, const int *atomsPerIon, 		\
	float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors,						\
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,		\
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);					\
int GPUCalcDebyeV4MAPS##SUFFIX(													\
	int numQValues, float qMin, float qMax, DFT *outData, int numAtoms, const int *atomsPerIon,			\
	float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors,						\
	float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1,		\
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop) {				\
		return																							\
			GPUCalcDebyeV4MAPS(																			\
				numQValues, qMin, qMax, outData, numAtoms, atomsPerIon,									\
				atomLocations, anomalousVals, bBfactors, BFactors,										\
				coeffs, bSol, bSolOnly, atmInd, atmRad, solvED, c1,										\
				progfunc, progargs, progmin, progmax, pStop);											\
	}

int GPUCalcDebyeV4MAPSD(int numQValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	\
		return																							\
		GPUCalcDebyeV4MAPS(\
		numQValues, qMin, qMax, outData, numAtoms, atomsPerIon, \
		atomLocations, anomalousVals, bBfactors, BFactors, \
		coeffs, bSol, bSolOnly, atmInd, atmRad, solvED, c1, \
		progfunc, progargs, progmin, progmax, pStop);											\
}
/* Disable single precision
int GPUCalcDebyeV4MAPSF(int numQValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop)
{
	\
		return																							\
		GPUCalcDebyeV4MAPS(\
		numQValues, qMin, qMax, outData, numAtoms, atomsPerIon, \
		atomLocations, anomalousVals, bBfactors, BFactors, \
		coeffs, bSol, bSolOnly, atmInd, atmRad, solvED, c1, \
		progfunc, progargs, progmin, progmax, pStop);											\
}
*/


#undef GET_EXPORTED_GPUCalcDebyeV4MAPS_MACRO

#pragma endregion