#ifndef __GPUHEAD
#define __GPUHEAD

#include "GPUInterface.h"
#include "GPUDirectCalc.cuh"
#include "GPUHybridCalc.cuh"
#include "Common.h"

#pragma region function declarations

int GPUCalculatePDBSphr(u64 voxels, unsigned short dimx, double qmax, unsigned short sections, double *outData,
	double *loc, u8 *ionInd,
	int numAtoms, double *coeffs, int numCoeffs, bool bSolOnly,
	u8 *atmInd, float *rad, double solvED, u8 solventType,	// FOR DUMMY ATOM SOLVENT
	double *solCOM, u64 *solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	double *outSolCOM, u64 *outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

int GPUCalculatePDBCart(u64 voxels, unsigned short dimx, double qmax, unsigned short sections, double *outData,
	double *loc, u8 *ionInd,
	int numAtoms, double *coeffs, int numCoeffs, bool bSolOnly,
	u8 *atmInd, float *rad, double solvED, u8 solventType,	// FOR DUMMY ATOM SOLVENT
	double *solCOM, u64 *solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	double *outSolCOM, u64 *outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

int GPUCalculateSpaceSymCart(u64 voxels, unsigned short dimx, double stepSize, double *data, const u64 *idx,
	const double av0, const double av1, const double av2, const double bv0, const double bv1,
	const double bv2, const double cv0, const double cv1, const double cv2, const double Na,
	const double Nb, const double Nc,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

int GPUCalculateSpaceSymSphr(u64 voxels, unsigned short dimx, double stepSize, double *data, const u64 *idx,
	const double av0, const double av1, const double av2, const double bv0, const double bv1,
	const double bv2, const double cv0, const double cv1, const double cv2, const double Na,
	const double Nb, const double Nc,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

IGPUCalculator *GPUCreateCalculatorDirect();

bool GPUDirect_SetPDBDLL(Workspace& work, const float4 *atomLocs,
	const unsigned char *ionInd,
	size_t numAtoms, const float *coeffs,
	const unsigned int *atomsPerIon,
	size_t numCoeffs);

bool GPUDirect_PDBAmplitudeDLL(Workspace& work, float3 rotation);

IGPUGridCalculator *GPUCreateCalculatorHybrid();

bool GPUHybrid_PDBAmplitudeDLL(GridWorkspace& work);

bool GPUHybrid_USphereAmplitudeDLL(GridWorkspace& work);


bool GPUHybrid_SetUSphereDLL(GridWorkspace& work, float2 *params, int numLayers, float* extras, int nExtras);

bool GPUDirect_SetSphereParamsDLL(Workspace& work, const float2 *params, int numLayers);

bool GPUHybrid_AmpGridAmplitudeDLL(GridWorkspace& work, double* amp);

bool GPUHybrid_SetSymmetryDLL(GridWorkspace& work, float4 *locs, float4 *rots, int numLocs);

bool GPUHybrid_ManSymmetryAmplitudeDLL(GridWorkspace& work, GridWorkspace& child, float4 trans, float4 rot);

bool GPUSemiHybrid_ManSymmetryAmplitudeDLL(GridWorkspace& work, double* amp, float4 trans, float4 rot);

int GPUCalculateDebyeD(int qValues, double *qVals, double *outData,
	double *loc, u8 *ionInd, int numAtoms, double *coeffs, int numCoeffs, bool bSol,
	u8 * atmInd, double *rad, double solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

/* Disable single precision
int GPUCalculateDebyeF(int qValues, float *qVals, float *outData,
	float *loc, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSol,
	float *rad, float solvED,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
*/
#pragma endregion 




#pragma region horrible things that are an abomination
/*
Calls from elsewhere:
GPUCalcManSymJacobSphereDF x2
GPUCalcDebyeV4MAPSD
GPUCalcDebyeV4MAPSF
GPUCalcMCOAJacobSphereDF
GPUCalcSpacesymmCart
GPUCalcSpcFillSymJacobSphereDF
GPUCalculateDebyeDv2
GPUCalculateDebyeFv2
GPUCalcSumGridsJacobSphereDF

*/


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
	float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs,						\
	double progmin, double progmax, int *pStop);	
//normal:
int GPUCalculateDebyeDv2(int qValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
/* Disable single precision
int GPUCalculateDebyeFv2(int qValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
*/
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
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
int GPUCalcDebyeV3MAPSD( int numQValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);;
/* Disable single precision
int GPUCalcDebyeV3MAPSF( int numQValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);;
*/
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
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
//normal
/* Disable single precision
int GPUCalcPDBJacobSphrFF( u64 voxels, int thDivs, int phDivs, float stepSize, float *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, float solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, float voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, float outerSolED, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcPDBJacobSphrFD(u64 voxels, int thDivs, int phDivs, float stepSize, double *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, float solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, float voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, float outerSolED, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/
int GPUCalcPDBJacobSphrDD(u64 voxels, int thDivs, int phDivs, double stepSize, double *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, double solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, double voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, double outerSolED, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);

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
		progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
//normal
int GPUCalcManSymJacobSphrDD( long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, double *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcManSymJacobSphrDF( long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, float *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcManSymJacobSphrFF( long long voxels, int thDivs, int phDivs, int numCopies, float stepSize, float *inModel, float *outData, float *ds, float *locX, float *locY, float *locZ, float *locA, float *locB, float *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
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
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
//normal
int GPUCalcSpcFillSymJacobSphrDD( long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, double *ds, double *vectorMatrix , double *repeats , double *innerRots , double *innerTrans , double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcSpcFillSymJacobSphrDF( long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, float *ds, double *vectorMatrix , double *repeats , double *innerRots , double *innerTrans , double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcSpcFillSymJacobSphrFF( long long voxels, int thDivs, int phDivs, float stepSize, float *inModel, float *outData, float *ds, float *vectorMatrix , float *repeats , float *innerRots , float *innerTrans , float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
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
			float progmin, float progmax, int *pStop);
//normal
int GPUCalcMCOAJacobSphrDD( long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, double *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcMCOAJacobSphrDF( long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, float *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcMCOAJacobSphrFF( long long voxels, int thDivs, int phDivs, float stepSz, float *inAmpData, float *inD, float *qs, float *intensities, int qPoints, long long maxIters, float convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
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
				float progmax, int *pStop);
//normal
int GPUSumGridsJacobSphrDD(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, double **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUSumGridsJacobSphrDF(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, float **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUSumGridsJacobSphrFF(long long voxels, int thDivs, int phDivs, double stepSz, float **inAmpData, float **inD, float *trans, float *rots, int numGrids, float *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
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
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
//normal
int GPUCalcDebyeV4MAPSD(int numQValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
/* Disable single precision
int GPUCalcDebyeV4MAPSF(int numQValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
*/
#undef GET_EXPORTED_GPUCalcDebyeV4MAPS_MACRO

#pragma endregion


#pragma region gpu things


void GPUReset(); //normal declaration

#pragma endregion

#endif