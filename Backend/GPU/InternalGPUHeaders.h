#pragma once

#include "Common.h"
#include "GPUInterface.h"

#pragma region///////////////////forward declarations://///////////////////////

bool GPUDirect_SetPDB(Workspace& work, const float4 *atomLocs,
	const unsigned char *ionInd,
	size_t numAtoms, const float *coeffs,
	const unsigned int *atomsPerIon,
	size_t numCoeffs);

bool GPUDirect_PDBAmplitude(Workspace& work, float3 rotation);

bool GPUHybrid_PDBAmplitude(GridWorkspace& work);


bool GPUHybrid_SetPDB(GridWorkspace& work, const std::vector<float4>& atomLocs,
	const std::vector<float>& atomicFormFactors_q_major_ordering,
// 	const std::vector<unsigned char>& ionInd,
// 	const std::vector<float>& coeffs,
// 	const std::vector<int>& atomsPerIon,
// 	int solventType,
// 	std::vector<unsigned char>& atmInd, std::vector<float>& atmRad, double solvED,
	double solventRad, // For dummy atom solvent
	float4 *solCOM, int4 *solDims, int solDimLen, float voxStep, // For voxel based solvent
	float4 *outSolCOM, int4 *outSolDims, int outSolDimLen, float outerSolED // For outer solvent layer
	);

bool GPUHybrid_USphereAmplitude(GridWorkspace& work);
bool GPUHybrid_SetUSphere(GridWorkspace& work, float2 *params, int numLayers, float* extras, int nExtras);

bool GPUDirect_SetSphereParams(Workspace& work, const float2 *params, int numLayers);
bool GPUHybrid_AmpGridAmplitude(GridWorkspace& work, double* amp);
bool GPUHybrid_SetSymmetry(GridWorkspace& work, float4 *locs, float4 *rots, int numLocs);
bool GPUHybrid_ManSymmetryAmplitude(GridWorkspace& work, GridWorkspace& child, float4 trans, float4 rot);
bool GPUSemiHybrid_ManSymmetryAmplitude(GridWorkspace& work, double* amp, float4 trans, float4 rot);


/************************************************************************/
/* Calculate the intensity from a PDB using the Debye formula.          */
/************************************************************************/
// /*extern "C" */int GPUCalcDebye(int qValues, double *qVals, double *outData,
// 	double *loc, u8 *ionInd, int numAtoms, double *coeffs, int numCoeffs, bool bSol,
// 	double *rad, double solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
// 
// // Float
// /*extern "C"*/ int GPUCalcDebye(int qValues, float *qVals, float *outData,
// 	float *loc, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSol,
// 	float *rad, float solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

#pragma endregion 


#pragma region terrible things
//forward:
int GPUCalcDebyeV2(int numQValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
int GPUCalcDebyeV2(int numQValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *loc, u8 *ionInd, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

//forward:
int GPUCalcPDBJacobSphr(u64 voxels, int thDivs, int phDivs, double stepSize, double *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, double solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, double voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, double outerSolED, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcPDBJacobSphr(u64 voxels, int thDivs, int phDivs, float stepSize, double *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, float solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, float voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, float outerSolED, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcPDBJacobSphr(u64 voxels, int thDivs, int phDivs, float stepSize, float *outData, float *locX, float *locY, float *locZ, u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly, u8 * atmInd, float *atmRad, float solvED, u8 solventType, float4 *solCOM, int4 *solDims, u64 solDimLen, float voxStep, float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, float outerSolED, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/

//forward
int GPUCalcManSymJacobSphr(long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, double *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcManSymJacobSphr(long long voxels, int thDivs, int phDivs, int numCopies, double stepSize, double *inModel, double *outData, float *ds, double *locX, double *locY, double *locZ, double *locA, double *locB, double *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcManSymJacobSphr(long long voxels, int thDivs, int phDivs, int numCopies, float stepSize, float *inModel, float *outData, float *ds, float *locX, float *locY, float *locZ, float *locA, float *locB, float *locC, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/

//forward
int GPUCalcSpcFillSymJacobSphr(long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, double *ds, double *vectorMatrix, double *repeats, double *innerRots, double *innerTrans, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcSpcFillSymJacobSphr(long long voxels, int thDivs, int phDivs, float stepSize, float *inModel, float *outData, float *ds, float *vectorMatrix, float *repeats, float *innerRots, float *innerTrans, float scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcSpcFillSymJacobSphr(long long voxels, int thDivs, int phDivs, double stepSize, double *inModel, double *outData, float *ds, double *vectorMatrix, double *repeats, double *innerRots, double *innerTrans, double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/

//forward
int GPUCalcMCOAJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, double *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUCalcMCOAJacobSphr(long long voxels, int thDivs, int phDivs, float stepSz, float *inAmpData, float *inD, float *qs, float *intensities, int qPoints, long long maxIters, float convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUCalcMCOAJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz, double *inAmpData, float *inD, double *qs, double *intensities, int qPoints, long long maxIters, double convergence, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/

//forward
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, double **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
/* Disable single precision
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, double stepSz, double **inAmpData, float **inD, double *trans, double *rots, int numGrids, double *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
int GPUSumGridsJacobSphr(long long voxels, int thDivs, int phDivs, float stepSz, float **inAmpData, float **inD, float *trans, float *rots, int numGrids, float *outAmpData, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
*/

//forward
int GPUCalcDebyeV4MAPS(int numQValues, float qMin, float qMax, double *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
int GPUCalcDebyeV4MAPS(int numQValues, float qMin, float qMax, float *outData, int numAtoms, const int *atomsPerIon, float4 *atomLocations, float2 *anomalousVals, bool bBfactors, float *BFactors, float *coeffs, bool bSol, bool bSolOnly, char* atmInd, float *atmRad, float solvED, float c1, progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);


#pragma endregion

void ResetGPU(); //forward
int GetNumGPUs(); //forward