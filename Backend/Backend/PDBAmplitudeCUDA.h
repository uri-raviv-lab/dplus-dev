#ifndef PDBAMPLITUDECUDA_H__
#define PDBAMPLITUDECUDA_H__

#include "Common.h"
#include "../GPU/Atomic Form Factor.h"
#include "../GPU/electron Atomic Form Factor.h"

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, double *outData,
	std::vector<float4> atomLocations,
	atomicFFCalculator &affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, float *outData,
	std::vector<float4> atomLocations,
	atomicFFCalculator &affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, double* outData,
	std::vector<float4> atomLocations,
	electronAtomicFFCalculator& affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop);

int PDBJacobianGridAmplitudeCalculation(
	u64 voxels, int thDivs, int phDivs, float stepSize, float* outData,
	std::vector<float4> atomLocations,
	electronAtomicFFCalculator& affCalculator,
	std::vector<float4> solCOM, std::vector<int4> solDims, float solvED, double voxStep,	// For voxel based solvent
	std::vector<float4> outSolCOM, std::vector<int4> outSolDims, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop);


#endif