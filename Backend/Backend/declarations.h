#ifndef __DECLAR
#define __DECLAR

#include "../GPU/GPUInterface.h"
#include "Grid.h"
#include "BackendInterface.h"
#include "md5.h"

//symmetries.cpp

typedef int(*GPUCalculateSymmetry_t)(u64 voxels, unsigned short dimx, double stepSize, double *data, const u64 *idx,
	const double av0, const double av1, const double av2, const double bv0, const double bv1,
	const double bv2, const double cv0, const double cv1, const double cv2, const double Na,
	const double Nb, const double Nc,
	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);


//template <typename Ftype1, typename Ftype2>
typedef double GRIDDATATYPE;
typedef double MODELPARAMTYPE;
typedef double INTERPOLATIONDATATYPE;

//
typedef int(*GPUCalculateManSymmetry_t)(long long voxels, int thDivs, int phDivs, int numCopies, MODELPARAMTYPE stepSize,
	GRIDDATATYPE *inModel, GRIDDATATYPE *outData, INTERPOLATIONDATATYPE *ds,
	MODELPARAMTYPE *locX, MODELPARAMTYPE *locY, MODELPARAMTYPE *locZ,
	MODELPARAMTYPE *locA, MODELPARAMTYPE *locB, MODELPARAMTYPE *locC,
	INTERPOLATIONDATATYPE scale,
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);


typedef int(*GPUCalculateSpcFllSymmetry_t)(long long voxels, int thDivs, int phDivs, MODELPARAMTYPE stepSize,
	GRIDDATATYPE *inModel, GRIDDATATYPE *outData, INTERPOLATIONDATATYPE *ds,
	MODELPARAMTYPE *vectorMatrix /*This is the three unit cell vectors*/,
	MODELPARAMTYPE *repeats /*This is the repeats in the dimensions*/,
	MODELPARAMTYPE *innerRots /*This is the three angles of the inner objects rotations*/,
	MODELPARAMTYPE *innerTrans /*This is the translation of the inner object*/,
	MODELPARAMTYPE scale,
	progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);


//
typedef bool(*GPUHybridSetSymmetries_t)(GridWorkspace& work, float4 *locs, float4 *rots, int numLocs);

//
typedef bool(*GPUHybridManSymmetryAmplitude_t)(GridWorkspace& work, GridWorkspace& child, float4 trans, float4 rot);

//
typedef bool(*GPUSemiHybridManSymmetryAmplitude_t)(GridWorkspace& work, double* amp, float4 trans, float4 rot);


//

extern GPUCalculateSymmetry_t gpuCalcSym;
extern GPUCalculateManSymmetry_t gpuCalcManSym;
extern GPUCalculateSpcFllSymmetry_t gpuCalcSpcFllSym;
extern GPUHybridSetSymmetries_t gpuHybridSetSymmetry;
extern GPUHybridManSymmetryAmplitude_t gpuManSymmetryHybridAmplitude;
extern GPUSemiHybridManSymmetryAmplitude_t gpuManSymmetrySemiHybridAmplitude;


#endif