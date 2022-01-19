#define AMP_EXPORTER

#include "../backend_version.h"

#include "Amplitude.h"
#include <math.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "boost/filesystem/fstream.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/uniform_int_distribution.hpp"
#include "boost/random/random_device.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/filesystem.hpp"
#include "boost/multi_array.hpp"
#include "boost/cstdint.hpp"
#include <boost/lexical_cast.hpp>
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iomanip>
#include <queue>
#include <chrono>
#include <algorithm>    // std::sort
#include "md5.h"

#include "PDBAmplitudeCUDA.h"
#include "AmplitudeCache.h"

#include "../GPU/GPUInterface.h"
#include <cuda_runtime.h> // For getdevicecount and memory copies
#include <vector_functions.h>

#ifdef _WIN32
#include <windows.h> // For LoadLibrary
#pragma comment(lib, "user32.lib") // TEMP TODO REMOVE
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym 
#endif

#include "Grid.h"

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

#include "BackendInterface.h"
#include "declarations.h"
#include "GPUHeader.h"
#include "InternalGPUHeaders.h" // The above should end up being remove in favor of the internal header now that there is no separate GPU dll
#include "UseGPU.h" 

#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>

using Eigen::ArrayXd;
using Eigen::Matrix3d;
namespace fs = boost::filesystem;

inline Eigen::ArrayXXf SincArrayXXf(const Eigen::ArrayXXf& x)
{
	Eigen::ArrayXXf res = x.sin() / x;
	res = (res != res).select(1.f, res);
	return res;
}

static const std::map < SolventSpace::ScalarType, Eigen::Array3i >
static_color_map = {
	{ 0, { 0, 0, 255 } },		// Blue, uninitialized
	{ 1, { 0, 255, 0 } },		// Green, atoms
	{ 2, { 255, 0, 0 } },		// Red, Probe around the atoms
	{ 3, { 135, 196, 250 } },	// Light Blue, Far solvent
	{ 4, { 255, 105, 180 } },	// Pink, Solvation
	{ 5, { 255, 255, 0 } },		// Yellow
	{ 6, { 211, 211, 211 } }	// Grey
};

static string GetTimeString(clock_t beg, clock_t en) {
	clock_t totu, tot = en - beg;
	std::stringstream res;
	int sec, minut, hr, gh = 0;

	totu = tot;
	hr = int(double(totu) / double(CLOCKS_PER_SEC) / 3600.0);
	totu -= hr * 3600 * CLOCKS_PER_SEC;
	minut = int(double(totu) / double(CLOCKS_PER_SEC) / 60.0);
	totu -= minut * 60 * CLOCKS_PER_SEC;
	sec =  int(double(totu) / double(CLOCKS_PER_SEC));
	totu -= sec * CLOCKS_PER_SEC;

	if(hr > 0) {
		gh++;
		res << " " << hr << " hour" << (hr > 1 ? "s" : "");
	}
	if(minut > 0) {
		gh++;
		res << " " << minut << " minute" << (minut > 1 ? "s" : "");
	}
	if(sec > 0 && gh < 2) {
		gh++;
		res << " " << sec << " second" << (sec > 1 ? "s" : "");
	}
	if(gh < 2) {
		res << " " << totu * double(1000.0 / double(CLOCKS_PER_SEC)) << " ms";
	}

	//if(tot > 10000)
	//	res << (tot / CLOCKS_PER_SEC) << " seconds";
	//else
	//	res << tot * double(1000.0 / double(CLOCKS_PER_SEC)) << " ms";
	return string(res.str().c_str());
}

template <typename T>
string GetChronoTimeString(std::chrono::time_point<T> beg, std::chrono::time_point<T> en)
{
	auto tot = en - beg;
	auto totu = tot;

	std::chrono::hours hours(std::chrono::duration_cast<std::chrono::hours>(totu));
	std::chrono::minutes minutes(std::chrono::duration_cast<std::chrono::minutes>(totu -= hours));
	std::chrono::seconds seconds(std::chrono::duration_cast<std::chrono::seconds>(totu -= minutes));
	std::chrono::milliseconds milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(totu -= seconds));


	std::stringstream res;
	int ms, sec, minut, hr, gh = 0;
	hr = hours.count();
	minut = minutes.count();
	sec = seconds.count();
	ms = milliseconds.count();

	if (hr > 0) {
		gh++;
		res << " " << hr << " hour" << (hr > 1 ? "s" : "");
	}
	if (minut > 0) {
		gh++;
		res << " " << minut << " minute" << (minut > 1 ? "s" : "");
	}
	if (sec > 0 && gh < 2) {
		gh++;
		res << " " << sec << " second" << (sec > 1 ? "s" : "");
	}
	if (gh < 2) {
		res << " " << ms << " ms";
	}

	return string(res.str().c_str());
}

#pragma region GPU Stuff

struct SolventBoxEntry {
	int4 len;
	float4 loc;
};
bool SortSolventBoxEntry(const SolventBoxEntry& a, const SolventBoxEntry& b) {
	if(a.len.x < b.len.x)
		return true;
	if(a.len.x > b.len.x)
		return false;
	if(a.len.y < b.len.y)
		return true;
	if(a.len.y > b.len.y)
		return false;
	if(a.len.z < b.len.z)
		return true;
	if(a.len.z > b.len.z)
		return false;
	//From here on, it's just to avoid an assertion failure
	if(a.loc.x < b.loc.x)
		return true;
	if(a.loc.x > b.loc.x)
		return false;
	if(a.loc.y < b.loc.y)
		return true;
	if(a.loc.y > b.loc.y)
		return false;
	if(a.loc.z < b.loc.z)
		return true;
	if(a.loc.z > b.loc.z)
		return false;
	return true;
}

typedef int (*GPUCalculatePDB_t)(u64 voxels, unsigned short dimx,
								 double qmax, unsigned short sections, double *outData,
								 double *loc, u8 *ionInd,
								 int numAtoms, double *coeffs, int numCoeffs, bool bSolOnly,
								 u8 * atmInd, float *rad, double solvED, u8 solventType,	// FOR DUMMY ATOM SOLVENT
								 double *solCOM, u64 *solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
								 double *outSolCOM, u64 *outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
								 progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);

GPUCalculatePDB_t gpuCalcPDB = NULL;

typedef double IN_JF_TYPE;
typedef int (*GPUCalculatePDBJ_t)(u64 voxels, int thDivs, int phDivs, IN_JF_TYPE stepSize, double *outData, float *locX, float *locY, float *locZ,
								  u8 *ionInd, int numAtoms, float *coeffs, int numCoeffs, bool bSolOnly,
								  u8 * atmInd, float *atmRad, IN_JF_TYPE solvED, u8 solventType,// FOR DUMMY ATOM SOLVENT
								  float4 *solCOM, int4 *solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
								  float4 *outSolCOM, int4 *outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
								  double scale, progressFunc progfunc, void *progargs, float progmin, float progmax, int *pStop);
GPUCalculatePDBJ_t gpuCalcPDBJ = NULL;

typedef bool (*GPUDirectSetPDB_t)(Workspace& work, const float4 *atomLocs,
								  const unsigned char *ionInd,
								  size_t numAtoms, const float *coeffs,
								  int *atomsPerIon,
								  size_t numCoeffs);
GPUDirectSetPDB_t gpuSetPDB = NULL;

// typedef bool(*GPUHybridSetPDB_t)(GridWorkspace& work, const std::vector<float4>& atomLocs,
// 								  const std::vector<unsigned char>& ionInd,
// 								  const std::vector<float>& coeffs,
// 								  std::vector<int>& atomsPerIon,
// 								  int solventType,
// 								  std::vector<u8>& atmInd, std::vector<float>& atmRad, double solvED, // For dummy atom solvent
// 								  float4 *solCOM, int4 *solDims, int solDimLen, float voxStep,	// For voxel based solvent
// 								  float4 *outSolCOM, int4 *outSolDims, int outSolDimLen, float outerSolED	// For outer solvent layer
// 								  );
// GPUHybridSetPDB_t gpuHybridSetPDB = NULL;

typedef bool (*GPUDirectPDBAmplitude_t)(Workspace& work, float3 rotation);
GPUDirectPDBAmplitude_t gpuPDBAmplitude = NULL;

typedef bool (*GPUHybridPDBAmplitude_t)(GridWorkspace& work);
GPUHybridPDBAmplitude_t gpuPDBHybridAmplitude = NULL;

#pragma endregion

#pragma region CPDB Reader class

PDBAmplitude::~PDBAmplitude() {
}

PDBAmplitude::PDBAmplitude(string filename, bool bCenter, string anomalousFilename, int model /*= 0*/) : Amplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb = PDBReader::PDBReaderOb<float>(filename, bCenter, model, anomalousFilename);

	bCentered = bCenter;

	SetPDBHash();
}

PDBAmplitude::PDBAmplitude(const char *buffer, size_t buffSize, const char *filenm, size_t fnSize, bool bCenter, const char *anomalousFilename, size_t anomBuffSize, int model) : Amplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb.fn.assign(filenm, fnSize);

	try
	{
		if (anomalousFilename && anomBuffSize > 0)
			status = pdb.readAnomalousbuffer(anomalousFilename, anomBuffSize);

		if (PDB_OK == status)
			status = pdb.readPDBbuffer(buffer, buffSize, bCenter, model);
	}
	catch (PDBReader::pdbReader_exception &e)
	{
		status = PDB_READER_ERRS::ERROR_IN_PDB_FILE;
		throw backend_exception(e.GetErrorCode(), e.GetErrorMessage().c_str());
	}
	bCentered = bCenter;

	SetPDBHash();
}

PDBAmplitude::PDBAmplitude() {
	gridStatus = AMP_UNINITIALIZED;

	status = UNINITIALIZED;
	initialize();
}

std::complex<FACC> PDBAmplitude::calcAmplitude(FACC qx, FACC qy, FACC qz) {
	FACC q = sqrt(qx*qx + qy*qy + qz*qz), resI = 0.0, resR = 0.0, aff = 0.0;
	Eigen::Matrix<float, 3, 1> qVec(qx, qy, qz);

	// Atomic form factors and dummy solvent
	if(bitwiseCalculationFlags & (CALC_ATOMIC_FORMFACTORS | CALC_DUMMY_SOLVENT) ) 
	{
		double phase = 0.0;
		int xSz = int(pdb.x.size());

		Eigen::Array<float, -1, 1> phases;
		phases = (atomLocs * qVec).array();
		
		// TODO: Think of a better way to reduce the branch, code size and copy pasta
		if (pdb.haveAnomalousAtoms)
		{
			Eigen::Array<std::complex<float>, Eigen::Dynamic, 1> affs(xSz);
			affCalculator.GetAllAFFs((float2*)(affs.data()), q);

			std::complex<float> tmpA = (affs * phases.sin().cast<std::complex<float>>()).sum();
			std::complex<float> tmpB = (affs * phases.cos().cast<std::complex<float>>()).sum();
			resI += tmpA.imag() + tmpB.imag();
			resR += tmpA.real() + tmpB.real();
		}
		else
		{
			Eigen::Array<float, Eigen::Dynamic, 1> affs(xSz);
			affCalculator.GetAllAFFs(affs.data(), q);

			resI += (affs * phases.sin()).sum();
			resR += (affs * phases.cos()).sum();
		}
	} // if bOnlySolvent

	// Subtract the solvent using voxels
	if(this->bSolventLoaded && this->solventBoxDims.size() > 0 &&
		(this->pdb.atmRadType == RAD_CALC || this->pdb.atmRadType == RAD_EMP || 
		this->pdb.atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb.atmRadType == RAD_VDW)
		) {
		int numVoxels = solventBoxDims_array.size();
		Eigen::ArrayXf vva;
		Eigen::ArrayXf vphase;

		vva = (
			(
			SincArrayXXf((solventBoxDims_array.matrix() * (0.5f * qVec).asDiagonal()).array()) * solventBoxDims_array
			).rowwise().prod() * 
			(float) (solventED)
			);
		vphase = (solventBoxCOM_array.matrix() * qVec.asDiagonal()).rowwise().sum();
		resI -= (vva * vphase.sin()).sum();
		resR -= (vva * vphase.cos()).sum();

	}
	
	// Add the outer solvent using voxels
	if (this->bSolventLoaded && this->outerSolventBoxDims.size() > 0 && outerSolventED != 0.0) {

		int numVoxels = outerSolventBoxDims_array.size();
		Eigen::ArrayXf vva;
		Eigen::ArrayXf vphase;

		vva = (
			(
			SincArrayXXf((outerSolventBoxDims_array.matrix() * (0.5f * qVec).asDiagonal()).array()) * outerSolventBoxDims_array
			).rowwise().prod() * 
			(float) (outerSolventED - solventED)
			);
		vphase = (outerSolventBoxCOM_array.matrix() * qVec.asDiagonal()).rowwise().sum();
		resI += (vva * vphase.sin()).sum();
		resR += (vva * vphase.cos()).sum();
	}

	return std::complex<FACC>(resR, resI) * scale;
}

void PDBAmplitude::calculateGrid(FACC qmax, int sections, progressFunc progFunc, void *progArgs, double progMin, double progMax, int *pStop) {
	if (gridStatus == AMP_CACHED) {
		if (PDB_OK == ReadAmplitudeFromCache())
			return;
	}

	// Hybrid
	if (!bUseGrid)
		return;

	/*
	//IF THIS COMMENTED SECTION IS UNCOMMENTED, MUST REPLACE GET PROCADDRESS CALLS WITH DIRECT CALLS TO FUNCTION
		if (!g_gpuModule) {
		load_gpu_backend(g_gpuModule);
		}
		if(g_gpuModule)
		{
		#ifdef USE_SPHERE_GRID
		#ifdef USE_JACOBIAN_SPHERE_GRID
		// double, double
		gpuCalcPDBJ = (GPUCalculatePDBJ_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalcPDBJacobSphrDD");
		#else
		gpuCalcPDB = (GPUCalculatePDB_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculatePDBSphr");
		#endif
		#else
		gpuCalcPDB = (GPUCalculatePDB_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculatePDBCart");
		#endif // USE_SPHERE_GRID
		}*/

	// 	Amplitude::calculateGrid(qmax, sections, progFunc, progArgs, progMin, progMax, pStop);
	// 	return;
	
	// TODO: Make this into a function that runs on startup and checks for specific compute capabilities
	int nDevices;
	auto getCountRes = UseGPUDevice(&nDevices);

	if (!bUseGPU || nDevices == 0 || getCountRes != cudaSuccess || g_useGPUAndAvailable == false)
	{
		Amplitude::calculateGrid(qmax, sections, progFunc, progArgs, progMin, progMax, pStop);
		return;
	}

	PDB_READER_ERRS dbg = getError();
	if (!bUseGrid) {
		std::cout << "Not using grid";
		return;
	}

	InitializeGrid(qmax, sections);

	u64 voxels = grid->GetRealSize() / (sizeof(double) * 2);
	const unsigned int dimx = grid->GetDimX();

	std::vector<float4> atomLocations(pdb.sortedX.size());
	for (int i = 0; i < pdb.sortedX.size(); i++)
	{
		atomLocations[i].x = pdb.sortedX[i];
		atomLocations[i].y = pdb.sortedY[i];
		atomLocations[i].z = pdb.sortedZ[i];
	}

	std::vector<float4> solCOM;
	std::vector<int4> solDims;

	std::vector<float4> outSolCOM;
	std::vector<int4> outSolDims;

	PrepareParametersForGPU( solCOM, solDims, outSolCOM, outSolDims);

	size_t ssz = solventBoxDims.size();
	size_t osz = outerSolventBoxDims.size();


/*
#ifdef USE_JACOBIAN_SPHERE_GRID
	int vecSize = pdb.ionInd.size();

	int gpuRes = gpuCalcPDBJ(voxels, grid->GetDimY(1) - 1, grid->GetDimZ(1,1), grid->GetStepSize(),
		grid->GetDataPointer(), &pdb.sortedX[0], &pdb.sortedY[0], &pdb.sortedZ[0], &pdb.sortedIonInd[0], vecSize, 
		pdb.sortedCoeffs.data(), pdb.sortedCoeffs.size() / 9, pdb.bOnlySolvent, &pdb.sortedAtmInd[0], &(pdb.rad->at(0)),
		solventED, int(pdb.GetRadiusType()),
		&solCOM[0], &solDims[0], ssz, voxelStep,
		&outSolCOM[0], &outSolDims[0], osz, outerSolventED, this->scale,
		progFunc, progArgs, progMin, progMax, pStop);
		((JacobianSphereGrid*)(grid))->CalculateSplines();
#else

	
	int gpuRes = gpuCalcPDB(voxels, dimx, qmax, sections,
		grid->GetPointerWithIndices(), &loc[0], &sortedIonInd[0],
		vecSize, newCoeffs.data(), newCoeffs.size(), pdb.bOnlySolvent,
		&(sortedAtmInd[0]), &(pdb.rad->at(0)), solventED, int(pdb.GetRadiusType()),
		&solBoxCOM[0], &solBoxDims[0], solventBoxDims.size(), voxelStep,
		&outSolBoxCOM[0], &outSolBoxDims[0], outerSolventBoxDims.size(), outerSolventED,
		progFunc, progArgs, progMin, progMax, pStop);
#endif // USE_JACOBIAN_SPHERE_GRID

*/
	int gpuRes = PDBJacobianGridAmplitudeCalculation
		(
		voxels, grid->GetDimY(1) - 1, grid->GetDimZ(1, 1), grid->GetStepSize(),
		grid->GetDataPointer(), atomLocations, affCalculator,
		solCOM, solDims, solventED, voxelStep, outSolCOM, outSolDims, outerSolventED,
		scale, progFunc, progArgs, progMin, progMax, pStop
		);

	if(gpuRes != 0) {
		std::cout << "Error in kernel: " << gpuRes << ". Starting CPU calculations." << std::endl;
		Amplitude::calculateGrid(qmax, sections, progFunc, progArgs, progMin, progMax, pStop);
		return;
	}

	((JacobianSphereGrid*)(grid))->CalculateSplines();

	gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;

}

std::complex<FACC> PDBAmplitude::calcAmplitude(int indqx, int indqy, int indqz) {
	// NOTEPAD:
	/*	n is the number of atoms
			Needed parameters:
				qx, qy, qz	--> can be obtained with 3 indices and a map[actualGridSize^2] qmax and stepSize
				x[n], y[n], z[n], ionInd[n]
				coefs[9] for atomicFF

				Solvent voxel:
					voxelStep
					solventED
					solventBoxDims[3xM]	--> M is an arbitrary number obtained at runtime
					solventBoxCOM[3xM]


				Solvent dummy atoms:
					solventED
					atmInd[n]
					rad[] --> can be either "n", and then we don't need atmInd or 108

				Outer solvent:
					k	--> Arbitrary number obtained at runtime, number of boxes to describe outer solvent
					outer solvent dimensions [3][k]
					outer solvent location [3][k]
					voxelStep
					outerSolventED
					
					
	*/

	double qx, qy, qz;
	grid->IndicesToVectors(indqx, indqy, indqz, qx, qy, qz);
	FACC q = sqrt(qx*qx + qy*qy + qz*qz), resI = 0.0, resR = 0.0, aff = 0.0;
	if(!pdb.bOnlySolvent) {
		double phase = 0.0;
		int xSz = (int)pdb.x.size();
		for(int i = 0; i < xSz; i++) {
			phase = qx * pdb.x[i] + qy * pdb.y[i] + qz * pdb.z[i];
			aff = atomicFF(q/(10.0), pdb.ionInd[i]);

			resI += aff * sin(phase);
			resR += aff * cos(phase);
		}
	} // if bOnlySolvent

	// Subtract the solvent using voxels
	if(this->bSolventLoaded && this->solventBoxDims.size() > 0 &&
		(this->pdb.atmRadType == RAD_CALC || this->pdb.atmRadType == RAD_EMP || 
		this->pdb.atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb.atmRadType == RAD_VDW)
		) {
			double solR = 0.0, solI = 0.0, phase, va;
			int sbd = (int)solventBoxDims.size();

			for(int i = 0; i < sbd; i++) {
				va = (sinc(qx * double((solventBoxDims[i])[0]) * this->voxelStep / 2.0)
					* sinc(qy * double((solventBoxDims[i])[1]) * this->voxelStep / 2.0)
					* sinc(qz * double((solventBoxDims[i])[2]) * this->voxelStep / 2.0)
					* this->voxelStep * double((solventBoxDims[i])[0])
					* this->voxelStep * double((solventBoxDims[i])[1])
					* this->voxelStep * double((solventBoxDims[i])[2]));

				va *= this->solventED;

				phase = (qx * (this->solventBoxCOM[i])[0] +
					qy * (this->solventBoxCOM[i])[1] +
					qz * (this->solventBoxCOM[i])[2]);

				solI += va * sin(phase);
				solR += va * cos(phase);
			}

			resR -= solR;
			resI -= solI;
	}

	// Subtract the solvent using dummy atoms
	if (this->pdb.atmRadType == RAD_DUMMY_ATOMS_ONLY && solventED != 0.0) {
		double solR = 0.0, solI = 0.0, phase, gi;
		int xSz = pdb.x.size();
		for(int i = 0; i < xSz; i++) {
			phase = qx * pdb.x[i] + qy * pdb.y[i] + qz * pdb.z[i];
#ifdef USE_FRASER
			gi = /*4.1887902047863909846*/ /*4\pi/3*/ 5.56832799683 /*pi^1.5*/ * (*this->pdb.rad)[pdb.atmInd[i]] * (*this->pdb.rad)[pdb.atmInd[i]] *
				(*this->pdb.rad)[pdb.atmInd[i]];
			gi *= exp(-sq((*this->pdb.rad)[pdb.atmInd[i]] * q / 2.0));
#else
			gi = 4.1887902047863909846 * (*pdb.rad)[pdb.atmInd[i]] * (*pdb.rad)[pdb.atmInd[i]] * (*pdb.rad)[pdb.atmInd[i]]
				* exp(-(0.20678349696647 * sq((*pdb.rad)[pdb.atmInd[i]] * q))); // 0.206... = (4pi/3)^(2/3) / (4pi)
#endif
			solI += gi * sin(phase);
			solR += gi * cos(phase);
		} // for i
		resR -= solR * this->solventED;
		resI -= solI * this->solventED;

	} // if RAD_DUMY_ATOMS_ONLY

	// Add the outer solvent using voxels
	if(this->bSolventLoaded && this->outerSolventBoxDims.size() > 0) {
		double solR = 0.0, solI = 0.0, phase, va;
		int osbd = (int)outerSolventBoxDims.size();

		for(int i = 0; i < osbd; i++) {
			va = (sinc(qx * double((outerSolventBoxDims[i])[0]) * this->voxelStep / 2.0)
				* sinc(qy * double((outerSolventBoxDims[i])[1]) * this->voxelStep / 2.0)
				* sinc(qz * double((outerSolventBoxDims[i])[2]) * this->voxelStep / 2.0)
				* this->voxelStep * double((outerSolventBoxDims[i])[0])
				* this->voxelStep * double((outerSolventBoxDims[i])[1])
				* this->voxelStep * double((outerSolventBoxDims[i])[2]));

			va *= this->outerSolventED;

			phase = (qx * (this->outerSolventBoxCOM[i])[0] +
				qy * (this->outerSolventBoxCOM[i])[1] +
				qz * (this->outerSolventBoxCOM[i])[2]);

			solI += va * sin(phase);
			solR += va * cos(phase);
		}

		resR += solR;
		resI += solI;
	}

	if (this->pdb.getBOnlySolvent() && solventED != 0.0) {
		return -std::complex<FACC>(resR, resI);
	}
	return std::complex<FACC>(resR, resI);

}

FACC PDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998); 
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

	for(int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2*i)) * exp(-atmFFcoefs(elem, (2*i) + 1) * sqq);
	
	return res;
}

/* This part is supposed to show the changes that will have to be made for electron diffraction (at least the mathematical part of it):

// Not sure what is better or easier but, we can either change both the x-ray and elec to be a 5-Gaussian (as shown underneath) and change the atomic form factors so that for x-ray the 10th col is 0, or instead build two separate options for calculations that will be chosen depending on what the user wants.

// In the direct method we will want to solve the integral:
m_0 = 9.1093837015e-31 // kg
e = 1.602176634e-19 // C
h = 6.62607015e-34 //in kg*m^2/s (Js), will probably have to change to A^2 or nm^2 (but not sure)
f(q) = (8*pi^2*m_0*e)/(h^2) Integral{r^2 * phi(r) * sinc(4*pi*q*r)dr, bounds = [0, infty] } // Not sure we'll want to have this as an option...
// sinc(x) = sin(x)/x


FACC PDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q // According to Doyle & Turner (1967) the exp was defined without the 1/(4pi)^2

	for(int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2*i)) * exp(-atmFFcoefs(elem, (2*i) + 1) * sqq);
	return res;

void PDBAmplitude::initialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
#pragma region Electronic Atomic form factor coefficients
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.057, 18.9525, 0.1195, 38.6269, //H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653, //He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337, //Li
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273, //Be
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314, //B
		0.0893, 0.2465, 0.2563, 1.7100, 0.5770, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523, //C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715, 48.1431, //N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.643, 0.6990, 12.7105, 0.2039, 32.4726 //O
		;
#pragma endregion
*/

/* This part is supposed to show the changes that will have to be made for neutron diffraction (at least the mathematical part of it):

// Since in this case we do noth have a Gaussian, maybe it is better to build each independently of the other.
// Do we want to put a resolution function here as shown in Pedersen, Posselt, Mortesen (1990)?


FACC PDBAmplitude::atomicFF(int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	
	// It might be interesting to enter a weighted sum of all the isotopes i.e. res = sum(p_i * b_i)
	// Also, put a default (default abundance in nature) and a possibility for the user to enter it's own abundance (maybe like with anomalous scattering for x-ray).
	
	res = atmFFcoefs(elem);
	return res;

void PDBAmplitude::initialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 1);
	// 'i' is the imaginary number
	atmFFcoefs << -3.7406 //H1
				, 6.671 //H2
				, 5.74 - i * 1.483 //He3
				, 3.26 //He4
				, 2 - 0.261 * i //Li6
				, -2.22 //Li7
		;

*/

void PDBAmplitude::initialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
#pragma region Electronic Atomic form factor coefficients
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.057, 18.9525, 0.1195, 38.6269, //H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653, //He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337, //Li
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273, //Be
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314, //B		
		0.0893, 0.2465, 0.2563, 1.7100, 0.5770, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523, //C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715, 48.1431, //N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.643, 0.6990, 12.7105, 0.2039, 32.4726, //O
		0.1083, 0.2057, 0.3175, 1.3439, 0.6487, 4.2788, 0.5846, 11.3932, 0.1421, 28.7881, // F
		0.1269, 0.2200, 0.3535, 1.3779, 0.5582, 4.0203, 0.4674, 9.4934, 0.1460, 23.1278// Ne
		;
#pragma endregion
}

FACC PDBAmplitude::GetVoxelStepSize() {
	return voxelStep;
}

void PDBAmplitude::SetVoxelStepSize(FACC vS) {
	voxelStep = vS;
}

void PDBAmplitude::SetOutputSlices(bool bOutput, std::string outPath /*= ""*/) {
	this->pdb.bOutputSlices = bOutput;
	if(bOutput) {
		this->pdb.slicesBasePathSt = outPath;
	}
}

void PDBAmplitude::SetSolventOnlyCalculation(bool bOnlySol) {
	this->pdb.bOnlySolvent = bOnlySol;
}

void PDBAmplitude::SetFillHoles(bool bFillHole) {
	this->pdb.bFillHoles = bFillHole;

}

PDB_READER_ERRS PDBAmplitude::CalculateSolventAmp(FACC stepSize, FACC solED, FACC outerSolvED, FACC solRad, FACC atmRadDiffIn /*= 0.0*/) {
	std::cout << "Step size " << stepSize << ". Solvent ED " << solED << ". Outer solvent ED " << outerSolvED << "\n";
	this->SetVoxelStepSize(stepSize);
	// Should already have the relevant information
	if(stepSize == voxelStep && solRad == solventRad && 
		(
		(solventBoxDims.size() == solventBoxCOM.size() && solventBoxCOM.size() > 0)
		||
		(outerSolventBoxDims.size() == outerSolventBoxDims.size() && outerSolventBoxCOM.size() > 0)
		)
		)
	{
		this->outerSolventED = outerSolvED;
		this->solventED = solED;
		return PDB_OK;
	}
	this->outerSolventED = outerSolvED;
	this->solventED = solED;
	this->solventRad = solRad;
	PDB_READER_ERRS er = this->FindPDBRanges();
	if(er) {
		std::cout << "Error finding PDB ranges. Error code: " << er;
		return er;
	}

	try
	{
		auto beg = std::chrono::high_resolution_clock::now();
		CalculateSolventSpace();
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Solvation space calculation time: " << GetChronoTimeString(beg, end) << "\n";
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
	return er;
}

PDB_READER_ERRS PDBAmplitude::FindPDBRanges() {
	if(this->pdb.x.size() < 1) {
		gridStatus = AMP_UNINITIALIZED;

		this->status = NO_ATOMS_IN_FILE;
		throw backend_exception(ERROR_INVALIDPARAMTREE, "There are no atoms in the PDB file.");
	}

	FACC dist = std::max(solventRad, solvationThickness) + 2.0 * voxelStep;

	this->xMin = this->pdb.x[0] - (*pdb.rad)[this->pdb.atmInd[0]];
	this->xMax = this->pdb.x[0] + (*pdb.rad)[this->pdb.atmInd[0]];
	this->yMin = this->pdb.y[0] - (*pdb.rad)[this->pdb.atmInd[0]];
	this->yMax = this->pdb.y[0] + (*pdb.rad)[this->pdb.atmInd[0]];
	this->zMin = this->pdb.z[0] - (*pdb.rad)[this->pdb.atmInd[0]];
	this->zMax = this->pdb.z[0] + (*pdb.rad)[this->pdb.atmInd[0]];

	for(unsigned int i = 1; i < pdb.x.size(); i++) {
		this->xMin = std::min(float(xMin), this->pdb.x[i] - (*pdb.rad)[this->pdb.atmInd[i]]);
		this->xMax = std::max(float(xMax), this->pdb.x[i] + (*pdb.rad)[this->pdb.atmInd[i]]);
		this->yMin = std::min(float(yMin), this->pdb.y[i] - (*pdb.rad)[this->pdb.atmInd[i]]);
		this->yMax = std::max(float(yMax), this->pdb.y[i] + (*pdb.rad)[this->pdb.atmInd[i]]);
		this->zMin = std::min(float(zMin), this->pdb.z[i] - (*pdb.rad)[this->pdb.atmInd[i]]);
		this->zMax = std::max(float(zMax), this->pdb.z[i] + (*pdb.rad)[this->pdb.atmInd[i]]);
	}
	this->xMin -= dist;
	this->xMax += dist;
	this->yMin -= dist;
	this->yMax += dist;
	this->zMin -= dist;
	this->zMax += dist;

	int factor;
	factor = int(2.0 + (this->xMax - this->xMin) / this->voxelStep);
	this->xMax = this->xMin + FACC(factor) * this->voxelStep;
	factor = int(2.0 + (this->yMax - this->yMin) / this->voxelStep);
	this->yMax = this->yMin + FACC(factor) * this->voxelStep;
	factor = int(2.0 + (this->zMax - this->zMin) / this->voxelStep);
	this->zMax = this->zMin + FACC(factor) * this->voxelStep;

	return PDB_OK;

}

PDB_READER_ERRS PDBAmplitude::AllocateSolventSpace() {
	solventBoxCOM.clear();
	solventBoxDims.clear();
	outerSolventBoxCOM.clear();
	outerSolventBoxDims.clear();
	return PDB_OK;
}

template <typename T, typename IND>
void Eigen2DArrayToBMPMatrix(const Eigen::Array<T, -1, -1>& arr,
	Eigen::Array<unsigned char, -1, -1, Eigen::RowMajor>& image,
	const std::map<IND, Eigen::Array3i>& colorMap)
{
	int height = arr.rows();
	int width = arr.cols();
	image.resize(height, 3 * width);
	image.setZero();

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
			{
			int type = arr(j, i);
			int x = (/*yres*/height - 1) - j;
			int y = i;
			auto f = colorMap.find(type);
			if (f == colorMap.end())
			{
				image(x, 3 * y + 2) = (unsigned char)(124);
				image(x, 3 * y + 1) = (unsigned char)(124);
				image(x, 3 * y + 0) = (unsigned char)(124);
				continue;
			}

			auto c = f->second;
			image(x, 3 * y + 2) = (unsigned char)(c(0));
			image(x, 3 * y + 1) = (unsigned char)(c(1));
			image(x, 3 * y + 0) = (unsigned char)(c(2));

		}
	}
}

void writeNewBMPFile(fs::path fName, unsigned char *img, int width, int height)
{
	int filesize = 54 + 3 * width*height;  //w is your image width, h is image height, both int

	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(width);
	bmpinfoheader[5] = (unsigned char)(width >> 8);
	bmpinfoheader[6] = (unsigned char)(width >> 16);
	bmpinfoheader[7] = (unsigned char)(width >> 24);
	bmpinfoheader[8] = (unsigned char)(height);
	bmpinfoheader[9] = (unsigned char)(height >> 8);
	bmpinfoheader[10] = (unsigned char)(height >> 16);
	bmpinfoheader[11] = (unsigned char)(height >> 24);

	FILE *f;
	f = fopen(fName.string().c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i < height; i++) {
		fwrite(img + (width*(height - i - 1) * 3), 3, width, f);
		fwrite(bmppad, 1, (4 - (width * 3) % 4) % 4, f);
	}
	fclose(f);

}

void writeBMPfile(fs::path fName, array3 &data, int slice, int width, int height) {
	unsigned char *img = NULL;
	int filesize = 54 + 3*width*height;  //w is your image width, h is image height, both int
	if( img )
		free( img );
	img = (unsigned char *)malloc(3*width*height);
	memset(img,0,sizeof(img));

	for(int i = 0; i < width; i++) {
		for(int j = 0; j < height; j++) {
			int col = data[slice][i][j];
			int x = i;
			int y = (/*yres*/height - 1) - j;
			unsigned char r,g,b;
			switch (col)
			{
			case 0:	// Initialized --> Blue
				r = 0;
				g = 0;
				b = 255;
				break;
			case 1:	// Atom --> Green
				r = 0;
				g = 255;
				b = 0;
				break;
			case 2: // Far solvent --> Red
				r = 255;
				g = 0;
				b = 0;
				break;
			case 3: // Outer solvent --> Light blue
				r = 135;
				g = 196;
				b = 250;
				break;
			case 4: // Holes --> Hot pink
				r = 255;
				g = 105;
				b = 180;
				break;
			case 5: // Marker --> Yellow
				r = 255;
				g = 255;
				b = 0;
				break;
			case 6: // Marker --> Gray
				r = 128;
				g = 128;
				b = 128;
				break;
			default:
				r = 124;
				g = 124;
				b = 124;
				break;
			}
			img[(x+y*width)*3+2] = (unsigned char)(r);
			img[(x+y*width)*3+1] = (unsigned char)(g);
			img[(x+y*width)*3+0] = (unsigned char)(b);


		} // for j
	} // for i

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
	unsigned char bmppad[3] = {0,0,0};

	bmpfileheader[ 2] = (unsigned char)(filesize    );
	bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
	bmpfileheader[ 4] = (unsigned char)(filesize>>16);
	bmpfileheader[ 5] = (unsigned char)(filesize>>24);

	bmpinfoheader[ 4] = (unsigned char)(       width    );
	bmpinfoheader[ 5] = (unsigned char)(       width>> 8);
	bmpinfoheader[ 6] = (unsigned char)(       width>>16);
	bmpinfoheader[ 7] = (unsigned char)(       width>>24);
	bmpinfoheader[ 8] = (unsigned char)(       height    );
	bmpinfoheader[ 9] = (unsigned char)(       height>> 8);
	bmpinfoheader[10] = (unsigned char)(       height>>16);
	bmpinfoheader[11] = (unsigned char)(       height>>24);

	FILE *f;
	f = fopen(fName.string().c_str(), "wb");
	fwrite(bmpfileheader,1,14,f);
	fwrite(bmpinfoheader,1,40,f);
	for(int i = 0; i < height; i++) {
		fwrite(img+(width*(height - i - 1) * 3),3,width,f);
		fwrite(bmppad,1,(4-(width*3)%4)%4,f);
	}
	fclose(f);
	free(img);
}


void PDBAmplitude::WriteEigenSlicesToFile(string filebase)
{
	return;

	boost::system::error_code er;
	std::cout << "Writing " << _solvent_space.dimensions()(0) << " files\n";
	std::stringstream ss;
	ss << voxelStep * 10.0;
	fs::path pt(filebase);
	pt = fs::absolute(pt);
	if (!fs::exists(pt)) {
		if (!fs::create_directories(pt, er)) {
			std::cout << "Error creating directory: " << pt.string() << "\n";
			std::cout << "Error code: " << er << "\n";
			return;
		}
	}

	Eigen::Array<unsigned char, -1, -1, Eigen::RowMajor> imageData;
	for (int i = 0; i < _solvent_space.dimensions()(0); i++) {
		//i = _solvent_space.dimensions()(0) / 2; // Write only the center slice
		std::stringstream fn;
		fn << filebase << "\\" << voxelStep * 10.0 << "a_" << i << ".abc";
		fs::path fnp(fn.str());
		if (!fs::exists(fnp.parent_path())) {
			if (!fs::create_directories(fnp.parent_path(), er)) {
				std::cout << "\nError code: " << er << "\n";
				continue;
			}
		}
		Eigen2DArrayToBMPMatrix(_solvent_space.SliceX(i).eval(), imageData, static_color_map);
		writeNewBMPFile(fnp.replace_extension("bmp"), imageData.data(), _solvent_space.dimensions()(1), _solvent_space.dimensions()(2));
		//i = _solvent_space.dimensions()(0) + 1; // Write only the center slice
	}
	std::cout << "Finished writing slices\n";

}

void PDBAmplitude::GetHeader(unsigned int depth, JsonWriter &writer)
{
	if (depth == 0)
	{
		writer.Key("Program revision");
		writer.String(BACKEND_VERSION);
	}

	writer.Key("Title");
	writer.String("PDB");

	writer.Key("Filename");
	writer.String(pdb.fn.c_str());

	writer.Key("Centered");
	writer.Bool(bCentered);

	writer.Key("Scale");
	writer.Double(scale);

	writer.Key("Position");
	writer.StartArray();
	writer.Double(tx);
	writer.Double(ty);
	writer.Double(tz);
	writer.EndArray();

	writer.Key("Rotation");
	writer.StartArray();
	writer.Double(ra);
	writer.Double(rb);
	writer.Double(rg);
	writer.EndArray();

	if (bUseGrid && grid) {
		writer.Key("N");
		writer.Double(grid->GetSize());

		writer.Key("qMax");
		writer.Double(grid->GetQMax());

		writer.Key("StepSize");
		writer.Double(grid->GetStepSize());
	}

	if (solventED != 0.0 || outerSolventED != 0.0) {

		writer.Key("onlySolvent");
		writer.Bool(pdb.bOnlySolvent);



		writer.Key("Solvent ED amplitude subtracted"); 
		writer.Double(solventED);


		writer.Key("c1");
		writer.Double(c1);

		writer.Key("Solvent probe radius");
		writer.Double(solventRad); 

		writer.Key("Solvation thickness"); 
		writer.Double(solvationThickness);

		writer.Key("Solvent ED step size");
		writer.Double(voxelStep);

		writer.Key("Outer Solvent ED"); 
		writer.Double(outerSolventED);


		writer.Key("Solvent radius type");
		switch (pdb.atmRadType) {
		case RAD_CALC:
			writer.String("Calculated using voxels");
			break;
		case RAD_EMP:
			writer.String("Empirical using voxels");
			break;
		case RAD_DUMMY_ATOMS_RADII:
			writer.String("Dummy atom radii using voxels");
			break;
		case RAD_DUMMY_ATOMS_ONLY:
			writer.String("Dummy atoms");
			break;
		case RAD_VDW:
			writer.String("van der Waals using voxels");
			break;
		case RAD_UNINITIALIZED:
		default:
			writer.String("Uninitialized");
			break;
		}

		writer.Key("Fill Holes");
		writer.Bool(pdb.bFillHoles);
	}

	writer.Key("Implicit");
	writer.StartObject();
	writer.Key("# Atoms treated as having implicit hydrogen");
	writer.Int(pdb.number_of_implicit_atoms);
	writer.Key("# Total atoms");
	writer.Int(pdb.atom.size());
	writer.Key("# implicit amino acids");
	writer.Int(pdb.number_of_implicit_amino_acids);
	writer.EndObject();

}

void PDBAmplitude::GetHeader(unsigned int depth, std::string &header) {
	std::string ampers;
	ampers.resize(depth+1, '#');
	ampers.append(" ");

	std::stringstream ss;

	if(depth == 0) {
		header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
	}

	header.append(ampers + "//////////////////////////////////////\n");

	ss << "PDB file: " << pdb.fn << "\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "The atoms were " << ( bCentered ? "" : "NOT ") << "centered to the center of mass" << std::endl;
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Position (" << tx << "," << ty << "," << tz << ")\n";
	header.append(ampers + ss.str());
	ss.str("");
	ss << "Rotation (" <<  ra << "," << rb << "," << rg << ")\n";
	header.append(ampers + ss.str());
	ss.str("");


	if(bUseGrid && grid) {
		ss << "N^3; N = " << grid->GetSize() << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "qMax = " << grid->GetQMax() << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Grid step size = " << grid->GetStepSize() << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

	if(solventED != 0.0 || outerSolventED != 0.0) {
		if(pdb.bOnlySolvent) {
			ss << "The amplitude calculated corresponds to the solvent (and outer solvent) only, not the atoms.\n";
			header.append(ampers + ss.str());
			ss.str("");
		}

		ss << "Solvent ED amplitude subtracted: " << solventED << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "c1: " << c1 << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Solvent probe radius: " << solventRad << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Solvation thickness: " << solvationThickness << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Solvent ED step size: " << voxelStep << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Outer Solvent ED: " << outerSolventED << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Solvent radius type: ";
		switch(pdb.atmRadType) {
		case RAD_CALC:
			ss << "Calculated using voxels";
			break;
		case RAD_EMP:
			ss << "Empirical using voxels";
			break;
		case RAD_DUMMY_ATOMS_RADII:
			ss << "Dummy atom radii using voxels";
			break;
		case RAD_DUMMY_ATOMS_ONLY:
			ss << "Dummy atoms";
			break;
		case RAD_VDW:
			ss << "van der Waals using voxels";
			break;
		case RAD_UNINITIALIZED:
		default:
			ss << "Uninitialized";			
			break;
		}
		ss << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Fill holes: " << (pdb.bFillHoles ? "true" : "false") << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

	ss << pdb.number_of_implicit_atoms << " of " << pdb.atom.size() << " atoms are treated as having implicit hydrogen atoms (" << pdb.number_of_implicit_amino_acids << " amino acids).\n";
	header.append(ampers + ss.str());
	ss.str("");
	

	header.append(ampers + "Scale: " + boost::lexical_cast<std::string>(this->scale) + "\n");

}

ATOM_RADIUS_TYPE IntToAtom_Radius_Type(int ind) {
	switch (ind) {
	case 1: return RAD_VDW;
	case 2: return RAD_EMP;
	case 3: return RAD_CALC;
	case 4: return RAD_DUMMY_ATOMS_ONLY;
	case 5: return RAD_DUMMY_ATOMS_RADII;
	default: return RAD_UNINITIALIZED;
	}
}

void PDBAmplitude::PreCalculate(VectorXd& p, int nLayers) {
	auto local_previousParameters = previousParameters;

	Amplitude::PreCalculate(p, nLayers);
	if (status == PDB_OK)
		return;
	scale				= p(0);
	solventED			= p(1);
	c1					= p(2);
	voxelStep			= p(3);
	solventRad			= p(4);
	solvationThickness	= p(5);
	outerSolventED		= p(6);
	this->SetFillHoles(fabs(p(7) - 1.0) < 1.0e-6);
	this->SetSolventOnlyCalculation(fabs(p(8) - 1.0) < 1.0e-6);
	pdb.SetRadiusType( IntToAtom_Radius_Type( int(p(9) + 0.1) ) );

	atomLocs.resize(pdb.sortedX.size(), 3);

	atomLocs.col(0) = Eigen::Map<Eigen::ArrayXf>(pdb.sortedX.data(), pdb.sortedX.size());
	atomLocs.col(1) = Eigen::Map<Eigen::ArrayXf>(pdb.sortedY.data(), pdb.sortedX.size());
	atomLocs.col(2) = Eigen::Map<Eigen::ArrayXf>(pdb.sortedZ.data(), pdb.sortedX.size());

	int numUIons = 1;
	int prevIon = pdb.sortedIonInd[0];
	int prevInd = 0;
	std::vector<int> uniIonInds, numIonsPerInd;
	std::vector<float> uniIonRads;
	for (int i = 1; i < pdb.sortedIonInd.size(); i++)
	{
		if (prevIon != pdb.sortedIonInd[i])
		{
			uniIonInds.push_back(prevIon);
			uniIonRads.push_back(pdb.rad->at(pdb.sortedAtmInd[prevInd]));
			numIonsPerInd.push_back(i - prevInd);
			prevInd = i;
			prevIon = pdb.sortedIonInd[i];
			numUIons++;
		}
	}
	uniIonInds.push_back(prevIon);
	uniIonRads.push_back(pdb.rad->at(pdb.sortedAtmInd[prevInd]));
	numIonsPerInd.push_back(pdb.sortedIonInd.size() - prevInd);


	uniqueIonsIndices = Eigen::Map<Eigen::ArrayXi>(uniIonInds.data(), numUIons);
	numberOfIonsPerIndex = Eigen::Map<Eigen::ArrayXi>(numIonsPerInd.data(), numUIons);
	uniqueIonRads = Eigen::Map<Eigen::ArrayXf>(uniIonRads.data(), numUIons);

	bitwiseCalculationFlags = 0;

	if (solventED != 0.0 &&
		(this->pdb.atmRadType == RAD_CALC || this->pdb.atmRadType == RAD_EMP ||
		this->pdb.atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb.atmRadType == RAD_VDW)
		)
		bitwiseCalculationFlags |= CALC_VOXELIZED_SOLVENT;
	if (outerSolventED != 0.0 &&
		(this->pdb.atmRadType == RAD_CALC || this->pdb.atmRadType == RAD_EMP ||
		this->pdb.atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb.atmRadType == RAD_VDW ||
		pdb.atmRadType == RAD_DUMMY_ATOMS_ONLY)
		)
		bitwiseCalculationFlags |= CALC_VOXELIZED_OUTER_SOLVENT;

	if (!pdb.bOnlySolvent) bitwiseCalculationFlags |= CALC_ATOMIC_FORMFACTORS;
	if ((RAD_DUMMY_ATOMS_ONLY == pdb.GetRadiusType())
		&& solventED != 0.0
		) bitwiseCalculationFlags |= CALC_DUMMY_SOLVENT;

	if (pdb.haveAnomalousAtoms) bitwiseCalculationFlags |= CALC_ANOMALOUS;

	affCalculator.Initialize(bitwiseCalculationFlags, pdb.sortedIonInd.size(), numUIons, pdb.sortedCoeffs.data(), numberOfIonsPerIndex.data());

	if (bitwiseCalculationFlags & CALC_DUMMY_SOLVENT)
		affCalculator.SetSolventED(solventED, c1, uniIonRads.data() , pdb.bOnlySolvent);
	if (bitwiseCalculationFlags & CALC_ANOMALOUS)
	{
		std::vector<float2> anomfPrimesAsFloat2;

		size_t sz = pdb.sortedAnomfPrimes.size();
		anomfPrimesAsFloat2.resize(sz);
 		Eigen::Map<Eigen::ArrayXf>((float*)anomfPrimesAsFloat2.data(), 2 * sz) =
 			(Eigen::Map<Eigen::ArrayXf>((float*)pdb.sortedAnomfPrimes.data(), 2 * sz)).cast<float>();

		affCalculator.SetAnomalousFactors((float2*)(anomfPrimesAsFloat2.data()));
	}

	if (! 
		(bitwiseCalculationFlags & CALC_VOXELIZED_OUTER_SOLVENT || 
		 bitwiseCalculationFlags & CALC_VOXELIZED_SOLVENT)
		)
		return;
	//////////////////////////////////////////////////////////////////////////
	// For Roi
	//SetOutputSlices(true, "C:\\Roi\\Amplitudes\\1SVA Slices");
	//SetOutputSlices(true, "C:\\Delete\\Slices\\Fixed Solvation\\Test1");

	//////////////////////////////////////////////////////////////////////////

	// We have no change of relevant parameters for the voxels, so no
	// reason to recalculate their size and positions.
	if (
		local_previousParameters.size() > 0 &&
		local_previousParameters(3) == voxelStep &&
		local_previousParameters(4) == solventRad &&
		local_previousParameters(5) == solvationThickness &&
		local_previousParameters(7) == p(7) && // Fill holes
		local_previousParameters(9) == p(9) && // Radius type

		// The previous generation had voxels
			(
				local_previousParameters(6) != 0.0 || 
				(
					pdb.GetRadiusType() != RAD_UNINITIALIZED && pdb.GetRadiusType() != RAD_DUMMY_ATOMS_ONLY
				)
			)
		)
		return;

	PDB_READER_ERRS er = FindPDBRanges();
	if(er) {
		std::cout << "Error finding PDB ranges. Error code: " << er;
		return;
	}

	try
	{
		auto beg = std::chrono::high_resolution_clock::now();
		CalculateSolventSpace();
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Solvation space calculation time: " << GetChronoTimeString(beg, end) << "\n";
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
	return;

}


void PDBAmplitude::PrepareParametersForGPU(std::vector<float4>& solCOM, std::vector<int4>& solDims,
										   std::vector<float4>& outSolCOM, std::vector<int4>& outSolDims) {
	// Voxel Based Solvent (incl. outer solvent)
	if(false && voxelStep > 0.0 &&
		( (solventED != 0.0 && (pdb.GetRadiusType() == RAD_CALC || pdb.GetRadiusType() == RAD_EMP ||
		pdb.GetRadiusType() == RAD_DUMMY_ATOMS_RADII || pdb.GetRadiusType() == RAD_VDW) )
		||
		(outerSolventED != 0.0 && solventRad > 0.0)
		)
		){
			CalculateSolventAmp(voxelStep, solventED, outerSolventED, solventRad);
	}

	u64 solSize = solventBoxDims.size();
	u64 outSolSize = outerSolventBoxDims.size();
	std::vector<SolventBoxEntry> solEntries(solventBoxDims.size());

	for(u64 i = 0; i < solventBoxDims.size(); i++) {
		solEntries[i].loc = make_float4((solventBoxCOM[i])[0], (solventBoxCOM[i])[1], (solventBoxCOM[i])[2], 0.0f);
		solEntries[i].len = make_int4((solventBoxDims[i])[0], (solventBoxDims[i])[1], (solventBoxDims[i])[2], 0);
	}
	std::sort(solEntries.begin(), solEntries.end(), SortSolventBoxEntry);
	solCOM.resize(solventBoxDims.size());
	solDims.resize(solventBoxDims.size());
	for(u64 i = 0; i < solventBoxDims.size(); i++) {
		solCOM[i]	= solEntries[i].loc;
		solDims[i]	= solEntries[i].len;
	}

	solEntries.resize(outerSolventBoxDims.size());
	for(u64 i = 0; i < outerSolventBoxDims.size(); i++) {
		solEntries[i].loc = make_float4((outerSolventBoxCOM[i])[0], (outerSolventBoxCOM[i])[1], (outerSolventBoxCOM[i])[2], 0.0f);
		solEntries[i].len = make_int4((outerSolventBoxDims[i])[0], (outerSolventBoxDims[i])[1], (outerSolventBoxDims[i])[2], 0);
	}
	std::sort(solEntries.begin(), solEntries.end(), SortSolventBoxEntry);
	outSolCOM.resize(outerSolventBoxDims.size());
	outSolDims.resize(outerSolventBoxDims.size());
	for(u64 i = 0; i < outerSolventBoxDims.size(); i++) {
		outSolCOM[i]	= solEntries[i].loc;
		outSolDims[i]	= solEntries[i].len;
	}
	// Ensure there's at least one element so that we can pass on the address
	if(solventBoxDims.size() == 0) {
		solCOM.resize(1);
		solDims.resize(1);
	}
	if(outerSolventBoxDims.size() == 0) {
		outSolCOM.resize(1);
		outSolDims.resize(1);
	}

	if(RAD_UNINITIALIZED == pdb.GetRadiusType()) {
		// This will load the default (vdw) as a way to make the rad pointer not problematic
		pdb.SetRadiusType(RAD_UNINITIALIZED);
	}
}

std::string PDBAmplitude::Hash() const
{
	std::string str = BACKEND_VERSION "PDB: ";
	str += std::to_string(voxelStep) + std::to_string(solventED) + std::to_string(c1)
		+ std::to_string(solventRad) + std::to_string(solvationThickness) + std::to_string(outerSolventED);

	str += pdb_hash;

	return md5(str);
}

bool PDBAmplitude::SetModel(Workspace& workspace) {
	if (!g_useGPUAndAvailable)
		return false;
	
	if (!gpuSetPDB)
		gpuSetPDB = (GPUDirectSetPDB_t)GPUDirect_SetPDBDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_SetPDBDLL");
	if(!gpuSetPDB)
		return false;
				
	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return gpuSetPDB(workspace, (const float4 *)&pdb.atomLocs[0], &pdb.sortedCoeffIonInd[0],
					 pdb.atomLocs.size(), &pdb.sortedCoeffs[0], 
					 &pdb.atomsPerIon[0],
					 pdb.sortedCoeffs.size() / 9);
}

bool PDBAmplitude::SetParameters(Workspace& workspace, const double *params, unsigned int numParams) {
	// TODO: Later (solvents and stuff)
	if (!g_useGPUAndAvailable)
		return false;

	return true;
}

bool PDBAmplitude::ComputeOrientation(Workspace& workspace, float3 rotation) {
	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuPDBAmplitude)
		gpuPDBAmplitude = (GPUDirectPDBAmplitude_t)GPUDirect_PDBAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_PDBAmplitudeDLL");
	if(!gpuPDBAmplitude)
		return false;
	
	return gpuPDBAmplitude(workspace, rotation);
}

std::string PDBAmplitude::GetName() const {
	return "PDB: " + pdb.fn;
}


bool PDBAmplitude::SetModel( GridWorkspace& workspace ) {
	if (!g_useGPUAndAvailable)
		return false;

	workspace.bSolOnly = pdb.bOnlySolvent;
	workspace.scale = scale;

	std::vector<float4> solCOM, outSolCOM;
	std::vector<int4> solDims, outSolDims;

	PrepareParametersForGPU(solCOM, solDims, outSolCOM, outSolDims);

	workspace.solventED = float(solventED);
	workspace.solventType = int(pdb.GetRadiusType());
	workspace.atomsPerIon = pdb.atomsPerIon.data();
	workspace.numUniqueIons = pdb.atomsPerIon.size();
	
	int comb = (workspace.bSolOnly ? 0 : CALC_ATOMIC_FORMFACTORS);
	comb |= ((solventED != 0.0 && int(pdb.GetRadiusType()) == 4) ? CALC_DUMMY_SOLVENT : 0);
	workspace.kernelComb = comb;

	std::vector<float> affs;
	affs.resize(workspace.qLayers * workspace.numUniqueIons);
	affCalculator.GetQMajorAFFMatrix(affs.data(), workspace.qLayers, workspace.stepSize);

	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return GPUHybrid_SetPDB(workspace, *reinterpret_cast<std::vector<float4>*>(&pdb.atomLocs),
		affs,
		/*pdb.sortedCoeffIonInd,
		pdb.sortedCoeffs, pdb.atomsPerIon,
		int(pdb.GetRadiusType()), pdb.sortedAtmInd, *(pdb.rad),
		solventED, */
		solventRad,
		solCOM.data(), solDims.data(), solDims.size(), voxelStep, outSolCOM.data(),
		outSolDims.data(), outSolDims.size(), outerSolventED );

	return true;
}

bool PDBAmplitude::CalculateGridGPU( GridWorkspace& workspace ) {

	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuPDBHybridAmplitude)
		gpuPDBHybridAmplitude = (GPUHybridPDBAmplitude_t)GPUHybrid_PDBAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_PDBAmplitudeDLL");
	if(!gpuPDBHybridAmplitude)
		return false;

	return gpuPDBHybridAmplitude(workspace);
}

bool PDBAmplitude::SavePDBFile(std::ostream &output) {
	std::vector<std::string> lines;
	std::vector<Eigen::Vector3f> locs;

	bool res = AssemblePDBFile(lines, locs);

	if(lines.size() != locs.size()) {
		std::cout << "Mismatched sizes" << std::endl;
		return false;
	}

	for(int i = 0; i < locs.size(); i++) {
		std::string line = lines[i];
		std::string xst, yst, zst;
		char grr = line[54];
		xst.resize(24);
		yst.resize(24);
		zst.resize(24);

		sprintf(&xst[0], "%8f", locs[i].x() * 10. );
		sprintf(&yst[0], "%8f", locs[i].y() * 10. );
		sprintf(&zst[0], "%8f", locs[i].z() * 10. );

		sprintf(&line[30], "%s", xst.substr(0,8).c_str());
		sprintf(&line[38], "%s", yst.substr(0,8).c_str());
		sprintf(&line[46], "%s", zst.substr(0,8).c_str());

		line[54] = grr;

		output << line << std::endl;
	}

	return true;
}

bool PDBAmplitude::AssemblePDBFile( std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs ) {
	int nAtoms = pdb.x.size();
	lines.resize(nAtoms);
	locs.resize(nAtoms);
	Eigen::Matrix3f rt = EulerD<float>(ra, rb, rg);
	Eigen::Vector3f tr(tx, ty, tz);
	std::string posPlaceHolder, occTemp;
	posPlaceHolder.resize(24, 'p');
	posPlaceHolder[0] = '-';
	posPlaceHolder[23] = '-';
	occTemp.resize(30);

	for(int pp = 0; pp < nAtoms; pp++)
	{
		lines[pp].resize(80);
		sprintf(&occTemp[0], "%6f", pdb.occupancy[pp]);
		sprintf(&occTemp[6], "%6f", pdb.BFactor[pp]);
		sprintf(&(lines[pp][0]),
			"ATOM  %5s %4s "
			//			 ^     ^   ^
			//			 0     6  11
			"%3s %c"
			//			 ^   ^
			//			16  21
			"%4s    "
			//			 ^     ^
			//			22    29
			"%s" // location
			//			 ^  ^  ^
			//			30 38 46
			"%s"	// occupancy tempFactor
			//			 ^  ^
			//			54 60
			"      "
			"%s%s",
			//			 ^ ^
			//			72 76
			pdb.pdbAtomSerNo[pp].substr(0, 5).c_str(), pdb.pdbAtomName[pp].substr(0, 4).c_str(),
			pdb.pdbResName[pp].substr(0, 3).c_str(), pdb.pdbChain[pp],
			pdb.pdbResNo[pp].substr(0, 4).c_str(),
			posPlaceHolder.substr(0, 24).c_str(),//pdb.x[pp]*10., pdb.y[pp]*10., pdb.z[pp]*10.,
			occTemp.substr(0,12).c_str(),
			pdb.pdbSegID[pp].substr(0, 4).c_str(), pdb.atom[pp].substr(0, 4).c_str());
		Eigen::Vector3f loc(pdb.x[pp], pdb.y[pp], pdb.z[pp]);
		locs[pp] = rt * loc + tr;
	} // for pp

	return true;
}

bool PDBAmplitude::ImplementedHybridGPU() {
	return true;
}

void PDBAmplitude::SetPDBHash()
{
	pdb_hash = "";
	pdb_hash += pdb.fn;
	pdb_hash += pdb.anomalousfn;

	//sortedX, sortedY, sortedZ;
	for (const auto& c : pdb.sortedX)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb.sortedY)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb.sortedZ)
		pdb_hash += std::to_string(c);
	//sortedAtmInd, sortedIonInd
	for (const auto& c : pdb.sortedAtmInd)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb.sortedIonInd)
		pdb_hash += std::to_string(c);

	// anomIndex, sortedAnomfPrimes
	for (const auto& c : pdb.anomIndex)
		pdb_hash += std::to_string(c.first) + std::to_string(c.second.first) + std::to_string(c.second.second);
	for (const auto& c : pdb.sortedAnomfPrimes)
		pdb_hash += std::to_string(c.real()) + std::to_string(c.imag());

	pdb_hash = md5(pdb_hash);

}

#pragma endregion	// CPDB Reader class
