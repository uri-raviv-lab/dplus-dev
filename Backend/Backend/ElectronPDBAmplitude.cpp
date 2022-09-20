#define AMP_EXPORTER

#include "../backend_version.h"

#include "ElectronPDBAmplitude.h"
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

// THIS REGION WILL BE DELETED ONCE ElectronPDBAmplitude INHERITS PDBAmplitude !!!

struct electronSolventBoxEntry {
	int4 len;
	float4 loc;
};
bool electronSortSolventBoxEntry(const electronSolventBoxEntry& a, const electronSolventBoxEntry& b) {
	if (a.len.x < b.len.x)
		return true;
	if (a.len.x > b.len.x)
		return false;
	if (a.len.y < b.len.y)
		return true;
	if (a.len.y > b.len.y)
		return false;
	if (a.len.z < b.len.z)
		return true;
	if (a.len.z > b.len.z)
		return false;
	//From here on, it's just to avoid an assertion failure
	if (a.loc.x < b.loc.x)
		return true;
	if (a.loc.x > b.loc.x)
		return false;
	if (a.loc.y < b.loc.y)
		return true;
	if (a.loc.y > b.loc.y)
		return false;
	if (a.loc.z < b.loc.z)
		return true;
	if (a.loc.z > b.loc.z)
		return false;
	return true;
}

typedef int (*electronGPUCalculatePDB_t)(u64 voxels, unsigned short dimx,
	double qmax, unsigned short sections, double* outData,
	double* loc, u8* ionInd,
	int numAtoms, double* coeffs, int numCoeffs, bool bSolOnly,
	u8* atmInd, float* rad, double solvED, u8 solventType,	// FOR DUMMY ATOM SOLVENT
	double* solCOM, u64* solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	double* outSolCOM, u64* outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	progressFunc progfunc, void* progargs, double progmin, double progmax, int* pStop);

electronGPUCalculatePDB_t electronGpuCalcPDB = NULL;

typedef double ELECTRON_IN_JF_TYPE;
typedef int (*electronGPUCalculatePDBJ_t)(u64 voxels, int thDivs, int phDivs, ELECTRON_IN_JF_TYPE stepSize, double* outData, float* locX, float* locY, float* locZ,
	u8* ionInd, int numAtoms, float* coeffs, int numCoeffs, bool bSolOnly,
	u8* atmInd, float* atmRad, ELECTRON_IN_JF_TYPE solvED, u8 solventType,// FOR DUMMY ATOM SOLVENT
	float4* solCOM, int4* solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	float4* outSolCOM, int4* outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop);
electronGPUCalculatePDBJ_t electronGpuCalcPDBJ = NULL;

typedef bool (*electronGPUDirectSetPDB_t)(Workspace& work, const float4* atomLocs,
	const unsigned char* ionInd,
	size_t numAtoms, const float* coeffs,
	int* atomsPerIon,
	size_t numCoeffs);
electronGPUDirectSetPDB_t electronGpuSetPDB = NULL;

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

typedef bool (*electronGPUDirectPDBAmplitude_t)(Workspace& work, float3 rotation);
electronGPUDirectPDBAmplitude_t electronGpuPDBAmplitude = NULL;

typedef bool (*electronGPUHybridPDBAmplitude_t)(GridWorkspace& work);
electronGPUHybridPDBAmplitude_t electronGpuPDBHybridAmplitude = NULL;

#pragma endregion

#pragma region CPDB Reader class

electronPDBAmplitude::~electronPDBAmplitude() {
}

electronPDBAmplitude::electronPDBAmplitude(string filename, bool bCenter, string anomalousFilename, int model /*= 0*/) : Amplitude() {
	electronInitialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb = ElectronPDBReader::electronPDBReaderOb<float>(filename, bCenter, model, anomalousFilename);

	bCentered = bCenter;

	SetPDBHash();
}

electronPDBAmplitude::electronPDBAmplitude(const char *buffer, size_t buffSize, const char *filenm, size_t fnSize, bool bCenter, const char *anomalousFilename, size_t anomBuffSize, int model) : Amplitude() {
	electronInitialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb.fn.assign(filenm, fnSize);

	try
	{
		if (anomalousFilename && anomBuffSize > 0)
			status = pdb.readAnomalousbuffer(anomalousFilename, anomBuffSize);

		if (PDB_OK == status)
			status = pdb.readPDBbuffer(buffer, buffSize, bCenter, model);
	}
	catch (ElectronPDBReader::pdbReader_exception &e)
	{
		status = PDB_READER_ERRS::ERROR_IN_PDB_FILE;
		throw backend_exception(e.GetErrorCode(), e.GetErrorMessage().c_str());
	}
	bCentered = bCenter;

	SetPDBHash();
}

electronPDBAmplitude::electronPDBAmplitude() {
	gridStatus = AMP_UNINITIALIZED;

	status = UNINITIALIZED;
	electronInitialize();
}

std::complex<FACC> electronPDBAmplitude::calcAmplitude(FACC qx, FACC qy, FACC qz) {
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
			electronAffCalculator.GetAllAFFs((float2*)(affs.data()), q);

			std::complex<float> tmpA = (affs * phases.sin().cast<std::complex<float>>()).sum();
			std::complex<float> tmpB = (affs * phases.cos().cast<std::complex<float>>()).sum();
			resI += tmpA.imag() + tmpB.imag();
			resR += tmpA.real() + tmpB.real();
		}
		else
		{
			Eigen::Array<float, Eigen::Dynamic, 1> affs(xSz);
			electronAffCalculator.GetAllAFFs(affs.data(), q);

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

void electronPDBAmplitude::calculateGrid(FACC qmax, int sections, progressFunc progFunc, void *progArgs, double progMin, double progMax, int *pStop) {
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

	// PDBJacobianGridAmplitudeCalculation DOES NOT USE ELECTRON affCalculator FUNCTIONS.
	// PASSING NON-ELECTRON TO AVOID CODE DUPLICATION (CHANGE THIS TO INHERITANCE LATER)
	int gpuRes = PDBJacobianGridAmplitudeCalculation
		(
		voxels, grid->GetDimY(1) - 1, grid->GetDimZ(1, 1), grid->GetStepSize(),
		grid->GetDataPointer(), atomLocations, electronAffCalculator,
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

std::complex<FACC> electronPDBAmplitude::calcAmplitude(int indqx, int indqy, int indqz) {
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
			aff = electronAtomicFF(q/(10.0), pdb.ionInd[i]);

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

FACC electronPDBAmplitude::electronAtomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998); 
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

	for(int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2*i)) * exp(-atmFFcoefs(elem, (2*i) + 1) * sqq);
	
	return res;
}

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

void electronPDBAmplitude::electronInitialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(ELECTRON_NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
	// I'm almost certain we can delete this part of the code but I changed just in case. Go to PDBReaderLib.cpp
#pragma region Atomic form factor coefficients - Peng
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.0573, 18.9525, 0.1195, 38.6269,	// H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653,	// He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337,	// Li
		0.00460, 0.0358, 0.0165, 0.239, 0.0435, 0.879, 0.0649, 2.64, 0.0270, 7.09, 			//Li+1
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273,	// Be
		0.00340, 0.0267, 0.0103, 0.162, 0.0233, 0.531, 0.0325, 1.48, 0.0120, 3.88, 			//Be+2
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314,	// B
		0.0893, 0.2465, 0.2563, 1.7100, 0.7570, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523,	// C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715, 48.1431,	// N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.6943, 0.6990, 12.7105, 0.2039, 32.4726,	// O
		0.205, 0.397, 0.628, 0.264, 1.17, 8.80, 1.03, 27.1, 0.290, 91.8, 					//O-1
		0.1083, 0.2057, 0.3175, 1.3439, 0.6487, 4.2788, 0.5846, 11.3932, 0.1421, 28.7881,	// F
		0.134, 0.228, 0.391, 1.47, 0.814, 4.68, 0.928, 13.2, 0.347, 36.0, 					//F-1
		0.1269, 0.2200, 0.3535, 1.3779, 0.5582, 4.0203, 0.4674, 9.4934, 0.1460, 23.1278,	// Ne
		0.2142, 0.334, 0.6853, 2.3446, 0.7692, 10.0830, 1.6589, 48.3037, 1.4482, 137.2700,	// Na
		0.0256, 0.0397, 0.0919, 0.287, 0.297, 1.18, 0.514, 3.75, 0.199, 10.8,				// Na+1
		0.2314, 0.3278, 0.6866, 2.2720, 0.9677, 10.9241, 2.1182, 32.2898, 1.1339, 101.9748,	// Mg
		0.0210, 0.0331, 0.0672, 0.222, 0.198, 0.838, 0.368, 2.48, 0.174, 6.75,				// Mg+2
		0.2390, 0.3138, 0.6573, 2.1063, 1.2011, 10.4163, 2.5586, 34.4552, 1.2312, 98.5344,	// Al
		0.0192, 0.0306, 0.0579, 0.198, 0.163, 0.713, 0.284, 2.04, 0.114, 5.25,				// Al+3
		0.2519, 0.3075, 0.6372, 2.0174, 1.3795, 9.6746, 2.5082, 29.3744, 1.0500, 80.4732,	// Si_v
		0.192, 0.359, 0.289, 1.96, 0.100, 9.34, -0.0728, 11.1, 0.00120, 13.4,				// Si+4
		0.2548, 0.2908, 0.6106, 1.8740, 1.4541, 8.5176, 2.3204, 24.3434, 0.8477, 63.2996,	// P
		0.2497, 0.2681, 0.5628, 1.6711, 1.3899, 7.0267, 2.1865, 19.5377, 0.7715, 50.3888,	// S
		0.2443, 0.2468, 0.5397, 1.5242, 1.3919, 6.1537, 2.0197, 16.6687, 0.6621, 42.3086,	// Cl
		0.265, 0.252, 0.596, 1.56, 1.60, 6.21, 2.69, 17.8, 1.23, 47.8,						// Cl-1
		0.2385, 0.2289, 0.5017, 1.3694, 1.3128, 5.2561, 1.8899, 14.0928, 0.6079, 35.5361,	//Ar
		0.4115, 0.3703, 1.4031, 3.3874, 2.2784, 13.1029, 2.6742, 68.9592, 2.2162, 194.4329,	//K
		0.199, 0.192, 0.396, 1.10, 0.928, 3.91, 1.45, 9.75, 0.450, 23.4,					//K+1
		0.4054, 0.3499, 1.880, 3.0991, 2.1602, 11.9608, 3.7532, 53.9353, 2.2063, 142.3892,	//Ca
		0.164, 0.157, 0.327, 0.894, 0.743, 3.15, 1.16, 7.67, 0.307, 17.7,					//Ca+2
		0.3787, 0.3133, 1.2181, 2.5856, 2.0594, 9.5813, 3.2618, 41.7688, 2.3870, 116.7282,	//Sc
		0.163, 0.157, 0.307, 0.899, 0.716, 3.06, 0.880, 7.05, 0.139, 16.1,					//Sc+3
		0.3825, 0.3040, 1.2598, 2.4863, 2.0008, 9.2783, 3.0617, 39.0751, 2.0694, 109.4583,	//Ti
		0.399, 0.376, 1.04, 2.74, 1.21, 8.10, -0.0797, 14.2, 0.352, 23.2,					//Ti+2
		0.364, 0.364, 0.919, 2.67, 1.35, 8.18, -0.933, 11.8, 0.589, 14.9,					//Ti+3
		0.116, 0.108, 0.256, 0.655, 0.565, 2.38, 0.772, 5.51, 0.32, 12.3,					//Ti+4
		0.3876, 0.2967, 1.2750, 2.3780, 1.9109, 8.7981, 2.8314, 35.9528, 1.8979, 101.7201,	//V
		0.317, 0.269, 0.939, 2.09, 1.49, 7.22, -1.31, 15.2, 1.47, 17.6,						//V+2
		0.341, 0.321, 0.805, 2.23, 0.942, 5.99, 0.0783, 13.4, 0.156, 16.9,					//V+3
		0.0367, 0.0330, 0.124, 0.222, 0.244, 0.824, 0.723, 2.8, 0.435, 6.70,				//V+5
		0.4046, 0.2986, 1.3696, 2.3958, 1.8941, 9.1406, 2.0800, 37.4701, 1.2196, 113.7121,	//Cr
		0.237, 0.177, 0.634, 1.35, 1.23, 4.30, 0.713, 12.2, 0.0859, 39.0,					//Cr+2
		0.393, 0.359, 1.05, 2.57, 1.62, 8.68, -1.15, 11.0, 0.407, 15.8,						//Cr+3
		0.3796, 0.2699, 1.2094, 2.0455, 1.7815, 7.4726, 2.5420, 31.0604, 1.5937, 91.5622,	//Mn
		0.0576, 0.0398, 0.210, 0.284, 0.604, 1.29, 1.32, 4.23, 0.659, 14.5,					//Mn+2
		0.116, 0.0117, 0.523, 0.876, 0.881, 3.06, 0.589, 6.44, 0.214, 14.3,					//Mn+3
		0.381, 0.354, 1.83, 2.72, -1.33, 3.47, 0.995, 5.47, 0.0618, 16.1,					//Mn+4
		0.3946, 0.2717, 1.2725, 2.0443, 1.7031, 7.6007, 2.3140, 29.9714, 1.4795, 86.2265,	//Fe
		0.307, 0.230, 0.838, 1.62, 1.11, 4.87, 0.280, 10.7, 0.277, 19.2,					//Fe+2
		0.198, 0.154, 0.384, 0.893, 0.889, 2.62, 0.709, 6.65, 0.117, 18.0,				 	//Fe+3
		0.4118, 0.2742, 1.3161, 2.0372, 1.6493, 7.7205, 2.1930, 29.9680, 1.2830, 84.9383,	//Co
		0.213, 0.148, 0.488, 0.939, 0.998, 2.78, 0.828, 7.31, 0.230, 20.7,					//Co+2
		0.331, 0.267, 0.487, 1.41, 0.729, 2.89, 0.608, 6.45, 0.131, 15.8,					//Co+3
		0.3860, 0.2478, 1.1765, 1.7660, 1.5451, 6.3107, 2.0730, 25.2204, 1.3814, 74.3146,	//Ni
		0.338, 0.237, 0.982, 1.67, 1.32, 5.73, -3.56, 11.4, 3.62, 12.1,						//Ni+2
		0.347, 0.260, 0.877, 1.71, 0.790, 4.75, 0.0538, 7.51, 0.192, 13.0,					//Ni+3
		0.4314, 0.2694, 1.3208, 1.9223, 1.5236, 7.3474, 1.4671, 28.9892, 0.8562, 90.6246,	//Cu
		0.312, 0.201, 0.812, 1.31, 1.11, 3.80, 0.794, 10.5, 0.257, 28.2,					//Cu+1
		0.224, 0.145, 0.544, 0.933, 0.970, 2.69, 0.727, 7.11, 0.1882, 19.4,					//Cu+2
		0.4288, 0.2593, 1.2646, 1.7998, 1.4472, 6.7500, 1.8294, 25.5860, 1.0934, 73.5284,	//Zn
		0.252, 0.161, 0.600, 1.01, 0.917, 2.76, 0.663, 7.08, 0.161, 19.0,					//Zn+2
		0.4818, 0.2825, 1.4032, 1.9785, 1.6561, 8.7546, 2.4605, 32.5238, 1.1054, 98.5523,	//Ga
		0.391, 0.264, 0.947, 1.65, 0.690, 4.82, 0.0709, 10.7, 0.0653, 15.2,					//Ga+3
		0.4655, 0.2647, 1.3014, 1.7926, 1.6088, 7.6071, 2.6998, 26.5541, 1.3003, 77.5238,	//Ge
		0.346, 0.232, 0.830, 1.45, 0.599, 4.08, 0.0949, 13.2, -0.0217, 29.5,				//Ge+4
		0.4517, 0.2493, 1.2229, 1.6436, 1.5852, 6.8154, 2.7958, 22.3681, 1.2638, 62.0390,	//As
		0.4477, 0.2405, 1.1678, 1.5442, 1.5843, 6.3231, 2.8087, 19.4610, 1.1956, 52.0233,	//Se
		0.4798, 0.2504, 1.1948, 1.5963, 1.8695, 6.9653, 2.6953, 19.8492, 0.8203, 50.3233,	//Br
		0.125, 0.0530, 0.563, 0.469, 1.43, 2.15, 3.25, 11.1, 3.22, 38.9,					//Br-1
		0.4546, 0.2309, 1.0993, 1.4279, 1.76966, 5.9449, 2.7068, 16.6752, 0.8672, 42.2243,	//Kr
		1.0160, 0.4853, 2.8528, 5.0925, 3.5466, 25.7851, -7.7804, 130.4515, 12.1148, 138.6775,//Rb
		0.368, 0.187, 0.884, 1.12, 1.12, 3.98, 2.26, 10.9, 0.881, 26.6,						//Rb+1
		0.6703, 0.3190, 1.4926, 2.2287, 3.3368, 10.3504, 4.4600, 52.3291, 3.1501, 151.2216,	//Sr
		0.346, 0.176, 0.804, 1.04, 0.988, 3.59, 1.89, 9.32, 0.609, 21.4,					//Sr+2
		0.6894, 0.3189, 1.5474, 2.2904, 3.2450, 10.0062, 4.2126, 44.0771, 2.9764, 125.0120,	//Y
		0.465, 0.240, 0.923, 1.43, 2.41, 6.45, -2.31, 9.97, 2.48, 12.2,						//Y+3
		0.6719, 0.3036, 1.4684, 2.1249, 3.1668, 8.9236, 3.9957, 36.8458, 2.8920, 108.2049,	//Zr
		0.34, 0.113, 0.642, 0.736, 0.747, 2.54, 1.47, 6.72, 0.377, 14.7,					//Zr+4
		0.6123, 0.2709, 1.2677, 1.7683, 3.0348, 7.2489, 3.3841, 27.9465, 2.3683, 98.5624,	//Nb
		0.377, 0.184, 0.749, 1.02, 1.29, 3.80, 1.61, 9.44, 0.481, 25.7,						//Nb+3
		0.0828, 0.0369, 0.271, 0.261, 0.654, 0.957, 1.24, 3.94, 0.829, 9.44,				//Nb+5
		0.6773, 0.2920, 1.4798, 2.0606, 3.1788, 8.1129, 3.0824, 30.5336, 1.8384, 100.0658,	//Mo
		0.401, 0.191, 0.756, 1.06, 1.38, 3.84, 1.58, 9.38, 0.497, 24.6,						//Mo+3
		0.479, 0.241, 0.846, 1.46, 15.6, 6.79, -15.2, 7.13, 1.60, 10.4,						//Mo+5
		0.203, 0.0971, 0.567, 0.647, 0.646, 2.28, 1.16, 5.61, 0.171, 12.4,					//Mo+6
		0.7082, 0.2976, 1.6392, 2.2106, 3.1993, 8.5246, 3.4327, 33.1456, 1.8711, 96.6377,	//Tc
		0.6735, 0.2773, 1.4934, 1.9716, 3.0966, 7.3249, 2.7254, 26.6891, 1.5597, 90.5581,	//Ru
		0.428, 0.191, 0.773, 1.09, 1.55, 3.82, 1.46, 9.08, 0.486, 21.7,						//Ru+3
		0.2882, 0.125, 0.653, 0.753, 1.14, 2.85, 1.53, 7.01, 0.418, 17.5,					//Ru+4
		0.6413, 0.2580, 1.3690, 1.7721, 2.9854, 6.3854, 2.6952, 23.2549, 1.5433, 58.1517,	//Rh
		0.352, 0.151, 0.723, 0.878, 1.50, 3.28, 1.63, 8.16, 0.499, 20.7,					//Rh+3
		0.397, 0.177, 0.725, 1.01, 1.51, 3.62, 1.19, 8.56, 0.251, 18.9,						//Rh+4
		0.5904, 0.2324, 1.1775, 1.5019, 2.6519, 5.1591, 2.2875, 15.5428, 0.8689, 46.8213,	//Pd
		0.935, 0.393, 3.11, 4.06, 24.6, 43.1, -43.6, 54.0, 21.2, 69.8,						//Pd+2
		0.348, 0.151, 0.640, 0.832, 1.22, 2.85, 1.45, 6.59, 0.427, 15.6,					//Pd+4
		0.6377, 0.2466, 1.3790, 1.6974, 2.8294, 5.7656, 2.3631, 20.0943, 1.4553, 76.7372,	//Ag
		0.503, 0.199, 0.940, 1.19, 2.17, 4.05, 1.99, 11.3, 0.726, 32.4,						//Ag+1
		0.431, 0.175, 0.756, 0.979, 1.72, 3.30, 1.78, 8.24, 0.526, 21.4,					//Ag+2
		0.6364, 0.2407, 1.4247, 1.6823, 2.7802, 5.6588, 2.5973, 20.7219, 1.7886, 69.1109,	//Cd
		0.425, 0.168, 0.745, 0.944, 1.73, 3.14, 1.74, 7.84, 0.487, 20.4,					//Cd+2
		0.6768, 0.2522, 1.6589, 1.8545, 2.7740, 6.2936, 3.1835, 25.1457, 2.1326, 84.5448,	//In
		0.417, 0.164, 0.755, 0.960, 1.59, 3.08, 1.36, 7.03, 0.451, 16.1,					//In+3
		0.7224, 0.2651, 1.9610, 2.0604, 2.7161, 7.3011, 3.5603, 27.5493, 1.8972, 81.3349,	//Sn
		0.797, 0.317, 2.13, 2.51, 2.15, 9.04, -1.64, 24.2, 2.72, 26.4,						//Sn+2
		0.261, 0.0957, 0.642, 0.625, 1.53, 2.51, 1.36, 6.31, 0.177, 15.9,					//Sn+4
		0.7106, 0.2562, 1.9247, 1.9646, 2.6149, 6.8852, 3.8322, 24.7648, 1.8899, 68.9168,	//Sb
		0.552, 0.212, 1.14, 1.42, 1.87, 4.21, 1.36, 12.5, 0.414, 29.0,						//Sb+3
		0.377, 0.151, 0.588, 0.812, 1.22, 2.40, 1.18, 5.27, 0.244, 11.9,					//Sb+5
		0.6947, 0.2459, 1.8690, 1.8542, 2.5356, 6.4411, 4.0013, 22.1730, 1.8955, 59.2206,	//Te
		0.7047, 0.2455, 1.9484, 1.8638, 2.5940, 6.7639, 4.1526, 21.8007, 1.5057, 56.4395,	//I
		0.901, 0.312, 2.80, 2.59, 5.61, 4.1, -8.69, 34.4, 12.6, 39.5,						//I-1
		0.6737, 0.2305, 1.7908, 1.6890, 2.4129, 5.8218, 4.5100, 18.3928, 1.7058, 47.2496,	//Xe
		1.2704, 0.4356, 3.8018, 4.2058, 5.6618, 23.4342, 0.9205, 136.7783, 4.8105, 171.7561,//Cs
		0.587, 0.200, 1.40, 1.38, 1.87, 4.12, 3.48, 13.0, 1.67, 31.8,						//Cs+1
		0.9049, 0.3066, 2.6076, 2.4363, 4.8498, 12.1821, 5.1603, 54.6135, 4.7388, 161.9978,	//Ba
		0.733, 0.258, 2.05, 1.96, 23.0, 11.8, -152, 14.4, 134, 14.9,						//Ba+2
		0.8405, 0.2791, 2.3863, 2.1410, 4.6139, 10.3400, 4.1514, 41.9148, 4.7949, 132.0204,	//La
		0.493, 0.167, 1.10, 1.11, 1.50, 3.11, 2.70, 9.61, 1.08, 21.2,						//La+3
		0.8551, 0.2805, 2.3915, 2.1200, 4.5772, 10.1808, 5.0278, 42.0633, 4.5118, 130.9893,	//Ce
		0.560, 0.190, 1.35, 1.30, 1.59, 3.93, 2.63, 10.7, 0.706, 23.8,						//Ce+3
		0.483, 0.165, 1.09, 1.10, 1.34, 3.02, 2.45, 8.85, 0.797, 18.8,						//Ce+4
		0.9096, 0.2939, 2.5313, 2.2471, 4.5266, 10.8266, 4.6376, 48.8842, 4.3690, 147.6020,	//Pr
		0.663, 0.226, 1.73, 1.61, 2.35, 6.33, 0.351, 11.0, 1.59, 16.9,						//Pr+3
		0.512, 0.177, 1.19, 1.17, 1.33, 3.28, 2.36, 8.94, 0.690, 19.3,						//Pr+4
		0.8807, 0.2802, 2.4183, 2.0836, 4.4448, 10.0357, 4.6858, 47.4506, 4.1725, 146.9976,	//Nd
		0.501, 0.162, 1.18, 1.08, 1.45, 3.06, 2.53, 8.80, 0.920, 19.6,						//Nd+3
		0.9471, 0.2977, 2.5463, 2.2276, 4.3523, 10.5762, 4.4789, 49.3619, 3.9080, 145.3580,	//Pm
		0.496, 0.156, 1.20, 1.05, 1.47, 3.07, 2.43, 8.56, 0.943, 19.2,						//Pm+3
		0.9699, 0.3003, 2.5837, 2.2447, 4.2778, 10.6487, 4.4575, 50.7994, 3.5985, 146.4176,	//Sm
		0.518, 0.163, 1.24, 1.08, 1.43, 3.11, 2.40, 8.52, 0.781, 19.1,						//Sm+3
		0.8694, 0.2653, 2.2413, 1.8590, 3.9196, 8.3998, 3.9694, 36.7397, 4.5498, 125.7089,	//Eu
		0.613, 0.190, 1.53, 1.27, 1.84, 4.18, 2.46, 10.7, 0.714, 26.2,						//Eu+2
		0.496, 0.152, 1.21, 1.01, 1.45, 2.95, 2.36, 8.18, 0.774, 18.5,						//Eu+3
		0.9673, 0.2909, 2.4702, 2.1014, 4.1148, 9.7067, 4.4972, 43.4270, 3.2099, 125.9474,	//Gd
		0.490, 0.148, 1.19, 0.974, 1.42, 2.81, 2.30, 7.78, 0.795, 17.7,						//Gd+3
		0.9325, 0.2761, 2.3673, 1.9511, 3.8791, 8.9296, 3.9674, 41.5937, 3.7996, 131.0122,	//Tb
		0.503, 0.150, 1.22, 0.982, 1.42, 2.86, 2.24, 7.77, 0.710, 17.7,						//Tb+3
		0.9505, 0.2773, 2.3705, 1.9469, 3.8218, 8.8862, 4.0471, 43.0938, 3.4451, 133.1396,	//Dy
		0.503, 0.148, 1.24, 0.970, 1.44, 2.88, 2.17, 7.73, 0.643, 17.6,						//Dy+3
		0.9248, 0.2660, 2.2428, 1.8183, 3.6182, 7.9655, 3.7910, 33.1129, 3.7912, 101.8139,	//Ho
		0.456, 0.129, 1.17, 0.869, 1.43, 2.61, 2.15, 7.24, 0.692, 16.7,						//Ho+3
		1.0373, 0.2944, 2.4824, 2.0797, 3.6558, 9.4156, 3.8925, 45.8056, 3.0056, 132.7720,	//Er
		0.522, 0.150, 1.28, 0.964, 1.46, 2.93, 2.05, 7.72, 0.508, 17.8,						//Er+3
		1.0075, 0.2816, 2.3787, 1.9486, 3.5440, 8.7162, 3.6932, 41.8420, 3.1759, 125.0320,	//Tm
		0.475, 0.132, 1.20, 0.864, 1.42, 2.60, 2.05, 7.09, 0.584, 16.6,						//Tm+3
		1.0347, 0.2855, 2.3911, 1.9679, 3.4619, 8.7619, 3.6556, 42.3304, 3.0052, 125.6499,	//Yb
		0.508, 0.136, 1.37, 0.922, 1.76, 3.12, 2.23, 8.722, 0.584, 23.7,					//Yb+2
		0.498, 0.138, 1.22, 0.881, 1.39, 2.63, 1.97, 6.99, 0.559, 16.3,						//Yb+3
		0.9927, 0.2701, 2.2436, 1.8073, 3.3554, 7.8112, 3.7813, 34.4849, 3.0994, 103.3526,	//Lu
		0.483, 0.131, 1.21, 0.845, 1.41, 2.57, 1.94, 6.88, 0.522, 16.2,						//Lu+3
		1.0295, 0.2761, 2.2911, 1.8625, 3.4110, 8.0961, 3.9497, 34.2712, 2.4925, 98.5295,	//Hf
		0.522, 0.145, 1.22, 0.896, 1.37, 2.74, 1.68, 6.91, 0.312, 16.1,						//Hf+4
		1.0190, 0.2694, 2.2291, 1.7962, 3.4097, 7.6944, 3.9252, 31.0942, 2.2679, 91.1089,	//Ta
		0.569, 0.161, 1.26, 0.972, 0.979, 2.76, 1.29, 5.40, 0.551, 10.9,					//Ta+5
		0.9853, 0.2569, 2.1167, 1.6745, 3.3570, 7.0098, 3.7981, 26.9234, 2.2798, 81.3910,	//W
		0.181, 0.0118, 0.873, 0.442, 1.18, 1.52, 1.48, 4.35, 0.56, 9.42,					//W+6
		0.9914, 0.2548, 2.0858, 1.6518, 3.4531, 6.8845, 3.8812, 26.7234, 1.8526, 81.7215,	//Re
		0.9813, 0.2487, 2.0322, 1.5973, 3.3665, 6.4737, 3.6235, 23.2817, 1.9741, 70.9254,	//Os
		0.586, 0.155, 1.31, 0.938, 1.63, 3.19, 1.71, 7.84, 0.540, 19.3,						//Os+4
		1.0194, 0.2554, 2.0645, 1.6475, 3.4425, 6.5966, 3.4914, 23.2269, 1.6976, 70.0272,	//Ir
		0.692, 0.182, 1.37, 1.04, 1.80, 3.47, 1.97, 8.51, 0.804, 21.2,						//Ir+3
		0.653, 0.174, 1.29, 0.992, 1.50, 3.14, 1.74, 7.22, 0.683, 17.2,						//Ir+4
		0.9148, 0.2263, 1.8096, 1.3813, 3.2134, 5.3243, 3.2953, 17.5987, 1.5754, 60.0171,	//Pt
		0.872, 0.223, 1.68, 1.35, 2.63, 4.99, 1.93, 13.6, 0.475, 33.0,						//Pt+2
		0.550, 0.142, 1.21, 0.833, 1.62, 2.81, 1.95, 7.21, 0.610, 17.7,						//Pt+4
		0.9674, 0.2358, 1.8916, 1.4712, 3.3993, 5.6758, 3.0524, 18.7119, 1.2607, 61.5286,	//Au
		0.811, 0.201, 1.57, 1.18, 2.63, 4.25, 2.68, 12.1, 0.998, 34.4,						//Au+1
		0.722, 0.184, 1.39, 1.06, 1.94, 3.58, 1.94, 8.56, 0.699, 20.4,						//Au+3
		1.0033, 0.2413, 1.9469, 1.5298, 3.4396, 5.8009, 3.1548, 19.4520, 1.4180, 60.5753,	//Hg
		0.796, 0.194, 1.56, 1.14, 2.72, 4.21, 2.76, 12.4, 1.18, 36.2,						//Hg+1
		0.773, 0.191, 1.49, 1.12, 2.45, 4.00, 2.23, 1.08, 0.570, 27.6,						//Hg+2
		1.0689, 0.2540, 2.1038, 1.6715, 3.6039, 6.3509, 3.4927, 23.1531, 1.8283, 78.7099,	//Tl
		0.820, 0.197, 1.57, 1.16, 2.78, 4.23, 2.82, 12.7, 1.31, 35.7,						//Tl+1
		0.836, 0.208, 1.43, 1.20, 0.394, 2.57, 2.51, 4.86, 1.50, 13.5,						//Tl+3
		1.0891, 0.2552, 2.1867, 1.7174, 3.6160, 6.5131, 3.8031, 23.9170, 1.8994, 74.7039,	//Pb
		0.755, 0.181, 1.44, 1.05, 2.48, 3.75, 2.45, 10.6, 1.03, 27.9,						//Pb+2
		0.583, 0.144, 1.14, 0.796, 1.60, 2.58, 2.06, 6.22, 0.662, 14.8,						//Pb+4
		1.1007, 0.2546, 2.2306, 1.7351, 3.5689, 6.4948, 4.1549, 23.6464, 2.0382, 70.3780,	//Bi
		0.708, 0.170, 1.35, 0.981, 2.28, 3.44, 2.18, 9.41, 0.797, 23.7,						//Bi+3
		0.654, 0.162, 1.18, 0.905, 1.25, 2.68, 1.66, 5.14, 0.778, 11.2,						//Bi+5
		1.1568, 0.2648, 2.4353, 1.8786, 3.6459, 7.1749, 4.4064, 25.1766, 1.7179, 69.2821,	//Po
		1.0909, 0.2466, 2.1976, 1.6707, 3.3831, 6.0197, 4.6700, 207657, 2.1277, 57.2663,	//At
		1.0756, 0.2402, 2.1630, 1.6169, 3.3178, 5.7644, 4.8852, 19.4568, 2.0489, 52.5009,	//Rn
		1.4282, 0.3183, 3.5081, 2.6889, 5.6767, 13.4816, 4.1964, 54.3866, 3.8946, 200.8321,	//Fr
		1.3127, 0.2887, 3.1243, 2.2897, 5.2988, 10.8276, 5.3891, 43.5389, 5.4133, 145.6109,	//Ra
		0.911, 0.204, 1.65, 1.26, 2.53, 1.03, 3.62, 12.6, 1.58, 30.0,						//Ra+2
		1.3128, 0.2861, 3.1021, 2.2509, 5.3385, 10.5287, 5.9611, 41.7796, 4.7562, 128.2973,	//Ac
		0.915, 0.205, 1.64, 1.28, 2.26, 3.92, 3.18, 11.3, 1.25, 25.1,						//Ac+3
		1.2553, 0.2701, 2.9178, 2.0636, 5.0862, 9.3051, 6.1206, 34.5977, 4.7122, 107.9200,	//Th
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 														//Th+4
		1.3218, 0.2827, 3.1444, 2.2250, 5.4371, 10.2454, 5.6444, 41.1162, 4.0107, 124.4449,	//Pa
		1.3382, 0.2838, 3.2043, 2.2452, 5.4558, 10.2519, 5.4839, 41.7251, 3.6342, 124.9023,	//U
		1.14, 0.250, 2.48, 1.84, 3.61, 7.39, 1.13, 18.0, 0.900, 22.7,						//U+3
		1.09, 0.243, 2.32, 1.75, 12.0, 7.79, -9.11, 8.31, 2.15, 16.5,						//U+4
		0.687, 0.154, 1.14, 0.861, 1.83, 2.58, 2.53, 7.70, 0.957, 15.9,						//U+6
		1.5193, 0.3213, 4.0053, 2.8206, 6.5327, 14.8878, -0.1402, 68.9103, 6.7489, 81.7257,	//Np
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+6
		1.3517, 0.2813, 3.2937, 2.2418, 5.3212, 9.9952, 4.6466, 42.7939, 3.5714, 132.1739,	//Pu
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+6
		1.2135, 0.2483, 2.7962, 1.8437, 4.7545, 7.5421, 4.5731, 29.3841, 4.4786, 112.4579,	//Am
		1.2937, 0.2638, 3.1100, 2.0341, 5.0393, 8.7101, 4.7546, 35.2992, 3.5031, 109.4972,	//Cm
		1.2915, 0.2611, 3.1023, 2.0023, 4.9309, 8.4377, 4.6009, 34.1559, 3.4661, 105.8911,	//Bk
		1.2089, 0.2421, 2.7391, 1.7487, 4.3482, 6.7262, 4.0047, 23.2153, 4.6497, 80.3108,	//Cf
		// Atomic groups:
		0.1796, 73.76, 0.8554, 5.399, 1.75, 27.15, 0.05001, 0.1116, 0.2037, 1.062,			//CH
		0.1575, 89.04, 0.8528, 4.637, 2.359, 30.92, 0.00496, -0.344, 0.1935, 0.6172,		// CH2
		0.4245, 4.092, 0.4256, 4.094, 0.2008, 74.32, 2.884, 33.65, 0.16, 0.4189,			// CH3
		0.1568, 64.9, 0.222, 1.017, 0.8391, 4.656, 1.469, 23.17, 0.05579, 0.11,				// NH
		1.991, 25.94, 0.2351, 74.54, 0.8575, 3.893, 5.336, 0.3422, -5.147, 0.3388,			// NH2
		-0.1646, 168.7, 0.2896, 147.3, 0.838, 3.546, 0.1736, 0.4059, 2.668, 29.57,			// NH3
		0.1597, 53.82, 0.2445, 0.7846, 0.8406, 4.042, 1.235, 20.92, 0.03234, -0.01414,		// OH
		-78.51, 9.013, 80.62, 9.014, 0.6401, 1.924, 2.665, 37.71, 0.2755, 0.2941,			// SH
		// Ions that had no x-ray form factors:
		0.0421, 0.0609, 0.210, 0.559, 0.852, 2.96, 1.82, 11.5, 1.17, 37.7,					// O-2
		0.132, 0.109, 0.292, 0.695, 0.703, 2.39, 0.692, 5.65, 0.0959, 14.7					//Cr+4
		;

#pragma endregion
}

FACC electronPDBAmplitude::GetVoxelStepSize() {
	return voxelStep;
}

void electronPDBAmplitude::SetVoxelStepSize(FACC vS) {
	voxelStep = vS;
}

void electronPDBAmplitude::SetOutputSlices(bool bOutput, std::string outPath /*= ""*/) {
	this->pdb.bOutputSlices = bOutput;
	if(bOutput) {
		this->pdb.slicesBasePathSt = outPath;
	}
}

void electronPDBAmplitude::SetSolventOnlyCalculation(bool bOnlySol) {
	this->pdb.bOnlySolvent = bOnlySol;
}

void electronPDBAmplitude::SetFillHoles(bool bFillHole) {
	this->pdb.bFillHoles = bFillHole;

}

PDB_READER_ERRS electronPDBAmplitude::CalculateSolventAmp(FACC stepSize, FACC solED, FACC outerSolvED, FACC solRad, FACC atmRadDiffIn /*= 0.0*/) {
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

PDB_READER_ERRS electronPDBAmplitude::FindPDBRanges() {
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

PDB_READER_ERRS electronPDBAmplitude::AllocateSolventSpace() {
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

void electronWriteNewBMPFile(fs::path fName, unsigned char *img, int width, int height)
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

void electronWriteBMPfile(fs::path fName, array3 &data, int slice, int width, int height) {
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


void electronPDBAmplitude::WriteEigenSlicesToFile(string filebase)
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
		electronWriteNewBMPFile(fnp.replace_extension("bmp"), imageData.data(), _solvent_space.dimensions()(1), _solvent_space.dimensions()(2));
		//i = _solvent_space.dimensions()(0) + 1; // Write only the center slice
	}
	std::cout << "Finished writing slices\n";

}

void electronPDBAmplitude::GetHeader(unsigned int depth, JsonWriter &writer)
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

void electronPDBAmplitude::GetHeader(unsigned int depth, std::string &header) {
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

ATOM_RADIUS_TYPE electronIntToAtom_Radius_Type(int ind) {
	switch (ind) {
	case 1: return RAD_VDW;
	case 2: return RAD_EMP;
	case 3: return RAD_CALC;
	case 4: return RAD_DUMMY_ATOMS_ONLY;
	case 5: return RAD_DUMMY_ATOMS_RADII;
	default: return RAD_UNINITIALIZED;
	}
}


void electronPDBAmplitude::PreCalculate(VectorXd& p, int nLayers) {
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
	pdb.SetRadiusType( electronIntToAtom_Radius_Type( int(p(9) + 0.1) ) );

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

	electronAffCalculator.Initialize(bitwiseCalculationFlags, pdb.sortedIonInd.size(), numUIons, pdb.sortedCoeffs.data(), numberOfIonsPerIndex.data());

	if (bitwiseCalculationFlags & CALC_DUMMY_SOLVENT)
		electronAffCalculator.SetSolventED(solventED, c1, uniIonRads.data() , pdb.bOnlySolvent);
	if (bitwiseCalculationFlags & CALC_ANOMALOUS)
	{
		std::vector<float2> anomfPrimesAsFloat2;

		size_t sz = pdb.sortedAnomfPrimes.size();
		anomfPrimesAsFloat2.resize(sz);
 		Eigen::Map<Eigen::ArrayXf>((float*)anomfPrimesAsFloat2.data(), 2 * sz) =
 			(Eigen::Map<Eigen::ArrayXf>((float*)pdb.sortedAnomfPrimes.data(), 2 * sz)).cast<float>();

		electronAffCalculator.SetAnomalousFactors((float2*)(anomfPrimesAsFloat2.data()));
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


void electronPDBAmplitude::PrepareParametersForGPU(std::vector<float4>& solCOM, std::vector<int4>& solDims,
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
	std::vector<electronSolventBoxEntry> solEntries(solventBoxDims.size());

	for(u64 i = 0; i < solventBoxDims.size(); i++) {
		solEntries[i].loc = make_float4((solventBoxCOM[i])[0], (solventBoxCOM[i])[1], (solventBoxCOM[i])[2], 0.0f);
		solEntries[i].len = make_int4((solventBoxDims[i])[0], (solventBoxDims[i])[1], (solventBoxDims[i])[2], 0);
	}
	std::sort(solEntries.begin(), solEntries.end(), electronSortSolventBoxEntry);
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
	std::sort(solEntries.begin(), solEntries.end(), electronSortSolventBoxEntry);
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

std::string electronPDBAmplitude::Hash() const
{
	std::string str = BACKEND_VERSION "PDB: ";
	str += std::to_string(voxelStep) + std::to_string(solventED) + std::to_string(c1)
		+ std::to_string(solventRad) + std::to_string(solvationThickness) + std::to_string(outerSolventED);

	str += pdb_hash;

	return md5(str);
}

bool electronPDBAmplitude::SetModel(Workspace& workspace) {
	// !!!!This has to changed appropriatly when converting to E+!!!!
	bool isE = true;
	int divider = 9;

	if (isE) { divider = 10; }

	if (!g_useGPUAndAvailable)
		return false;
	
	if (!electronGpuSetPDB)
		electronGpuSetPDB = (electronGPUDirectSetPDB_t)GPUDirect_SetPDBDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_SetPDBDLL");
	if(!electronGpuSetPDB)
		return false;
				
	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return electronGpuSetPDB(workspace, (const float4 *)&pdb.atomLocs[0], &pdb.sortedCoeffIonInd[0],
					 pdb.atomLocs.size(), &pdb.sortedCoeffs[0], 
					 &pdb.atomsPerIon[0],
					 pdb.sortedCoeffs.size() / divider);
}

bool electronPDBAmplitude::SetParameters(Workspace& workspace, const double *params, unsigned int numParams) {
	// TODO: Later (solvents and stuff)
	if (!g_useGPUAndAvailable)
		return false;

	return true;
}

bool electronPDBAmplitude::ComputeOrientation(Workspace& workspace, float3 rotation) {
	if (!g_useGPUAndAvailable)
		return false;

	if (!electronGpuPDBAmplitude)
		electronGpuPDBAmplitude = (electronGPUDirectPDBAmplitude_t)GPUDirect_PDBAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_PDBAmplitudeDLL");
	if(!electronGpuPDBAmplitude)
		return false;
	
	return electronGpuPDBAmplitude(workspace, rotation);
}

std::string electronPDBAmplitude::GetName() const {
	return "PDB: " + pdb.fn;
}


bool electronPDBAmplitude::SetModel( GridWorkspace& workspace ) {
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
	electronAffCalculator.GetQMajorAFFMatrix(affs.data(), workspace.qLayers, workspace.stepSize);

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

bool electronPDBAmplitude::CalculateGridGPU( GridWorkspace& workspace ) {

	if (!g_useGPUAndAvailable)
		return false;

	if (!electronGpuPDBHybridAmplitude)
		electronGpuPDBHybridAmplitude = (electronGPUHybridPDBAmplitude_t)GPUHybrid_PDBAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_PDBAmplitudeDLL");
	if(!electronGpuPDBHybridAmplitude)
		return false;

	return electronGpuPDBHybridAmplitude(workspace);
}

bool electronPDBAmplitude::SavePDBFile(std::ostream &output) {
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

bool electronPDBAmplitude::AssemblePDBFile( std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs ) {
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

bool electronPDBAmplitude::ImplementedHybridGPU() {
	return true;
}

void electronPDBAmplitude::SetPDBHash()
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

void electronDebyeCalTester::SetStop(int* stop) {
	pStop = stop;
}

void electronDebyeCalTester::OrganizeParameters(const VectorXd& p, int nLayers) {
	return;	// For now (during testing), there will be no parameters
	//throw std::exception("The method or operation is not implemented.");
}

void electronDebyeCalTester::PreCalculate(VectorXd& p, int nLayers) {
	// This "optimization" slows down the calculation by about a minute
	/*
	distances.resize(pdb->sortedX.size(), pdb->sortedX.size());
	for(int i = 0; i < pdb->sortedX.size(); i++) {
	for(int j = 0; j < i; j++) {
	distances(i,j) = sq(pdb->sortedX[i] - pdb->sortedX[j]) +
	sq(pdb->sortedY[i] - pdb->sortedY[j]) +
	sq(pdb->sortedZ[i] - pdb->sortedZ[j]);
	}
	}
	distances = distances.sqrt();
	*/
	typedef Eigen::Map<Eigen::Matrix<F_TYPE, 1, Eigen::Dynamic>> mp;
	int num = pdb->sortedX.size();
	sortedLocations.resize(Eigen::NoChange, num);
	sortedLocations.row(0) = (mp(pdb->sortedX.data(), num)).cast<float>();
	sortedLocations.row(1) = (mp(pdb->sortedY.data(), num)).cast<float>();
	sortedLocations.row(2) = (mp(pdb->sortedZ.data(), num)).cast<float>();
	if (sortedLocations.RowsAtCompileTime == 4)
		sortedLocations.row(3).setZero();

	pdb->SetRadiusType(RAD_DUMMY_ATOMS_ONLY);

	int comb = CALC_ATOMIC_FORMFACTORS;
	comb |= ((p(0) /*solED*/ != 0.0) ? CALC_DUMMY_SOLVENT : 0x00);

	//	comb |= (anomalousVals ? CALC_ANOMALOUS : 0x00);

	int numUnIons = pdb->atomsPerIon.size();

	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.size());
	Eigen::Map<Eigen::ArrayXf>(fAtmFFcoefs.data(), fAtmFFcoefs.size()) = (Eigen::Map<Eigen::ArrayXd>(pdb->sortedCoeffs.data(), fAtmFFcoefs.size())).cast<float>();

	if (_electronAff_calculator)
		delete _electronAff_calculator;
	_electronAff_calculator = new atomicFFCalculator(comb, num, numUnIons, fAtmFFcoefs.data(), pdb->atomsPerIon.data(), true);

	if (comb & CALC_DUMMY_SOLVENT)
	{
		std::vector<float> rads(pdb->rad->size());
		for (int i = 0; i < pdb->rad->size(); i++)
			rads[i] = pdb->rad->at(i);

		std::vector<float> ionRads(numUnIons);
		int off = 0;
		for (int i = 0; i < numUnIons; i++) {
			ionRads[i] = rads[pdb->sortedAtmInd[off]];
			off += pdb->atomsPerIon[i];
		}

		_electronAff_calculator->SetSolventED(p(0), p(2), ionRads.data());
	}
}



VectorXd electronDebyeCalTester::CalculateVector(
	const std::vector<double>& q,
	int nLayers,
	VectorXd& p /*= VectorXd( ) */,
	progressFunc progress /*= NULL*/,
	void* progressArgs /*= NULL*/)
{
	size_t points = q.size();
	VectorXd vecCPU = VectorXd::Zero(points);
	VectorXd vecGPU = VectorXd::Zero(points);
	std::vector<double> rVec(points);

	progFunc = progress;
	progArgs = progressArgs;

	PreCalculate(p, nLayers);
	clock_t cpuBeg, gpuBeg, cpuEnd, gpuEnd;

	// Determine if we have and want to use a GPU
	{
		int devCount;
		cudaError_t t = UseGPUDevice(&devCount);
	}


	if (g_useGPUAndAvailable) {
		gpuBeg = clock();

		vecGPU = CalculateVectorGPU(q, nLayers, p, progFunc, progArgs);
		gpuEnd = clock();

		printf("\n\n Timing:\n\tGPU %f seconds",
			double(gpuEnd - gpuBeg) / CLOCKS_PER_SEC);

		return vecGPU;
	}

	const double cProgMin = 0.0, cProgMax = 1.0;
	int prog = 0;

	if (progFunc)
		progFunc(progArgs, cProgMin);
	cpuBeg = clock();

#pragma omp parallel for if(true || pdb->atomLocs.size() > 500000) schedule(dynamic, q.size() / 50)
	for (int i = 0; i < q.size(); i++) {
		if (pStop && *pStop)
			continue;

		vecCPU[i] = Calculate(q[i], nLayers, p);

#pragma omp critical
		{
			if (progFunc)
				progFunc(progArgs, (cProgMax - cProgMin) * (double(++prog) / double(q.size())) + cProgMin);
		}
	}	// for i

	cpuEnd = clock();

	printf("\n\n Timing:\n\tCPU %f seconds\n",
		double(cpuEnd - cpuBeg) / CLOCKS_PER_SEC);

	return vecCPU;
}

VectorXd electronDebyeCalTester::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

double electronDebyeCalTester::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd( ) */) {
	int pdbLen = pdb->sortedX.size();

	F_TYPE res = 0.0;
	F_TYPE fq = (F_TYPE)q;

	int atmI = -1;
	int prevIon = 255;

	// Calculate the atomic form factors for this q
	Eigen::Array<float, -1, 1> atmAmps;
	//Eigen::ArrayXf atmAmps;
	F_TYPE aff = 0.0;
	F_TYPE fq10 = q / (10.0);
	atmAmps.resize(pdbLen);
	_electronAff_calculator->GetAllAFFs(atmAmps.data(), q);

	if (p(1) > 0.1) // Debye-Waller
	{
		for (int i = 0; i < pdbLen; i++)
		{
			atmAmps(i) *= exp(-(pdb->sortedBFactor[i] * fq10 * fq10 / (16. * M_PI * M_PI)));
		}
	}

	// Sum the Debye contributions
	for (int i = 1; i < pdbLen; i++) {
		res += 2.0 * ((atmAmps(i) * atmAmps.head(i)).cast<F_TYPE>() *
			SincArrayXf(
				(float(fq) * (sortedLocations.leftCols(i).colwise() - sortedLocations.col(i)).colwise().norm()).array()
			).cast<F_TYPE>()
			).sum();
	}
	res += atmAmps.square().sum();

	return res;
}

VectorXd electronDebyeCalTester::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

PDB_READER_ERRS electronDebyeCalTester::LoadPDBFile(string filename, int model /*= 0*/) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

void electronDebyeCalTester::GetHeader(unsigned int depth, JsonWriter& writer)
{
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

void electronDebyeCalTester::GetHeader(unsigned int depth, std::string& header) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

// typedef int(*GPUCalculateDebye_t)(int qValues, F_TYPE *qVals, F_TYPE *outData,
// 	F_TYPE *loc, u8 *ionInd, int numAtoms, F_TYPE *coeffs, int numCoeffs, bool bSol,
// 	u8 * atmInd, F_TYPE *rad, double solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
// 
// GPUCalculateDebye_t gpuCalcDebye = NULL;



// THE NEXT 2 FUNCTIONS SHOULD BE DELETED ONCE ElectronPDBAmplitude INHERITS PDBAmplitude !!!
typedef int(*electronGPUCalculateDebyeV2_t)(int numQValues, float qMin, float qMax,
	F_TYPE* outData,
	int numAtoms, const int* atomsPerIon,
	float4* loc, u8* ionInd, float2* anomalousVals,
	bool bBfactors, float* BFactors,
	float* coeffs, bool bSol,
	bool bSolOnly, char* atmInd, float* atmRad, float solvED, float c1,
	progressFunc progfunc, void* progargs,
	double progmin, double progmax, int* pStop);
electronGPUCalculateDebyeV2_t electronGpuCalcDebyeV2 = NULL;

//GPUCalculateDebyeV2_t gpuCalcDebyeV3MAPS = NULL;

typedef int(*electronGPUCalcDebyeV4MAPS_t)(
	int numQValues, float qMin, float qMax, F_TYPE* outData, int numAtoms,
	const int* atomsPerIon, float4* atomLocations, float2* anomFactors, bool bBfactors, float* BFactors,
	float* coeffs, bool bSol, bool bSolOnly, char* atmInd, float* atmRad, float solvED, float c1,
	progressFunc progfunc, void* progargs, double progmin, double progmax, int* pStop);
electronGPUCalcDebyeV4MAPS_t electronThisSucks = NULL;

VectorXd electronDebyeCalTester::CalculateVectorGPU(
	const std::vector<double>& q,
	int nLayers,
	VectorXd& p /*= VectorXd( ) */,
	progressFunc progress /*= NULL*/,
	void* progressArgs /*= NULL*/)
{

	if (g_useGPUAndAvailable) {
		if (sizeof(F_TYPE) == sizeof(double))
		{
			electronGpuCalcDebyeV2 = (electronGPUCalculateDebyeV2_t)GPUCalculateDebyeDv2;
			electronThisSucks = (electronGPUCalcDebyeV4MAPS_t)GPUCalcDebyeV4MAPSD;
		}
	}

	if (g_useGPUAndAvailable && (/*gpuCalcDebye == NULL || */electronGpuCalcDebyeV2 == NULL || electronThisSucks == NULL)) {
		printf("Error loading GPU functions\n");
		return VectorXd();
	}
	std::vector<F_TYPE> ftq(q.size());
	for (int i = 0; i < ftq.size(); i++)
		ftq[i] = (F_TYPE)(q[i]);
	// LOC
	std::vector<F_TYPE> loc(this->pdb->sortedX.size() * 3);

	int pos = 0;
	for (int i = 0; i < pdb->sortedX.size(); i++) {
		loc[pos++] = pdb->sortedX[i];
		loc[pos++] = pdb->sortedY[i];
		loc[pos++] = pdb->sortedZ[i];
	}
	Eigen::Matrix<F_TYPE, -1, 1, 0, -1, 1> res;
	res.resize(ftq.size());

	std::vector<float> rads(pdb->rad->size());
	for (int i = 0; i < pdb->rad->size(); i++) {
		rads[i] = pdb->rad->at(i);
	}

	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.size());
	Eigen::Map<Eigen::ArrayXf>(fAtmFFcoefs.data(), fAtmFFcoefs.size()) = (Eigen::Map<Eigen::ArrayXd>(pdb->sortedCoeffs.data(), fAtmFFcoefs.size())).cast<float>();

	std::vector<float2> anomfPrimesAsFloat2;

	if (pdb->haveAnomalousAtoms)
	{
		size_t sz = pdb->sortedAnomfPrimes.size();
		anomfPrimesAsFloat2.resize(sz);
		Eigen::Map<Eigen::ArrayXf>((float*)anomfPrimesAsFloat2.data(), 2 * sz) =
			(Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>((float*)pdb->sortedAnomfPrimes.data(), 2 * sz)).cast<float>();
	}
	bool bSol = (p.size() > 0 && p(0) > 0.);
	bool bDW = (p.size() > 1 && p(1) > 0.1);
	int gpuRes;
	const float c1 = (p.size() > 2 ? p(2) : 1.0f);

	switch (kernelVersion)
	{
		// 	case 1:
		// 		gpuRes = gpuCalcDebye(ftq.size(), &ftq[0], &res(0), &loc[0], (u8*)&pdb->sortedIonInd[0],
		// 			pdb->sortedIonInd.size(), &atmFFcoefs(0, 0), atmFFcoefs.size(), bSol,
		// 			NULL, NULL, 0.0,
		// 			progFunc, progArgs, 0.0, 1.0, pStop);
		// 		break;
	case 2:
		gpuRes = electronGpuCalcDebyeV2(ftq.size(), q.front(), q.back(), res.data(),
			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
			(float4*)pdb->atomLocs.data(),
			(u8*)pdb->sortedCoeffIonInd.data(),
			pdb->haveAnomalousAtoms ? anomfPrimesAsFloat2.data() : NULL,
			bDW, pdb->sortedBFactor.data(),
			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
			rads.data(), (float)(p[0]), c1, progress, progressArgs, 0.0, 1.0, pStop);
		break;

	case 4:
		gpuRes = electronThisSucks(ftq.size(), q.front(), q.back(), res.data(),
			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
			(float4*)pdb->atomLocs.data(), NULL,
			bDW, pdb->sortedBFactor.data(),
			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
			rads.data(), (float)(p[0]), c1, progress, progressArgs, 0.0, 1.0, pStop);
		break;
	default:

		break;
	}

	return res.cast<double>();
}


void electronDebyeCalTester::electronInitialize() {
	pdb = nullptr;
	_electronAff_calculator = nullptr;

	atmFFcoefs.resize(ELECTRON_NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
	// I'm almost certain we can delete this part of the code but I changed just in case.
#pragma region Atomic form factor coefficients - Peng
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.0573, 18.9525, 0.1195, 38.6269,	// H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653,	// He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337,	// Li
		0.00460, 0.0358, 0.0165, 0.239, 0.0435, 0.879, 0.0649, 2.64, 0.0270, 7.09, 			//Li+1
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273,	// Be
		0.00340, 0.0267, 0.0103, 0.162, 0.0233, 0.531, 0.0325, 1.48, 0.0120, 3.88, 			//Be+2
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314,	// B
		0.0893, 0.2465, 0.2563, 1.7100, 0.7570, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523,	// C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715, 48.1431,	// N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.6943, 0.6990, 12.7105, 0.2039, 32.4726,	// O
		0.205, 0.397, 0.628, 0.264, 1.17, 8.80, 1.03, 27.1, 0.290, 91.8, 					//O-1
		0.1083, 0.2057, 0.3175, 1.3439, 0.6487, 4.2788, 0.5846, 11.3932, 0.1421, 28.7881,	// F
		0.134, 0.228, 0.391, 1.47, 0.814, 4.68, 0.928, 13.2, 0.347, 36.0, 					//F-1
		0.1269, 0.2200, 0.3535, 1.3779, 0.5582, 4.0203, 0.4674, 9.4934, 0.1460, 23.1278,	// Ne
		0.2142, 0.334, 0.6853, 2.3446, 0.7692, 10.0830, 1.6589, 48.3037, 1.4482, 137.2700,	// Na
		0.0256, 0.0397, 0.0919, 0.287, 0.297, 1.18, 0.514, 3.75, 0.199, 10.8,				// Na+1
		0.2314, 0.3278, 0.6866, 2.2720, 0.9677, 10.9241, 2.1182, 32.2898, 1.1339, 101.9748,	// Mg
		0.0210, 0.0331, 0.0672, 0.222, 0.198, 0.838, 0.368, 2.48, 0.174, 6.75,				// Mg+2
		0.2390, 0.3138, 0.6573, 2.1063, 1.2011, 10.4163, 2.5586, 34.4552, 1.2312, 98.5344,	// Al
		0.0192, 0.0306, 0.0579, 0.198, 0.163, 0.713, 0.284, 2.04, 0.114, 5.25,				// Al+3
		0.2519, 0.3075, 0.6372, 2.0174, 1.3795, 9.6746, 2.5082, 29.3744, 1.0500, 80.4732,	// Si_v
		0.192, 0.359, 0.289, 1.96, 0.100, 9.34, -0.0728, 11.1, 0.00120, 13.4,				// Si+4
		0.2548, 0.2908, 0.6106, 1.8740, 1.4541, 8.5176, 2.3204, 24.3434, 0.8477, 63.2996,	// P
		0.2497, 0.2681, 0.5628, 1.6711, 1.3899, 7.0267, 2.1865, 19.5377, 0.7715, 50.3888,	// S
		0.2443, 0.2468, 0.5397, 1.5242, 1.3919, 6.1537, 2.0197, 16.6687, 0.6621, 42.3086,	// Cl
		0.265, 0.252, 0.596, 1.56, 1.60, 6.21, 2.69, 17.8, 1.23, 47.8,						// Cl-1
		0.2385, 0.2289, 0.5017, 1.3694, 1.3128, 5.2561, 1.8899, 14.0928, 0.6079, 35.5361,	//Ar
		0.4115, 0.3703, 1.4031, 3.3874, 2.2784, 13.1029, 2.6742, 68.9592, 2.2162, 194.4329,	//K
		0.199, 0.192, 0.396, 1.10, 0.928, 3.91, 1.45, 9.75, 0.450, 23.4,					//K+1
		0.4054, 0.3499, 1.880, 3.0991, 2.1602, 11.9608, 3.7532, 53.9353, 2.2063, 142.3892,	//Ca
		0.164, 0.157, 0.327, 0.894, 0.743, 3.15, 1.16, 7.67, 0.307, 17.7,					//Ca+2
		0.3787, 0.3133, 1.2181, 2.5856, 2.0594, 9.5813, 3.2618, 41.7688, 2.3870, 116.7282,	//Sc
		0.163, 0.157, 0.307, 0.899, 0.716, 3.06, 0.880, 7.05, 0.139, 16.1,					//Sc+3
		0.3825, 0.3040, 1.2598, 2.4863, 2.0008, 9.2783, 3.0617, 39.0751, 2.0694, 109.4583,	//Ti
		0.399, 0.376, 1.04, 2.74, 1.21, 8.10, -0.0797, 14.2, 0.352, 23.2,					//Ti+2
		0.364, 0.364, 0.919, 2.67, 1.35, 8.18, -0.933, 11.8, 0.589, 14.9,					//Ti+3
		0.116, 0.108, 0.256, 0.655, 0.565, 2.38, 0.772, 5.51, 0.32, 12.3,					//Ti+4
		0.3876, 0.2967, 1.2750, 2.3780, 1.9109, 8.7981, 2.8314, 35.9528, 1.8979, 101.7201,	//V
		0.317, 0.269, 0.939, 2.09, 1.49, 7.22, -1.31, 15.2, 1.47, 17.6,						//V+2
		0.341, 0.321, 0.805, 2.23, 0.942, 5.99, 0.0783, 13.4, 0.156, 16.9,					//V+3
		0.0367, 0.0330, 0.124, 0.222, 0.244, 0.824, 0.723, 2.8, 0.435, 6.70,				//V+5
		0.4046, 0.2986, 1.3696, 2.3958, 1.8941, 9.1406, 2.0800, 37.4701, 1.2196, 113.7121,	//Cr
		0.237, 0.177, 0.634, 1.35, 1.23, 4.30, 0.713, 12.2, 0.0859, 39.0,					//Cr+2
		0.393, 0.359, 1.05, 2.57, 1.62, 8.68, -1.15, 11.0, 0.407, 15.8,						//Cr+3
		0.3796, 0.2699, 1.2094, 2.0455, 1.7815, 7.4726, 2.5420, 31.0604, 1.5937, 91.5622,	//Mn
		0.0576, 0.0398, 0.210, 0.284, 0.604, 1.29, 1.32, 4.23, 0.659, 14.5,					//Mn+2
		0.116, 0.0117, 0.523, 0.876, 0.881, 3.06, 0.589, 6.44, 0.214, 14.3,					//Mn+3
		0.381, 0.354, 1.83, 2.72, -1.33, 3.47, 0.995, 5.47, 0.0618, 16.1,					//Mn+4
		0.3946, 0.2717, 1.2725, 2.0443, 1.7031, 7.6007, 2.3140, 29.9714, 1.4795, 86.2265,	//Fe
		0.307, 0.230, 0.838, 1.62, 1.11, 4.87, 0.280, 10.7, 0.277, 19.2,					//Fe+2
		0.198, 0.154, 0.384, 0.893, 0.889, 2.62, 0.709, 6.65, 0.117, 18.0,				 	//Fe+3
		0.4118, 0.2742, 1.3161, 2.0372, 1.6493, 7.7205, 2.1930, 29.9680, 1.2830, 84.9383,	//Co
		0.213, 0.148, 0.488, 0.939, 0.998, 2.78, 0.828, 7.31, 0.230, 20.7,					//Co+2
		0.331, 0.267, 0.487, 1.41, 0.729, 2.89, 0.608, 6.45, 0.131, 15.8,					//Co+3
		0.3860, 0.2478, 1.1765, 1.7660, 1.5451, 6.3107, 2.0730, 25.2204, 1.3814, 74.3146,	//Ni
		0.338, 0.237, 0.982, 1.67, 1.32, 5.73, -3.56, 11.4, 3.62, 12.1,						//Ni+2
		0.347, 0.260, 0.877, 1.71, 0.790, 4.75, 0.0538, 7.51, 0.192, 13.0,					//Ni+3
		0.4314, 0.2694, 1.3208, 1.9223, 1.5236, 7.3474, 1.4671, 28.9892, 0.8562, 90.6246,	//Cu
		0.312, 0.201, 0.812, 1.31, 1.11, 3.80, 0.794, 10.5, 0.257, 28.2,					//Cu+1
		0.224, 0.145, 0.544, 0.933, 0.970, 2.69, 0.727, 7.11, 0.1882, 19.4,					//Cu+2
		0.4288, 0.2593, 1.2646, 1.7998, 1.4472, 6.7500, 1.8294, 25.5860, 1.0934, 73.5284,	//Zn
		0.252, 0.161, 0.600, 1.01, 0.917, 2.76, 0.663, 7.08, 0.161, 19.0,					//Zn+2
		0.4818, 0.2825, 1.4032, 1.9785, 1.6561, 8.7546, 2.4605, 32.5238, 1.1054, 98.5523,	//Ga
		0.391, 0.264, 0.947, 1.65, 0.690, 4.82, 0.0709, 10.7, 0.0653, 15.2,					//Ga+3
		0.4655, 0.2647, 1.3014, 1.7926, 1.6088, 7.6071, 2.6998, 26.5541, 1.3003, 77.5238,	//Ge
		0.346, 0.232, 0.830, 1.45, 0.599, 4.08, 0.0949, 13.2, -0.0217, 29.5,				//Ge+4
		0.4517, 0.2493, 1.2229, 1.6436, 1.5852, 6.8154, 2.7958, 22.3681, 1.2638, 62.0390,	//As
		0.4477, 0.2405, 1.1678, 1.5442, 1.5843, 6.3231, 2.8087, 19.4610, 1.1956, 52.0233,	//Se
		0.4798, 0.2504, 1.1948, 1.5963, 1.8695, 6.9653, 2.6953, 19.8492, 0.8203, 50.3233,	//Br
		0.125, 0.0530, 0.563, 0.469, 1.43, 2.15, 3.25, 11.1, 3.22, 38.9,					//Br-1
		0.4546, 0.2309, 1.0993, 1.4279, 1.76966, 5.9449, 2.7068, 16.6752, 0.8672, 42.2243,	//Kr
		1.0160, 0.4853, 2.8528, 5.0925, 3.5466, 25.7851, -7.7804, 130.4515, 12.1148, 138.6775,//Rb
		0.368, 0.187, 0.884, 1.12, 1.12, 3.98, 2.26, 10.9, 0.881, 26.6,						//Rb+1
		0.6703, 0.3190, 1.4926, 2.2287, 3.3368, 10.3504, 4.4600, 52.3291, 3.1501, 151.2216,	//Sr
		0.346, 0.176, 0.804, 1.04, 0.988, 3.59, 1.89, 9.32, 0.609, 21.4,					//Sr+2
		0.6894, 0.3189, 1.5474, 2.2904, 3.2450, 10.0062, 4.2126, 44.0771, 2.9764, 125.0120,	//Y
		0.465, 0.240, 0.923, 1.43, 2.41, 6.45, -2.31, 9.97, 2.48, 12.2,						//Y+3
		0.6719, 0.3036, 1.4684, 2.1249, 3.1668, 8.9236, 3.9957, 36.8458, 2.8920, 108.2049,	//Zr
		0.34, 0.113, 0.642, 0.736, 0.747, 2.54, 1.47, 6.72, 0.377, 14.7,					//Zr+4
		0.6123, 0.2709, 1.2677, 1.7683, 3.0348, 7.2489, 3.3841, 27.9465, 2.3683, 98.5624,	//Nb
		0.377, 0.184, 0.749, 1.02, 1.29, 3.80, 1.61, 9.44, 0.481, 25.7,						//Nb+3
		0.0828, 0.0369, 0.271, 0.261, 0.654, 0.957, 1.24, 3.94, 0.829, 9.44,				//Nb+5
		0.6773, 0.2920, 1.4798, 2.0606, 3.1788, 8.1129, 3.0824, 30.5336, 1.8384, 100.0658,	//Mo
		0.401, 0.191, 0.756, 1.06, 1.38, 3.84, 1.58, 9.38, 0.497, 24.6,						//Mo+3
		0.479, 0.241, 0.846, 1.46, 15.6, 6.79, -15.2, 7.13, 1.60, 10.4,						//Mo+5
		0.203, 0.0971, 0.567, 0.647, 0.646, 2.28, 1.16, 5.61, 0.171, 12.4,					//Mo+6
		0.7082, 0.2976, 1.6392, 2.2106, 3.1993, 8.5246, 3.4327, 33.1456, 1.8711, 96.6377,	//Tc
		0.6735, 0.2773, 1.4934, 1.9716, 3.0966, 7.3249, 2.7254, 26.6891, 1.5597, 90.5581,	//Ru
		0.428, 0.191, 0.773, 1.09, 1.55, 3.82, 1.46, 9.08, 0.486, 21.7,						//Ru+3
		0.2882, 0.125, 0.653, 0.753, 1.14, 2.85, 1.53, 7.01, 0.418, 17.5,					//Ru+4
		0.6413, 0.2580, 1.3690, 1.7721, 2.9854, 6.3854, 2.6952, 23.2549, 1.5433, 58.1517,	//Rh
		0.352, 0.151, 0.723, 0.878, 1.50, 3.28, 1.63, 8.16, 0.499, 20.7,					//Rh+3
		0.397, 0.177, 0.725, 1.01, 1.51, 3.62, 1.19, 8.56, 0.251, 18.9,						//Rh+4
		0.5904, 0.2324, 1.1775, 1.5019, 2.6519, 5.1591, 2.2875, 15.5428, 0.8689, 46.8213,	//Pd
		0.935, 0.393, 3.11, 4.06, 24.6, 43.1, -43.6, 54.0, 21.2, 69.8,						//Pd+2
		0.348, 0.151, 0.640, 0.832, 1.22, 2.85, 1.45, 6.59, 0.427, 15.6,					//Pd+4
		0.6377, 0.2466, 1.3790, 1.6974, 2.8294, 5.7656, 2.3631, 20.0943, 1.4553, 76.7372,	//Ag
		0.503, 0.199, 0.940, 1.19, 2.17, 4.05, 1.99, 11.3, 0.726, 32.4,						//Ag+1
		0.431, 0.175, 0.756, 0.979, 1.72, 3.30, 1.78, 8.24, 0.526, 21.4,					//Ag+2
		0.6364, 0.2407, 1.4247, 1.6823, 2.7802, 5.6588, 2.5973, 20.7219, 1.7886, 69.1109,	//Cd
		0.425, 0.168, 0.745, 0.944, 1.73, 3.14, 1.74, 7.84, 0.487, 20.4,					//Cd+2
		0.6768, 0.2522, 1.6589, 1.8545, 2.7740, 6.2936, 3.1835, 25.1457, 2.1326, 84.5448,	//In
		0.417, 0.164, 0.755, 0.960, 1.59, 3.08, 1.36, 7.03, 0.451, 16.1,					//In+3
		0.7224, 0.2651, 1.9610, 2.0604, 2.7161, 7.3011, 3.5603, 27.5493, 1.8972, 81.3349,	//Sn
		0.797, 0.317, 2.13, 2.51, 2.15, 9.04, -1.64, 24.2, 2.72, 26.4,						//Sn+2
		0.261, 0.0957, 0.642, 0.625, 1.53, 2.51, 1.36, 6.31, 0.177, 15.9,					//Sn+4
		0.7106, 0.2562, 1.9247, 1.9646, 2.6149, 6.8852, 3.8322, 24.7648, 1.8899, 68.9168,	//Sb
		0.552, 0.212, 1.14, 1.42, 1.87, 4.21, 1.36, 12.5, 0.414, 29.0,						//Sb+3
		0.377, 0.151, 0.588, 0.812, 1.22, 2.40, 1.18, 5.27, 0.244, 11.9,					//Sb+5
		0.6947, 0.2459, 1.8690, 1.8542, 2.5356, 6.4411, 4.0013, 22.1730, 1.8955, 59.2206,	//Te
		0.7047, 0.2455, 1.9484, 1.8638, 2.5940, 6.7639, 4.1526, 21.8007, 1.5057, 56.4395,	//I
		0.901, 0.312, 2.80, 2.59, 5.61, 4.1, -8.69, 34.4, 12.6, 39.5,						//I-1
		0.6737, 0.2305, 1.7908, 1.6890, 2.4129, 5.8218, 4.5100, 18.3928, 1.7058, 47.2496,	//Xe
		1.2704, 0.4356, 3.8018, 4.2058, 5.6618, 23.4342, 0.9205, 136.7783, 4.8105, 171.7561,//Cs
		0.587, 0.200, 1.40, 1.38, 1.87, 4.12, 3.48, 13.0, 1.67, 31.8,						//Cs+1
		0.9049, 0.3066, 2.6076, 2.4363, 4.8498, 12.1821, 5.1603, 54.6135, 4.7388, 161.9978,	//Ba
		0.733, 0.258, 2.05, 1.96, 23.0, 11.8, -152, 14.4, 134, 14.9,						//Ba+2
		0.8405, 0.2791, 2.3863, 2.1410, 4.6139, 10.3400, 4.1514, 41.9148, 4.7949, 132.0204,	//La
		0.493, 0.167, 1.10, 1.11, 1.50, 3.11, 2.70, 9.61, 1.08, 21.2,						//La+3
		0.8551, 0.2805, 2.3915, 2.1200, 4.5772, 10.1808, 5.0278, 42.0633, 4.5118, 130.9893,	//Ce
		0.560, 0.190, 1.35, 1.30, 1.59, 3.93, 2.63, 10.7, 0.706, 23.8,						//Ce+3
		0.483, 0.165, 1.09, 1.10, 1.34, 3.02, 2.45, 8.85, 0.797, 18.8,						//Ce+4
		0.9096, 0.2939, 2.5313, 2.2471, 4.5266, 10.8266, 4.6376, 48.8842, 4.3690, 147.6020,	//Pr
		0.663, 0.226, 1.73, 1.61, 2.35, 6.33, 0.351, 11.0, 1.59, 16.9,						//Pr+3
		0.512, 0.177, 1.19, 1.17, 1.33, 3.28, 2.36, 8.94, 0.690, 19.3,						//Pr+4
		0.8807, 0.2802, 2.4183, 2.0836, 4.4448, 10.0357, 4.6858, 47.4506, 4.1725, 146.9976,	//Nd
		0.501, 0.162, 1.18, 1.08, 1.45, 3.06, 2.53, 8.80, 0.920, 19.6,						//Nd+3
		0.9471, 0.2977, 2.5463, 2.2276, 4.3523, 10.5762, 4.4789, 49.3619, 3.9080, 145.3580,	//Pm
		0.496, 0.156, 1.20, 1.05, 1.47, 3.07, 2.43, 8.56, 0.943, 19.2,						//Pm+3
		0.9699, 0.3003, 2.5837, 2.2447, 4.2778, 10.6487, 4.4575, 50.7994, 3.5985, 146.4176,	//Sm
		0.518, 0.163, 1.24, 1.08, 1.43, 3.11, 2.40, 8.52, 0.781, 19.1,						//Sm+3
		0.8694, 0.2653, 2.2413, 1.8590, 3.9196, 8.3998, 3.9694, 36.7397, 4.5498, 125.7089,	//Eu
		0.613, 0.190, 1.53, 1.27, 1.84, 4.18, 2.46, 10.7, 0.714, 26.2,						//Eu+2
		0.496, 0.152, 1.21, 1.01, 1.45, 2.95, 2.36, 8.18, 0.774, 18.5,						//Eu+3
		0.9673, 0.2909, 2.4702, 2.1014, 4.1148, 9.7067, 4.4972, 43.4270, 3.2099, 125.9474,	//Gd
		0.490, 0.148, 1.19, 0.974, 1.42, 2.81, 2.30, 7.78, 0.795, 17.7,						//Gd+3
		0.9325, 0.2761, 2.3673, 1.9511, 3.8791, 8.9296, 3.9674, 41.5937, 3.7996, 131.0122,	//Tb
		0.503, 0.150, 1.22, 0.982, 1.42, 2.86, 2.24, 7.77, 0.710, 17.7,						//Tb+3
		0.9505, 0.2773, 2.3705, 1.9469, 3.8218, 8.8862, 4.0471, 43.0938, 3.4451, 133.1396,	//Dy
		0.503, 0.148, 1.24, 0.970, 1.44, 2.88, 2.17, 7.73, 0.643, 17.6,						//Dy+3
		0.9248, 0.2660, 2.2428, 1.8183, 3.6182, 7.9655, 3.7910, 33.1129, 3.7912, 101.8139,	//Ho
		0.456, 0.129, 1.17, 0.869, 1.43, 2.61, 2.15, 7.24, 0.692, 16.7,						//Ho+3
		1.0373, 0.2944, 2.4824, 2.0797, 3.6558, 9.4156, 3.8925, 45.8056, 3.0056, 132.7720,	//Er
		0.522, 0.150, 1.28, 0.964, 1.46, 2.93, 2.05, 7.72, 0.508, 17.8,						//Er+3
		1.0075, 0.2816, 2.3787, 1.9486, 3.5440, 8.7162, 3.6932, 41.8420, 3.1759, 125.0320,	//Tm
		0.475, 0.132, 1.20, 0.864, 1.42, 2.60, 2.05, 7.09, 0.584, 16.6,						//Tm+3
		1.0347, 0.2855, 2.3911, 1.9679, 3.4619, 8.7619, 3.6556, 42.3304, 3.0052, 125.6499,	//Yb
		0.508, 0.136, 1.37, 0.922, 1.76, 3.12, 2.23, 8.722, 0.584, 23.7,					//Yb+2
		0.498, 0.138, 1.22, 0.881, 1.39, 2.63, 1.97, 6.99, 0.559, 16.3,						//Yb+3
		0.9927, 0.2701, 2.2436, 1.8073, 3.3554, 7.8112, 3.7813, 34.4849, 3.0994, 103.3526,	//Lu
		0.483, 0.131, 1.21, 0.845, 1.41, 2.57, 1.94, 6.88, 0.522, 16.2,						//Lu+3
		1.0295, 0.2761, 2.2911, 1.8625, 3.4110, 8.0961, 3.9497, 34.2712, 2.4925, 98.5295,	//Hf
		0.522, 0.145, 1.22, 0.896, 1.37, 2.74, 1.68, 6.91, 0.312, 16.1,						//Hf+4
		1.0190, 0.2694, 2.2291, 1.7962, 3.4097, 7.6944, 3.9252, 31.0942, 2.2679, 91.1089,	//Ta
		0.569, 0.161, 1.26, 0.972, 0.979, 2.76, 1.29, 5.40, 0.551, 10.9,					//Ta+5
		0.9853, 0.2569, 2.1167, 1.6745, 3.3570, 7.0098, 3.7981, 26.9234, 2.2798, 81.3910,	//W
		0.181, 0.0118, 0.873, 0.442, 1.18, 1.52, 1.48, 4.35, 0.56, 9.42,					//W+6
		0.9914, 0.2548, 2.0858, 1.6518, 3.4531, 6.8845, 3.8812, 26.7234, 1.8526, 81.7215,	//Re
		0.9813, 0.2487, 2.0322, 1.5973, 3.3665, 6.4737, 3.6235, 23.2817, 1.9741, 70.9254,	//Os
		0.586, 0.155, 1.31, 0.938, 1.63, 3.19, 1.71, 7.84, 0.540, 19.3,						//Os+4
		1.0194, 0.2554, 2.0645, 1.6475, 3.4425, 6.5966, 3.4914, 23.2269, 1.6976, 70.0272,	//Ir
		0.692, 0.182, 1.37, 1.04, 1.80, 3.47, 1.97, 8.51, 0.804, 21.2,						//Ir+3
		0.653, 0.174, 1.29, 0.992, 1.50, 3.14, 1.74, 7.22, 0.683, 17.2,						//Ir+4
		0.9148, 0.2263, 1.8096, 1.3813, 3.2134, 5.3243, 3.2953, 17.5987, 1.5754, 60.0171,	//Pt
		0.872, 0.223, 1.68, 1.35, 2.63, 4.99, 1.93, 13.6, 0.475, 33.0,						//Pt+2
		0.550, 0.142, 1.21, 0.833, 1.62, 2.81, 1.95, 7.21, 0.610, 17.7,						//Pt+4
		0.9674, 0.2358, 1.8916, 1.4712, 3.3993, 5.6758, 3.0524, 18.7119, 1.2607, 61.5286,	//Au
		0.811, 0.201, 1.57, 1.18, 2.63, 4.25, 2.68, 12.1, 0.998, 34.4,						//Au+1
		0.722, 0.184, 1.39, 1.06, 1.94, 3.58, 1.94, 8.56, 0.699, 20.4,						//Au+3
		1.0033, 0.2413, 1.9469, 1.5298, 3.4396, 5.8009, 3.1548, 19.4520, 1.4180, 60.5753,	//Hg
		0.796, 0.194, 1.56, 1.14, 2.72, 4.21, 2.76, 12.4, 1.18, 36.2,						//Hg+1
		0.773, 0.191, 1.49, 1.12, 2.45, 4.00, 2.23, 1.08, 0.570, 27.6,						//Hg+2
		1.0689, 0.2540, 2.1038, 1.6715, 3.6039, 6.3509, 3.4927, 23.1531, 1.8283, 78.7099,	//Tl
		0.820, 0.197, 1.57, 1.16, 2.78, 4.23, 2.82, 12.7, 1.31, 35.7,						//Tl+1
		0.836, 0.208, 1.43, 1.20, 0.394, 2.57, 2.51, 4.86, 1.50, 13.5,						//Tl+3
		1.0891, 0.2552, 2.1867, 1.7174, 3.6160, 6.5131, 3.8031, 23.9170, 1.8994, 74.7039,	//Pb
		0.755, 0.181, 1.44, 1.05, 2.48, 3.75, 2.45, 10.6, 1.03, 27.9,						//Pb+2
		0.583, 0.144, 1.14, 0.796, 1.60, 2.58, 2.06, 6.22, 0.662, 14.8,						//Pb+4
		1.1007, 0.2546, 2.2306, 1.7351, 3.5689, 6.4948, 4.1549, 23.6464, 2.0382, 70.3780,	//Bi
		0.708, 0.170, 1.35, 0.981, 2.28, 3.44, 2.18, 9.41, 0.797, 23.7,						//Bi+3
		0.654, 0.162, 1.18, 0.905, 1.25, 2.68, 1.66, 5.14, 0.778, 11.2,						//Bi+5
		1.1568, 0.2648, 2.4353, 1.8786, 3.6459, 7.1749, 4.4064, 25.1766, 1.7179, 69.2821,	//Po
		1.0909, 0.2466, 2.1976, 1.6707, 3.3831, 6.0197, 4.6700, 207657, 2.1277, 57.2663,	//At
		1.0756, 0.2402, 2.1630, 1.6169, 3.3178, 5.7644, 4.8852, 19.4568, 2.0489, 52.5009,	//Rn
		1.4282, 0.3183, 3.5081, 2.6889, 5.6767, 13.4816, 4.1964, 54.3866, 3.8946, 200.8321,	//Fr
		1.3127, 0.2887, 3.1243, 2.2897, 5.2988, 10.8276, 5.3891, 43.5389, 5.4133, 145.6109,	//Ra
		0.911, 0.204, 1.65, 1.26, 2.53, 1.03, 3.62, 12.6, 1.58, 30.0,						//Ra+2
		1.3128, 0.2861, 3.1021, 2.2509, 5.3385, 10.5287, 5.9611, 41.7796, 4.7562, 128.2973,	//Ac
		0.915, 0.205, 1.64, 1.28, 2.26, 3.92, 3.18, 11.3, 1.25, 25.1,						//Ac+3
		1.2553, 0.2701, 2.9178, 2.0636, 5.0862, 9.3051, 6.1206, 34.5977, 4.7122, 107.9200,	//Th
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 														//Th+4
		1.3218, 0.2827, 3.1444, 2.2250, 5.4371, 10.2454, 5.6444, 41.1162, 4.0107, 124.4449,	//Pa
		1.3382, 0.2838, 3.2043, 2.2452, 5.4558, 10.2519, 5.4839, 41.7251, 3.6342, 124.9023,	//U
		1.14, 0.250, 2.48, 1.84, 3.61, 7.39, 1.13, 18.0, 0.900, 22.7,						//U+3
		1.09, 0.243, 2.32, 1.75, 12.0, 7.79, -9.11, 8.31, 2.15, 16.5,						//U+4
		0.687, 0.154, 1.14, 0.861, 1.83, 2.58, 2.53, 7.70, 0.957, 15.9,						//U+6
		1.5193, 0.3213, 4.0053, 2.8206, 6.5327, 14.8878, -0.1402, 68.9103, 6.7489, 81.7257,	//Np
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Np+6
		1.3517, 0.2813, 3.2937, 2.2418, 5.3212, 9.9952, 4.6466, 42.7939, 3.5714, 132.1739,	//Pu
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+4
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,														//Pu+6
		1.2135, 0.2483, 2.7962, 1.8437, 4.7545, 7.5421, 4.5731, 29.3841, 4.4786, 112.4579,	//Am
		1.2937, 0.2638, 3.1100, 2.0341, 5.0393, 8.7101, 4.7546, 35.2992, 3.5031, 109.4972,	//Cm
		1.2915, 0.2611, 3.1023, 2.0023, 4.9309, 8.4377, 4.6009, 34.1559, 3.4661, 105.8911,	//Bk
		1.2089, 0.2421, 2.7391, 1.7487, 4.3482, 6.7262, 4.0047, 23.2153, 4.6497, 80.3108,	//Cf
		// Atomic groups:
		0.1796, 73.76, 0.8554, 5.399, 1.75, 27.15, 0.05001, 0.1116, 0.2037, 1.062,			//CH
		0.1575, 89.04, 0.8528, 4.637, 2.359, 30.92, 0.00496, -0.344, 0.1935, 0.6172,		// CH2
		0.4245, 4.092, 0.4256, 4.094, 0.2008, 74.32, 2.884, 33.65, 0.16, 0.4189,			// CH3
		0.1568, 64.9, 0.222, 1.017, 0.8391, 4.656, 1.469, 23.17, 0.05579, 0.11,				// NH
		1.991, 25.94, 0.2351, 74.54, 0.8575, 3.893, 5.336, 0.3422, -5.147, 0.3388,			// NH2
		-0.1646, 168.7, 0.2896, 147.3, 0.838, 3.546, 0.1736, 0.4059, 2.668, 29.57,			// NH3
		0.1597, 53.82, 0.2445, 0.7846, 0.8406, 4.042, 1.235, 20.92, 0.03234, -0.01414,		// OH
		-78.51, 9.013, 80.62, 9.014, 0.6401, 1.924, 2.665, 37.71, 0.2755, 0.2941,			// SH
		// Ions that had no x-ray form factors:
		0.0421, 0.0609, 0.210, 0.559, 0.852, 2.96, 1.82, 11.5, 1.17, 37.7,					// O-2
		0.132, 0.109, 0.292, 0.695, 0.703, 2.39, 0.692, 5.65, 0.0959, 14.7					//Cr+4
		;

#pragma endregion
}

F_TYPE electronDebyeCalTester::electronAtomicFF(F_TYPE q, int elem) {
	//This code is not being used in the DC
	F_TYPE res = 0.0;
	F_TYPE sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

#pragma unroll 5
	for (int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	return res;
}

bool electronDebyeCalTester::GetHasAnomalousScattering()
{
	return pdb->getHasAnomalous();
}

electronDebyeCalTester::~electronDebyeCalTester()
{
	if (pdb) delete pdb;
	if (_electronAff_calculator) delete _electronAff_calculator;
}


double electronAnomDebyeCalTester::Calculate(double q, int nLayers, VectorXd& p)
{
	//////////////////////////////////////////////////////////////////////////
	// The regular atoms
	int pdbLen = pdb->sortedX.size();
	F_TYPE res = 0.0;
	F_TYPE fq = (F_TYPE)q;

	int atmI = -1;
	int prevIon = 255;

	// Calculate the atomic form factors for this q
	Eigen::ArrayXcf atmAmps;
	F_TYPE aff = 0.0;
	const F_TYPE fq10 = q / (10.0);
	atmAmps.resize(pdbLen);
	_electronAff_calculator->GetAllAFFs((float2*)atmAmps.data(), q);
	for (int i = 0; i < pdbLen; i++) {
		// 		if (prevIon != pdb->sortedIonInd[i]) {
		// 			aff = atomicFF(fq10, pdb->sortedIonInd[i]);
		// 			prevIon = pdb->sortedIonInd[i];
		// 			if (p[0] != 0.)	// Subtract dummy atom solvent ED
		// 			{
		// 				F_TYPE rad = pdb->rad->at(pdb->sortedAtmInd[i]);
		// #ifdef USE_FRASER
		// 				aff -= 5.5683279968317084528/*4.1887902047863909846 /*4\pi/3*/ * rad * rad * rad * exp(-(rad * rad * (q*q) / 4.)) * p(0);
		// #else
		// 				aff -= 4.1887902047863909846 * rad * rad * rad * exp(-(0.20678349696647 * sq(rad * q))) * p(0);
		// #endif
		// 			}
		// 		}

		atmAmps(i) += pdb->sortedAnomfPrimes[i];

		if (p(1) > 0.1) { // Debye-Waller
			atmAmps(i) *= exp(-(pdb->sortedBFactor[i] * fq10 * fq10 / (16. * M_PI * M_PI)));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Calculate the intensity
	double intensity = 0.;
	// Sum the Debye contributions
#pragma omp parallel for if(pdbLen >= 500000) reduction(+:intensity) schedule(dynamic, pdbLen / 500)
	for (int i = 1; i < pdbLen; i++) {
		intensity += 2.0 * (
			(atmAmps(i).real() * atmAmps.head(i).real() + atmAmps(i).imag() * atmAmps.head(i).imag()).array().cast<F_TYPE>() *
			SincArrayXf(
				(float(fq) * (sortedLocations.leftCols(i).colwise() - sortedLocations.col(i)).colwise().norm()).array()
			).cast<F_TYPE>()
			).sum();
	} // for i

	intensity += atmAmps.abs2().sum();

	return intensity;
}

VectorXd electronAnomDebyeCalTester::CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress /*= NULL*/, void* progressArgs /*= NULL*/)
{
	return VectorXd();
	// IF THIS COMMENT IS UNCOMMENTED, MUST REPLACE GETPROCADDRESS CALLS WITH DIRECT FUNCTION CALLS
	// 	if (!g_gpuModule) {
	// 		load_gpu_backend(g_gpuModule);
	// 		if (g_gpuModule) {
	// 			if (sizeof(F_TYPE) == sizeof(double)) {
	// 				gpuCalcAnomDebye = (GPUCalculateDebye_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeD");
	// 				gpuCalcAnomDebyeV2 = (GPUCalculateAnomDebyeV2_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeDv2");
	// 				thisSucksAnom = (GPUCalcDebyeV4MAPS_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalcDebyeV4MAPSD");
	// 			}
	// 			else if (sizeof(F_TYPE) == sizeof(float)) {
	// 				gpuCalcAnomDebye = (GPUCalculateDebye_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeF");
	// 				gpuCalcAnomDebyeV2 = (GPUCalculateAnomDebyeV2_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalculateAnomDebyeFv2");
	// 				thisSucksAnom = (GPUCalcAnomDebyeV4MAPS_t)GetProcAddress((HMODULE)g_gpuModule, "GPUCalcAnomDebyeV4MAPSF");
	// 			}
	// 		} // if g_gpuModule
	// 	}
	// 
	// 	if (g_gpuModule != NULL && (gpuCalcDebye == NULL || gpuCalcDebyeV2 == NULL || thisSucks == NULL)) {
	// 		printf("Error loading GPU functions\n");
	// 		return VectorXd();
	// 	}
	// 	std::vector<F_TYPE> ftq(q.size());
	// 	for (int i = 0; i < ftq.size(); i++)
	// 		ftq[i] = (F_TYPE)(q[i]);
	// 	// LOC
	// 	std::vector<F_TYPE> loc(this->pdb->sortedX.size() * 3);
	// 
	// 	int pos = 0;
	// 	for (int i = 0; i < pdb->sortedX.size(); i++) {
	// 		loc[pos++] = pdb->sortedX[i];
	// 		loc[pos++] = pdb->sortedY[i];
	// 		loc[pos++] = pdb->sortedZ[i];
	// 	}
	// 	Eigen::Matrix<F_TYPE, -1, 1, 0, -1, 1> res;
	// 	res.resize(ftq.size());
	// 
	// 	std::vector<float> rads(pdb->rad->size());
	// 	for (int i = 0; i < pdb->rad->size(); i++) {
	// 		rads[i] = pdb->rad->at(i);
	// 	}
	// 
	// 	std::vector<float> fAtmFFcoefs(pdb->sortedCoeffs.begin(), pdb->sortedCoeffs.end());
	// 
	// 	bool bSol = (p.size() > 0 && p(0) > 0.);
	// 	bool bDW = (p.size() > 1 && p(1) > 0.1);
	// 	int gpuRes;
	// 
	// 	switch (kernelVersion)
	// 	{
	// 	case 1:
	// 		if (gpuCalcDebye == NULL)
	// 		{
	// 
	// 		}
	// 		gpuRes = gpuCalcDebye(ftq.size(), &ftq[0], &res(0), &loc[0], (u8*)&pdb->sortedIonInd[0],
	// 			pdb->sortedIonInd.size(), &atmFFcoefs(0, 0), atmFFcoefs.size(), bSol,
	// 			NULL, NULL, 0.0,
	// 			progFunc, progArgs, 0.0, 1.0, pStop);
	// 		break;
	// 	case 2:
	// 		gpuRes = gpuCalcDebyeV2(ftq.size(), q.front(), q.back(), res.data(),
	// 			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
	// 			(float4 *)pdb->atomLocs.data(),
	// 			(u8*)pdb->sortedCoeffIonInd.data(), bDW, pdb->sortedBFactor.data(),
	// 			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
	// 			rads.data(), (float)(p[0]), progress, progressArgs, 0.0, 1.0, pStop);
	// 		break;
	// 
	// 	case 4:
	// 		gpuRes = thisSucks(ftq.size(), q.front(), q.back(), res.data(),
	// 			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
	// 			(float4 *)pdb->atomLocs.data(),
	// 			bDW, pdb->sortedBFactor.data(),
	// 			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
	// 			rads.data(), (float)(p[0]), progress, progressArgs, 0.0, 1.0, pStop);
	// 		break;
	// 	default:
	//
	//		break;
	// 	}
	// 
	// 	return res.cast<double>();
}

void electronAnomDebyeCalTester::PreCalculate(VectorXd& p, int nLayers)
{
	electronDebyeCalTester::PreCalculate(p, nLayers);
}

std::complex<F_TYPE> electronAnomDebyeCalTester::anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime)
{
	return electronDebyeCalTester::electronAtomicFF(q, elem) + fPrime + std::complex<F_TYPE>(0, 1) * fPrimePrime;
}

bool electronPDBAmplitude::GetHasAnomalousScattering()
{
	return pdb.getHasAnomalous();
}

#pragma endregion	// CPDB Reader class
