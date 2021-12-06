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
				* exp(-(0.20678349696647 * sq((*pdb.rad)[pdb.atmInd[i]] * q)));
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

	for(int i = 0; i < 4; i++)
		res += (atmFFcoefs(elem, 2*i)) * exp(-atmFFcoefs(elem, (2*i) + 1) * sqq);
	res += (atmFFcoefs(elem, 8));
	return res;
}

/* This part is supposed to show the changes that will have to be made for electron diffraction (at least the mathematical part of it):

// Not sure what is better or easier but, we can either change both the x-ray and elec to be a 5-Gaussian (as shown underneath) and change the atomic form factors so that for x-ray the 10th col is 0, or instead build two separate options for calculations that will be chosen depending on what the user wants.


FACC PDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

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
	atmFFcoefs << 0.0349, 0.5347, 0.1201, 3.5867, 0.1970, 12.3471, 0.057, 18.9525, 0.1195, 38.6269 //H
		0.0317, 0.2507, 0.0838, 1.4751, 0.1526, 4.4938, 0.1334, 12.6646, 0.0164, 31.1653 //He
		0.0750, 0.3864, 0.2249, 2.9383, 0.5548, 15.3829, 1.4954, 53.5545, 0.9354, 138.7337 //Li
		0.0780, 0.3131, 0.2210, 2.2381, 0.6740, 10.1517, 1.3867, 30.9061, 0.6925, 78.3273 //Be
		0.0909, 0.2995, 0.2551, 2.1155, 0.7738, 8.3816, 1.2136, 24.1292, 0.4606, 63.1314 //B		
		0.0893, 0.2465, 0.2563, 1.7100, 0.5770, 6.4094, 1.0487, 18.6113, 0.3575, 50.2523 //C
		0.1022, 0.2451, 0.3219, 1.7481, 0.7982, 6.1925, 0.8197, 17.3894, 0.1715 ,48.1431 //N
		0.0974, 0.2067, 0.2921, 1.3815, 0.6910, 4.643, 0.6990, 12.7105, 0.2039, 32.4726 //O
		;

*/

/* This part is supposed to show the changes that will have to be made for neutron diffraction (at least the mathematical part of it):

// Since in this case we do noth have a Gaussian, maybe it is better to build each independently of the other.


FACC PDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

	// Somehow, it would be interesting to enter a weighted sum of all the isotopes i.e. res = sum(p_i * b_i)
	// Also, put a default (default abundance in nature) and a possibility for the user to enter it's own abundance (maybe like with anomalous scattering for x-ray).
	
	res = atmFFcoefs(elem, 1);
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
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 9);
#pragma region Atomic form factor coefficients
	atmFFcoefs << 0.49300,	10.51090,	0.32290,	26.1257,	0.14020,	3.14240,	0.04080,	57.79980,	0.0030,
		0.87340,	9.10370,	0.63090,	3.35680,	0.31120,	22.9276,	0.17800,	0.98210,	0.0064,
		1.12820,	3.95460,	0.75080,	1.05240,	0.61750,	85.3905,	0.46530,	168.26100,	0.0377,
		0.69680,	4.62370,	0.78880,	1.95570,	0.34140,	0.63160,	0.15630,	10.09530,	0.0167,
		1.59190,	43.6427,	1.12780,	1.86230,	0.53910,	103.483,	0.70290,	0.54200,	0.0385,
		6.26030,	0.00270,	0.88490,	0.93130,	0.79930,	2.27580,	0.16470,	5.11460,	-6.1092,
		2.05450,	23.2185,	1.33260,	1.02100,	1.09790,	60.3498,	0.70680,	0.14030,	-0.1932,
		2.31000,	20.8439,	1.02000,	10.2075,	1.58860,	0.56870,	0.86500,	51.65120,	0.2156, // Carbon
		12.2126,	0.00570,	3.13220,	9.89330,	2.01250,	28.9975,	1.16630,	0.58260,	-11.5290,
		3.04850,	13.2771,	2.28680,	5.70110,	1.54630,	0.32390,	0.86700,	32.90890,	0.2508,
		4.19160,	12.8573,	1.63969,	4.17236,	1.52673,	47.0179,	-20.307,	-0.01404,	21.9412,
		3.53920,	10.2825,	2.64120,	4.29440,	1.51700,	0.26150,	1.02430,	26.14760,	0.2776,	// F
		3.63220,	5.27756,	3.51057,	14.7353,	1.26064,	0.442258,	0.940706,	47.34370,	0.653396,
		3.95530,	8.40420,	3.11250,	3.42620,	1.45460,	0.23060,	1.12510,	21.71840,	0.3515,
		4.76260,	3.28500,	3.17360,	8.84220,	1.26740,	0.31360,	1.11280,	129.42400,	0.676,
		3.25650,	2.66710,	3.93620,	6.11530,	1.39980,	0.20010,	1.00320,	14.03900,	0.404,
		5.42040,	2.82750,	2.17350,	79.2611,	1.22690,	0.38080,	2.30730,	7.19370,	0.8584,
		3.49880,	2.16760,	3.83780,	4.75420,	1.32840,	0.18500,	0.84970,	10.14110,	0.4853,
		6.42020,	3.03870,	1.90020,	0.74260,	1.59360,	31.5472,	1.96460,	85.08860,	1.1151,
		4.17448,	1.93816,	3.38760,	4.14553,	1.20296,	0.228753,	0.528137,	8.28524,	0.706786,
		6.29150,	2.43860,	3.03530,	32.3337,	1.98910,	0.67850,	1.54100,	81.69370,	1.1407,
		4.43918,	1.64167,	3.20345,	3.43757,	1.19453,	0.21490,	0.41653,	6.65365,	0.746297,
		6.43450,	1.90670,	4.17910,	27.1570,	1.78000,	0.52600,	1.49080,	68.16450,	1.1149,
		6.29150,	2.43860,	3.03530,	32.3337,	1.98910,	0.67850,	1.54100,	81.69370,	1.1407,
		11.46040,	0.01040,	7.19640,	1.16620,	6.25560,	18.5194,	1.64550,	47.77840,	-9.5574,
		18.29150,	0.00660,	7.40840,	1.17170,	6.53370,	19.5424,	2.33860,	60.44860,	-16.378,
		7.48450,	0.90720,	6.77230,	14.8407,	0.65390,	43.8983,	1.64420,	33.39290,	1.4445,
		8.21860,	12.79490,	7.43980,	0.77480,	1.05190,	213.187,	0.86590,	41.68410,	1.4228,
		7.95780,	12.63310,	7.49170,	0.76740,	6.35900,	-0.0020,	1.19150,	31.91280,	-4.9978,
		8.62660,	10.44210,	7.38730,	0.65990,	1.58990,	85.7484,	1.02110,	178.43700,	1.3751,
		15.63480,	-0.00740,	7.95180,	0.60890,	8.43720,	10.3116,	0.85370,	25.99050,	-14.875,
		9.18900,	9.02130,	7.36790,	0.57290,	1.64090,	136.108,	1.46800,	51.35310,	1.3329,
		13.40080,	0.29854,	8.02730,	7.96290,	1.65943,	-0.28604,	1.57936,	16.06620,	-6.6667,
		9.75950,	7.85080,	7.35580,	0.50000,	1.69910,	35.6338,	1.90210,	116.10500,	1.2807,
		9.11423,	7.52430,	7.62174,	0.457585,	2.27930,	19.5361,	0.087899,	61.65580,	0.897155,
		17.73440,	0.22061,	8.73816,	7.04716,	5.25691,	-0.15762,	1.92134,	15.97680,	-14.652,
		19.51140,	0.178847,	8.23473,	6.67018,	2.01341,	-0.29263,	1.52080,	12.94640,	-13.28,
		10.29710,	6.86570,	7.35110,	0.43850,	2.07030,	26.8938,	2.05710,	102.47800,	1.2199,
		10.10600,	6.88180,	7.35410,	0.44090,	2.28840,	20.3004,	0.02230,	115.12200,	1.2298,
		9.43141,	6.39535,	7.74190,	0.383349,	2.15343,	15.1908,	0.016865,	63.96900,	0.656565,
		15.68870,	0.679003,	8.14208,	5.40135,	2.03081,	9.97278,	-9.57600,	0.940464,	1.7143,
		10.64060,	6.10380,	7.35370,	0.39200,	3.32400,	20.26260,	1.49220,	98.73990,	1.1832,
		9.54034,	5.66078,	7.75090,	0.344261,	3.58274,	13.30750,	0.509107,	32.42240,	0.616898,
		9.68090,	5.59463,	7.81136,	0.334393,	2.87603,	12.82880,	0.113575,	32.87610,	0.518275,
		11.28190,	5.34090,	7.35730,	0.34320,	3.01930,	17.86740,	2.24410,	83.75430,	1.0896,
		10.80610,	5.27960,	7.36200,	0.34350,	3.52680,	14.34300,	0.21840,	41.32350,	1.0874,
		9.84521,	4.91797,	7.87194,	0.294393,	3.56531,	10.81710,	0.323613,	24.12810,	0.393974,
		9.96253,	4.84850,	7.97057,	0.283303,	2.76067,	10.48520,	0.054447,	27.57300,	0.251877,
		11.76950,	4.76110,	7.35730,	0.30720,	3.52220,	15.35350,	2.30450,	76.88050,	1.0369,
		11.04240,	4.65380,	7.37400,	0.30530,	4.13460,	12.05460,	0.43990,	31.28090,	1.0097,
		11.17640,	4.61470,	7.38630,	0.30050,	3.39480,	11.67290,	0.07240,	38.55660,	0.9707,
		12.28410,	4.27910,	7.34090,	0.27840,	4.00340,	13.53590,	2.34880,	71.16920,	1.0118,
		11.22960,	4.12310,	7.38830,	0.27260,	4.73930,	10.24430,	0.71080,	25.64660,	0.9324,
		10.33800,	3.90969,	7.88173,	0.238668,	4.76795,	8.35583,	0.725591,	18.34910,	0.286667,
		12.83760,	3.87850,	7.29200,	0.25650,	4.44380,	12.17630,	2.38000,	66.34210,	1.0341,
		11.41660,	3.67660,	7.40050,	0.24490,	5.34420,	8.87300,	0.97730,	22.16260,	0.8614,
		10.78060,	3.54770,	7.75868,	0.22314,	5.22746,	7.64468,	0.847114,	16.96730,	0.386044,
		13.33800,	3.58280,	7.16760,	0.24700,	5.61580,	11.39660,	1.67350,	64.81260,	1.191,
		11.94750,	3.36690,	7.35730,	0.22740,	6.24550,	8.66250,	1.55780,	25.84870,	0.89,
		11.81680,	3.37484,	7.11181,	.244078,	5.78135,	7.98760,	1.14523,	19.89700,	1.14431,
		14.07430,	3.26550,	7.03180,	0.23330,	5.16520,	10.31630,	2.41000,	58.70970,	1.3041,
		11.97190,	2.99460,	7.38620,	0.20310,	6.46680,	7.08260,	1.39400,	18.09950,	0.7807,
		15.23540,	3.06690,	6.70060,	0.24120,	4.35910,	10.78050,	2.96230,	61.41350,	1.7189,
		12.69200,	2.81262,	6.69883,	0.22789,	6.06692,	6.36441,	1.00660,	14.41220,	1.53545,
		16.08160,	2.85090,	6.37470,	0.25160,	3.70680,	11.44680,	3.68300,	54.76250,	2.1313,
		12.91720,	2.53718,	6.70003,	0.205855,	6.06791,	5.47913,	0.859041,	11.60300,	1.45572,
		16.67230,	2.63450,	6.07010,	0.26470,	3.43130,	12.94790,	4.27790,	47.79720,	2.531,
		17.00600,	2.40980,	5.81960,	0.27260,	3.97310,	15.23720,	4.35436,	43.81630,	2.8409,
		17.17890,	2.17230,	5.23580,	16.57960,	5.63770,	0.26090,	3.98510,	41.43280,	2.9557,
		17.17180,	2.20590,	6.33380,	19.33450,	5.57540,	0.28710,	3.72720,	58.15350,	3.1776,
		17.35550,	1.93840,	6.72860,	16.56230,	5.54930,	0.22610,	3.53750,	39.39720,	2.825,
		17.17840,	1.78880,	9.64350,	17.31510,	5.13990,	0.27480,	1.52920,	164.93400,	3.4873,
		17.58160,	1.71390,	7.65980,	14.79570,	5.89810,	0.16030,	2.78170,	31.20870,	2.0782,
		17.56630,	1.55640,	9.81840,	14.09880,	5.42200,	0.16640,	2.66940,	132.37600,	2.5064,
		18.08740,	1.49070,	8.13730,	12.69630,	2.56540,	24.56510,	-34.19300,	-0.01380,	41.4025,
		17.77600,	1.40290,	10.29460,	12.80060,	5.72629,	0.125599,	3.26588,	104.35400,	1.91213,
		17.92680,	1.35417,	9.15310,	11.21450,	1.76795,	22.65990,	-33.10800,	-0.01319,	40.2602,
		17.87650,	1.27618,	10.94800,	11.91600,	5.41732,	0.117622,	3.65721,	87.66270,	2.06929,
		18.16680,	1.21480,	10.05620,	10.14830,	1.01118,	21.60540,	-2.64790,	-0.10276,	9.41454,
		17.61420,	1.18865,	12.01440,	11.76600,	4.04183,	0.204785,	3.53346,	69.79570,	3.75591,
		19.88120,	0.019175,	18.06530,	1.13305,	11.01770,	10.16210,	1.94715,	28.33890,	-12.912,
		17.91630,	1.12446,	13.34170,	0.028781,	10.79900,	9.28206,	0.337905,	25.72280,	-6.3934,
		3.70250,	0.27720,	17.23560,	1.09580,	12.88760,	11.00400,	3.74290,	61.65840,	4.3875,
		21.16640,	0.014734,	18.20170,	1.03031,	11.74230,	9.53659,	2.30951,	26.63070,	-14.421,
		21.01490,	0.014345,	18.09920,	1.02238,	11.46320,	8.78809,	0.740625,	23.34520,	-14.316,
		17.88710,	1.03649,	11.17500,	8.48061,	6.57891,	0.058881,	0.00000,	0.00000,	0.344941,
		19.13010,	0.864132,	11.09480,	8.14487,	4.64901,	21.57070,	2.71263,	86.84720,	5.40428,
		19.26740,	0.80852,	12.91820,	8.43467,	4.86337,	24.79970,	1.56756,	94.29280,	5.37874,
		18.56380,	0.847329,	13.28850,	8.37164,	9.32602,	0.017662,	3.00964,	22.88700,	-3.1892,
		18.50030,	0.844582,	13.17870,	8.12534,	4.71304,	0.036495,	2.18535,	20.85040,	1.42357,
		19.29570,	0.751536,	14.35010,	8.21758,	4.73425,	25.87490,	1.28918,	98.60620,	5.328,
		18.87850,	0.764252,	14.12590,	7.84438,	3.32515,	21.24870,	-6.19890,	-0.01036,	11.8678,
		18.85450,	0.760825,	13.98060,	7.62436,	2.53464,	19.33170,	-5.65260,	-0.01020,	11.2835,
		19.33190,	0.69866,	15.50170,	7.98939,	5.29537,	25.20520,	0.60584,	76.89860,	5.26593,
		19.17010,	0.696219,	15.20960,	7.55573,	4.32234,	22.50570,	0.00000,	0.00000,	5.2916,
		19.24930,	0.683839,	14.79000,	7.14833,	2.89289,	17.91440,	-7.94920,	0.005127,	13.0174,
		19.28080,	0.64460,	16.68850,	7.47260,	4.80450,	24.66050,	1.04630,	99.81560,	5.179,
		19.18120,	0.646179,	15.97190,	7.19123,	5.27475,	21.73260,	0.357534,	66.11470,	5.21572,
		19.16430,	0.645643,	16.24560,	7.18544,	4.37090,	21.40720,	0.00000,	0.00000,	5.21404,
		19.22140,	0.59460,	17.64440,	6.90890,	4.46100,	24.70080,	1.60290,	87.48250,	5.0694,
		19.15140,	0.597922,	17.25350,	6.80639,	4.47128,	20.25210,	0.00000,	0.00000,	5.11937,
		19.16240,	0.54760,	18.55960,	6.37760,	4.29480,	25.84990,	2.03960,	92.80290,	4.9391,
		19.10450,	0.551522,	18.11080,	6.32470,	3.78897,	17.35950,	0.00000,	0.00000,	4.99635,
		19.18890,	5.83030,	19.10050,	0.50310,	4.45850,	26.89090,	2.46630,	83.95710,	4.7821,
		19.10940,	0.50360,	19.05480,	5.83780,	4.56480,	23.37520,	0.48700,	62.20610,	4.7861,
		18.93330,	5.76400,	19.71310,	0.46550,	3.41820,	14.00490,	0.01930,	-0.75830,	3.9182,
		19.64180,	5.30340,	19.04550,	0.46070,	5.03710,	27.90740,	2.68270,	75.28250,	4.5909,
		18.97550,	0.467196,	18.93300,	5.22126,	5.10789,	19.59020,	0.288753,	55.51130,	4.69626,
		19.86850,	5.44853,	19.03020,	0.467973,	2.41253,	14.12590,	0.00000,	0.00000,	4.69263,
		19.96440,	4.81742,	19.01380,	0.420885,	6.14487,	28.52840,	2.52390,	70.84030,	4.352,
		20.14720,	4.34700,	18.99490,	0.23140,	7.51380,	27.76600,	2.27350,	66.87760,	4.07121,
		20.23320,	4.35790,	18.99700,	0.38150,	7.80690,	29.52590,	2.88680,	84.93040,	4.0714,
		20.29330,	3.92820,	19.02980,	0.34400,	8.97670,	26.46590,	1.99000,	64.26580,	3.7118,
		20.38920,	3.56900,	19.10620,	0.31070,	10.66200,	24.38790,	1.49530,	213.90400,	3.3352,
		20.35240,	3.55200,	19.12780,	0.30860,	10.28210,	23.71280,	0.96150,	59.45650,	3.2791,
		20.33610,	3.21600,	19.29700,	0.27560,	10.88800,	20.20730,	2.69590,	167.20200,	2.7731,
		20.18070,	3.21367,	19.11360,	0.28331,	10.90540,	20.05580,	0.77634,	51.74600,	3.02902,
		20.57800,	2.94817,	19.59900,	0.244475,	11.37270,	18.77260,	3.28719,	133.12400,	2.14678,
		20.24890,	2.92070,	19.37630,	0.250698,	11.63230,	17.82110,	0.336048,	54.94530,	2.4086,
		21.16710,	2.81219,	19.76950,	0.226836,	11.85130,	17.60830,	3.33049,	127.11300,	1.86264,
		20.80360,	2.77691,	19.55900,	0.23154,	11.93690,	16.54080,	0.612376,	43.16920,	2.09013,
		20.32350,	2.65941,	19.81860,	0.21885,	12.12330,	15.79920,	0.144583,	62.23550,	1.5918,
		22.04400,	2.77393,	19.66970,	0.222087,	12.38560,	16.76690,	2.82428,	143.64400,	2.0583,
		21.37270,	2.64520,	19.74910,	0.214299,	12.13290,	15.32300,	0.97518,	36.40650,	1.77132,
		20.94130,	2.54467,	20.05390,	0.202481,	12.46680,	14.81370,	0.296689,	45.46430,	1.24285,
		22.68450,	2.66248,	19.68470,	0.210628,	12.77400,	15.88500,	2.85137,	137.90300,	1.98486,
		21.96100,	2.52722,	19.93390,	0.199237,	12.12000,	14.17830,	1.51031,	30.87170,	1.47588,
		23.34050,	2.56270,	19.60950,	0.202088,	13.12350,	15.10090,	2.87516,	132.72100,	2.02876,
		22.55270,	2.41740,	20.11080,	0.185769,	12.06710,	13.12750,	2.07492,	27.44910,	1.19499,
		24.00420,	2.47274,	19.42580,	0.19651,	13.43960,	14.39960,	2.89604,	128.00700,	2.20963,
		23.15040,	2.31641,	20.25990,	.174081,	11.92020,	12.15710,	2.71488,	24.82420,	.954586,
		24.62740,	2.38790,	19.08860,	0.19420,	13.76030,	17.75460,	2.92270,	123.17400,	2.5745,
		24.00630,	2.27783,	19.95040,	0.17353,	11.80340,	11.60960,	3.87243,	26.51560,	1.36389,
		23.74970,	2.22258,	20.37450,	0.16394,	11.85090,	11.31100,	3.26503,	22.99660,	0.759344,
		25.07090,	2.25341,	19.07980,	0.181951,	13.85180,	12.93310,	3.54545,	101.39800,	2.4196,
		24.34660,	2.15530,	20.42080,	0.15552,	11.87080,	10.57820,	3.71490,	21.70290,	0.64509,
		25.89760,	2.24256,	18.21850,	0.196143,	14.31670,	12.66480,	2.95354,	115.36200,	3.58324,
		24.95590,	2.05601,	20.32710,	0.149525,	12.24710,	10.04990,	3.77300,	21.27730,	0.691967,
		26.50700,	2.18020,	17.63830,	0.202172,	14.55960,	12.18990,	2.96577,	111.87400,	4.29728,
		25.53950,	1.98040,	20.28610,	0.143384,	11.98120,	9.34972,	4.50073,	19.58100,	0.68969,
		26.90490,	2.07051,	17.29400,	0.19794,	14.55830,	11.44070,	3.63837,	92.65660,	4.56796,
		26.12960,	1.91072,	20.09940,	0.139358,	11.97880,	8.80018,	4.93676,	18.59080,	0.852795,
		27.65630,	2.07356,	16.42850,	0.223545,	14.97790,	11.36040,	2.98233,	105.70300,	5.92046,
		26.72200,	1.84659,	19.77480,	0.13729,	12.15060,	8.36225,	5.17379,	17.89740,	1.17613,
		28.18190,	2.02859,	15.88510,	0.238849,	15.15420,	10.99750,	2.98706,	102.96100,	6.75621,
		27.30830,	1.78711,	19.33200,	0.136974,	12.33390,	7.96778,	5.38348,	17.29220,	1.63929,
		28.66410,	1.98890,	15.43450,	0.257119,	15.30870,	10.66470,	2.98963,	100.41700,	7.56672,
		28.12090,	1.78503,	17.68170,	0.15997,	13.33350,	8.18304,	5.14657,	20.39000,	3.70983,
		27.89170,	1.73272,	18.76140,	0.13879,	12.60720,	7.64412,	5.47647,	16.81530,	2.26001,
		28.94760,	1.90182,	15.22080,	9.98519,	15.10000,	0.261033,	3.71601,	84.32980,	7.97628,
		28.46280,	1.68216,	18.12100,	0.142292,	12.84290,	7.33727,	5.59415,	16.35350,	2.97573,
		29.14400,	1.83262,	15.17260,	9.59990,	14.75860,	0.275116,	4.30013,	72.02900,	8.58154,
		28.81310,	1.59136,	18.46010,	0.128903,	12.72850,	6.76232,	5.59927,	14.03660,	2.39699,
		29.20240,	1.77333,	15.22930,	9.37046,	14.51350,	0.295977,	4.76492,	63.36440,	9.24354,
		29.15870,	1.50711,	18.84070,	0.116741,	12.82680,	6.31524,	5.38695,	12.42440,	1.78555,
		29.08180,	1.72029,	15.43000,	9.22590,	14.43270,	0.321703,	5.11982,	57.05600,	9.8875,
		29.49360,	1.42755,	19.37630,	0.104621,	13.05440,	5.93667,	5.06412,	11.19720,	1.01074,
		28.76210,	1.67191,	15.71890,	9.09227,	14.55640,	0.35050,	5.44174,	52.08610,	10.472,
		28.18940,	1.62903,	16.15500,	8.97948,	14.93050,	0.38266,	5.67589,	48.16470,	11.0005,
		30.41900,	1.37113,	15.26370,	6.84706,	14.74580,	0.165191,	5.06795,	18.00300,	6.49804,
		27.30490,	1.59279,	16.72960,	8.86553,	15.61150,	0.41792,	5.83377,	45.00110,	11.4722,
		30.41560,	1.34323,	15.86200,	7.10909,	13.61450,	0.204633,	5.82008,	20.32540,	8.27903,
		30.70580,	1.30923,	15.55120,	6.71983,	14.23260,	0.167252,	5.53672,	17.49110,	6.96824,
		27.00590,	1.51293,	17.76390,	8.81174,	15.71310,	.424593,	5.78370,	38.61030,	11.6883,
		29.84290,	1.32927,	16.72240,	7.38979,	13.21530,	0.263297,	6.35234,	22.94260,	9.85329,
		30.96120,	1.24813,	15.98290,	6.60834,	13.73480,	0.16864,	5.92034,	16.93920,	7.39534,
		16.88190,	0.46110,	18.59130,	8.62160,	25.55820,	1.48260,	5.86000,	36.39560,	12.0658,
		28.01090,	1.35321,	17.82040,	7.73950,	14.33590,	0.356752,	6.58077,	26.40430,	11.2299,
		30.68860,	1.21990,	16.90290,	6.82872,	12.78010,	0.212867,	6.52354,	18.65900,	9.0968,
		20.68090,	0.54500,	19.04170,	8.44840,	21.65750,	1.57290,	5.96760,	38.32460,	12.6089,
		25.08530,	1.39507,	18.49730,	7.65105,	16.88830,	0.443378,	6.48216,	28.22620,	12.0205,
		29.56410,	1.21152,	18.06000,	7.05639,	12.83740,	.284738,	6.89912,	20.74820,	10.6268,
		27.54460,	0.65515,	19.15840,	8.70751,	15.53800,	1.96347,	5.52593,	45.81490,	13.1746,
		21.39850,	1.47110,	20.47230,	0.517394,	18.74780,	7.43463,	6.82847,	28.84820,	12.5258,
		30.86950,	1.10080,	18.38410,	6.53852,	11.93280,	0.219074,	7.00574,	17.21140,	9.8027,
		31.06170,	0.69020,	13.06370,	2.35760,	18.44200,	8.61800,	5.96960,	47.25790,	13.4118,
		21.78860,	1.33660,	19.56820,	0.48838,	19.14060,	6.77270,	7.01107,	23.81320,	12.4734,
		32.12440,	1.00566,	18.80030,	6.10926,	12.01750,	0.147041,	6.96886,	14.71400,	8.08428,
		33.36890,	0.70400,	12.95100,	2.92380,	16.58770,	8.79370,	6.46920,	48.00930,	13.5782,
		21.80530,	1.23560,	19.50260,	6.24149,	19.10530,	0.469999,	7.10295,	20.31850,	12.4711,
		33.53640,	0.91654,	25.09460,	0.039042,	19.24970,	5.71414,	6.91555,	12.82850,	-6.7994,
		34.67260,	0.700999,	15.47330,	3.55078,	13.11380,	9.55642,	7.02588,	47.00450,	13.677,
		35.31630,	0.68587,	19.02110,	3.97458,	9.49887,	11.38240,	7.42518,	45.47150,	13.7108,
		35.56310,	0.66310,	21.28160,	4.06910,	8.00370,	14.04220,	7.44330,	44.24730,	13.6905,
		35.92990,	0.646453,	23.05470,	4.17619,	12.14390,	23.10520,	2.11253,	150.64500,	13.7247,
		35.76300,	0.616341,	22.90640,	3.87135,	12.47390,	19.98870,	3.21097,	142.32500,	13.6211,
		35.21500,	0.604909,	21.67000,	3.57670,	7.91342,	12.60100,	7.65078,	29.84360,	13.5431,
		35.65970,	0.589092,	23.10320,	3.65155,	12.59770,	18.59900,	4.08655,	117.02000,	13.5266,
		35.17360,	0.579689,	22.11120,	3.41437,	8.19216,	12.91870,	7.05545,	25.94430,	13.4637,
		35.56450,	0.563359,	23.42190,	3.46204,	12.74730,	17.83090,	4.80703,	99.17220,	13.4314,
		35.10070,	0.555054,	22.44180,	3.24498,	9.78554,	13.46610,	5.29444,	23.95330,	13.376,
		35.88470,	0.547751,	23.29480,	3.41519,	14.18910,	16.92350,	4.17287,	105.25100,	13.4287,
		36.02280,	0.52930,	23.41280,	3.32530,	14.94910,	16.09270,	4.18800,	100.61300,	13.3966,
		35.57470,	0.52048,	22.52590,	3.12293,	12.21650,	12.71480,	5.37073,	26.33940,	13.3092,
		35.37150,	0.516598,	22.53260,	3.05053,	12.02910,	12.57230,	4.79840,	23.45820,	13.2671,
		34.85090,	0.507079,	22.75840,	2.89030,	14.00990,	13.17670,	1.21457,	25.20170,	13.1665,
		36.18740,	0.511929,	23.59640,	3.25396,	15.64020,	15.36220,	4.18550,	97.49080,	13.3573,
		35.70740,	0.502322,	22.61300,	3.03807,	12.98980,	12.14490,	5.43227,	25.49280,	13.2544,
		35.51030,	0.498626,	22.57870,	2.96627,	12.77660,	11.94840,	4.92159,	22.75020,	13.2116,
		35.01360,	0.48981,	22.72860,	2.81099,	14.38840,	12.33000,	1.75669,	22.65810,	13.113,
		36.52540,	0.499384,	23.80830,	3.26371,	16.77070,	14.94550,	3.47947,	105.9800,	13.3812,
		35.84000,	0.484938,	22.71690,	2.96118,	13.58070,	11.53310,	5.66016,	24.39920,	13.1991,
		35.64930,	0.481422,	22.64600,	2.89020,	13.35950,	11.31600,	5.18831,	21.83010,	13.1555,
		35.17360,	0.473204,	22.71810,	2.73848,	14.76350,	11.55300,	2.28678,	20.93030,	13.0582,
		36.67060,	0.483629,	24.09920,	3.20647,	17.34150,	14.31360,	3.49331,	102.2730,	13.3592,
		36.64880,	0.465154,	24.40960,	3.08997,	17.39900,	13.43460,	4.21665,	88.48340,	13.2887,
		36.78810,	0.451018,	24.77360,	3.04619,	17.89190,	12.89460,	4.23284,	86.00300,	13.2754,
		36.91850,	0.437533,	25.19950,	3.00775,	18.33170,	12.40440,	4.24391,	83.78810,	13.2674,
		//////////////////////////////////////////////////////////////////////////
		// Modified atomic form factors
		0.894937, 55.7145, 0.894429, 4.03158, 3.78824, 24.8323, 3.14683e-6, 956.628, 1.42149,	// CH
		1.61908, 52.1451, 2.27205, 24.6589, 2.1815, 24.6587, 0.0019254, 152.165, 1.92445,		// CH2
		12.5735, 38.7341, -0.456658, -6.28167, 5.71547, 54.955, -11.711, 47.898, 2.87762,		// CH3
		0.00506991, 108.256, 2.03147, 14.6199, 1.82122, 14.628, 2.06506, 35.4102, 2.07168,		// NH
		3.00872, 28.3717, 0.288137, 63.9637, 3.39248, 3.51866, 2.03511, 28.3675, 0.269952,		// NH2
		0.294613, 67.4408, 6.48379, 29.1576, 5.67182, 0.54735, 6.57164, 0.547493, -9.02757,		// NH3
		-2.73406, 22.1288, 0.00966263, 94.3428, 6.64439, 13.9044, 2.67949, 32.7607, 2.39981,	// OH
		-127.811, 7.19935, 62.5514, 12.1591, 160.747, 1.88979, 2.34822, 55.952, -80.836			// SH
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
