#define AMP_EXPORTER

#include "../backend_version.h"

#include "PDBAmplitude.h"
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
	double qmax, unsigned short sections, double* outData,
	double* loc, u8* ionInd,
	int numAtoms, double* coeffs, int numCoeffs, bool bSolOnly,
	u8* atmInd, float* rad, double solvED, u8 solventType,	// FOR DUMMY ATOM SOLVENT
	double* solCOM, u64* solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	double* outSolCOM, u64* outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	progressFunc progfunc, void* progargs, double progmin, double progmax, int* pStop);

GPUCalculatePDB_t gpuCalcPDB = NULL;

typedef double IN_JF_TYPE;
typedef int (*GPUCalculatePDBJ_t)(u64 voxels, int thDivs, int phDivs, IN_JF_TYPE stepSize, double* outData, float* locX, float* locY, float* locZ,
	u8* ionInd, int numAtoms, float* coeffs, int numCoeffs, bool bSolOnly,
	u8* atmInd, float* atmRad, IN_JF_TYPE solvED, u8 solventType,// FOR DUMMY ATOM SOLVENT
	float4* solCOM, int4* solDims, u64 solDimLen, double voxStep,	// For voxel based solvent
	float4* outSolCOM, int4* outSolDims, u64 outSolDimLen, double outerSolED,	// For outer solvent layer
	double scale, progressFunc progfunc, void* progargs, float progmin, float progmax, int* pStop);
GPUCalculatePDBJ_t gpuCalcPDBJ = NULL;

typedef bool (*GPUDirectSetPDB_t)(Workspace& work, const float4* atomLocs,
	const unsigned char* ionInd,
	size_t numAtoms, const float* coeffs,
	int* atomsPerIon,
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

PDBAmplitude::~PDBAmplitude() 
{
	delete pdb;
}

std::complex<FACC> PDBAmplitude::calcAmplitude(FACC qx, FACC qy, FACC qz) {
	FACC q = sqrt(qx*qx + qy*qy + qz*qz), resI = 0.0, resR = 0.0, aff = 0.0;
	Eigen::Matrix<float, 3, 1> qVec(qx, qy, qz);

	// Atomic form factors and dummy solvent
	if(bitwiseCalculationFlags & (CALC_ATOMIC_FORMFACTORS | CALC_DUMMY_SOLVENT) ) 
	{
		double phase = 0.0;
		int xSz = int(pdb->x.size());

		Eigen::Array<float, -1, 1> phases;
		phases = (atomLocs * qVec).array();
		
		// TODO: Think of a better way to reduce the branch, code size and copy pasta
		if (pdb->haveAnomalousAtoms)
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
		(this->pdb->atmRadType == RAD_CALC || this->pdb->atmRadType == RAD_EMP || 
		this->pdb->atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb->atmRadType == RAD_VDW)
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

	std::vector<float4> atomLocations(pdb->sortedX.size());
	for (int i = 0; i < pdb->sortedX.size(); i++)
	{
		atomLocations[i].x = pdb->sortedX[i];
		atomLocations[i].y = pdb->sortedY[i];
		atomLocations[i].z = pdb->sortedZ[i];
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
	int vecSize = pdb->ionInd.size();

	int gpuRes = gpuCalcPDBJ(voxels, grid->GetDimY(1) - 1, grid->GetDimZ(1,1), grid->GetStepSize(),
		grid->GetDataPointer(), &pdb->sortedX[0], &pdb->sortedY[0], &pdb->sortedZ[0], &pdb->sortedIonInd[0], vecSize, 
		pdb->sortedCoeffs.data(), pdb->sortedCoeffs.size() / 9, pdb->bOnlySolvent, &pdb->sortedAtmInd[0], &(pdb->rad->at(0)),
		solventED, int(pdb->GetRadiusType()),
		&solCOM[0], &solDims[0], ssz, voxelStep,
		&outSolCOM[0], &outSolDims[0], osz, outerSolventED, this->scale,
		progFunc, progArgs, progMin, progMax, pStop);
		((JacobianSphereGrid*)(grid))->CalculateSplines();
#else

	
	int gpuRes = gpuCalcPDB(voxels, dimx, qmax, sections,
		grid->GetPointerWithIndices(), &loc[0], &sortedIonInd[0],
		vecSize, newCoeffs.data(), newCoeffs.size(), pdb->bOnlySolvent,
		&(sortedAtmInd[0]), &(pdb->rad->at(0)), solventED, int(pdb->GetRadiusType()),
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
	if(!pdb->bOnlySolvent) {
		double phase = 0.0;
		int xSz = (int)pdb->x.size();
		for(int i = 0; i < xSz; i++) {
			phase = qx * pdb->x[i] + qy * pdb->y[i] + qz * pdb->z[i];
			aff = atomicFF(q/(10.0), pdb->ionInd[i]);

			resI += aff * sin(phase);
			resR += aff * cos(phase);
		}
	} // if bOnlySolvent

	// Subtract the solvent using voxels
	if(this->bSolventLoaded && this->solventBoxDims.size() > 0 &&
		(this->pdb->atmRadType == RAD_CALC || this->pdb->atmRadType == RAD_EMP || 
		this->pdb->atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb->atmRadType == RAD_VDW)
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
	if (this->pdb->atmRadType == RAD_DUMMY_ATOMS_ONLY && solventED != 0.0) {
		double solR = 0.0, solI = 0.0, phase, gi;
		int xSz = pdb->x.size();
		for(int i = 0; i < xSz; i++) {
			phase = qx * pdb->x[i] + qy * pdb->y[i] + qz * pdb->z[i];
#ifdef USE_FRASER
			gi = /*4.1887902047863909846*/ /*4\pi/3*/ 5.56832799683 /*pi^1.5*/ * (*this->pdb->rad)[pdb->atmInd[i]] * (*this->pdb->rad)[pdb->atmInd[i]] *
				(*this->pdb->rad)[pdb->atmInd[i]];
			gi *= exp(-sq((*this->pdb->rad)[pdb->atmInd[i]] * q / 2.0));
#else
			gi = 4.1887902047863909846 * (*pdb->rad)[pdb->atmInd[i]] * (*pdb->rad)[pdb->atmInd[i]] * (*pdb->rad)[pdb->atmInd[i]]
				* exp(-(0.20678349696647 * sq((*pdb->rad)[pdb->atmInd[i]] * q))); // 0.206... = (4pi/3)^(2/3) / (4pi)
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

	if (this->pdb->getBOnlySolvent() && solventED != 0.0) {
		return -std::complex<FACC>(resR, resI);
	}
	return std::complex<FACC>(resR, resI);

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



FACC PDBAmplitude::GetVoxelStepSize() {
	return voxelStep;
}

void PDBAmplitude::SetVoxelStepSize(FACC vS) {
	voxelStep = vS;
}

void PDBAmplitude::SetOutputSlices(bool bOutput, std::string outPath /*= ""*/) {
	this->pdb->bOutputSlices = bOutput;
	if(bOutput) {
		this->pdb->slicesBasePathSt = outPath;
	}
}

void PDBAmplitude::SetSolventOnlyCalculation(bool bOnlySol) {
	this->pdb->bOnlySolvent = bOnlySol;
}

void PDBAmplitude::SetFillHoles(bool bFillHole) {
	this->pdb->bFillHoles = bFillHole;

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
	if(this->pdb->x.size() < 1) {
		gridStatus = AMP_UNINITIALIZED;

		this->status = NO_ATOMS_IN_FILE;
		throw backend_exception(ERROR_INVALIDPARAMTREE, "There are no atoms in the PDB file.");
	}

	FACC dist = std::max(solventRad, solvationThickness) + 2.0 * voxelStep;

	this->xMin = this->pdb->x[0] - (*pdb->rad)[this->pdb->atmInd[0]];
	this->xMax = this->pdb->x[0] + (*pdb->rad)[this->pdb->atmInd[0]];
	this->yMin = this->pdb->y[0] - (*pdb->rad)[this->pdb->atmInd[0]];
	this->yMax = this->pdb->y[0] + (*pdb->rad)[this->pdb->atmInd[0]];
	this->zMin = this->pdb->z[0] - (*pdb->rad)[this->pdb->atmInd[0]];
	this->zMax = this->pdb->z[0] + (*pdb->rad)[this->pdb->atmInd[0]];

	for(unsigned int i = 1; i < pdb->x.size(); i++) {
		this->xMin = std::min(float(xMin), this->pdb->x[i] - (*pdb->rad)[this->pdb->atmInd[i]]);
		this->xMax = std::max(float(xMax), this->pdb->x[i] + (*pdb->rad)[this->pdb->atmInd[i]]);
		this->yMin = std::min(float(yMin), this->pdb->y[i] - (*pdb->rad)[this->pdb->atmInd[i]]);
		this->yMax = std::max(float(yMax), this->pdb->y[i] + (*pdb->rad)[this->pdb->atmInd[i]]);
		this->zMin = std::min(float(zMin), this->pdb->z[i] - (*pdb->rad)[this->pdb->atmInd[i]]);
		this->zMax = std::max(float(zMax), this->pdb->z[i] + (*pdb->rad)[this->pdb->atmInd[i]]);
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
	writer.String(pdb->fn.c_str());

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
		writer.Bool(pdb->bOnlySolvent);



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
		switch (pdb->atmRadType) {
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
		writer.Bool(pdb->bFillHoles);
	}

	writer.Key("Implicit");
	writer.StartObject();
	writer.Key("# Atoms treated as having implicit hydrogen");
	writer.Int(pdb->number_of_implicit_atoms);
	writer.Key("# Total atoms");
	writer.Int(pdb->atom.size());
	writer.Key("# implicit amino acids");
	writer.Int(pdb->number_of_implicit_amino_acids);
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

	ss << "PDB file: " << pdb->fn << "\n";
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
		if(pdb->bOnlySolvent) {
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
		switch(pdb->atmRadType) {
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

		ss << "Fill holes: " << (pdb->bFillHoles ? "true" : "false") << "\n";
		header.append(ampers + ss.str());
		ss.str("");
	}

	ss << pdb->number_of_implicit_atoms << " of " << pdb->atom.size() << " atoms are treated as having implicit hydrogen atoms (" << pdb->number_of_implicit_amino_acids << " amino acids).\n";
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
	pdb->SetRadiusType( IntToAtom_Radius_Type( int(p(9) + 0.1) ) );

	atomLocs.resize(pdb->sortedX.size(), 3);

	atomLocs.col(0) = Eigen::Map<Eigen::ArrayXf>(pdb->sortedX.data(), pdb->sortedX.size());
	atomLocs.col(1) = Eigen::Map<Eigen::ArrayXf>(pdb->sortedY.data(), pdb->sortedX.size());
	atomLocs.col(2) = Eigen::Map<Eigen::ArrayXf>(pdb->sortedZ.data(), pdb->sortedX.size());

	int numUIons = 1;
	int prevIon = pdb->sortedIonInd[0];
	int prevInd = 0;
	std::vector<int> uniIonInds, numIonsPerInd;
	std::vector<float> uniIonRads;
	for (int i = 1; i < pdb->sortedIonInd.size(); i++)
	{
		if (prevIon != pdb->sortedIonInd[i])
		{
			uniIonInds.push_back(prevIon);
			uniIonRads.push_back(pdb->rad->at(pdb->sortedAtmInd[prevInd]));
			numIonsPerInd.push_back(i - prevInd);
			prevInd = i;
			prevIon = pdb->sortedIonInd[i];
			numUIons++;
		}
	}
	uniIonInds.push_back(prevIon);
	uniIonRads.push_back(pdb->rad->at(pdb->sortedAtmInd[prevInd]));
	numIonsPerInd.push_back(pdb->sortedIonInd.size() - prevInd);


	uniqueIonsIndices = Eigen::Map<Eigen::ArrayXi>(uniIonInds.data(), numUIons);
	numberOfIonsPerIndex = Eigen::Map<Eigen::ArrayXi>(numIonsPerInd.data(), numUIons);
	uniqueIonRads = Eigen::Map<Eigen::ArrayXf>(uniIonRads.data(), numUIons);

	bitwiseCalculationFlags = 0;

	if (solventED != 0.0 &&
		(this->pdb->atmRadType == RAD_CALC || this->pdb->atmRadType == RAD_EMP ||
		this->pdb->atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb->atmRadType == RAD_VDW)
		)
		bitwiseCalculationFlags |= CALC_VOXELIZED_SOLVENT;
	if (outerSolventED != 0.0 &&
		(this->pdb->atmRadType == RAD_CALC || this->pdb->atmRadType == RAD_EMP ||
		this->pdb->atmRadType == RAD_DUMMY_ATOMS_RADII || this->pdb->atmRadType == RAD_VDW ||
		pdb->atmRadType == RAD_DUMMY_ATOMS_ONLY)
		)
		bitwiseCalculationFlags |= CALC_VOXELIZED_OUTER_SOLVENT;

	if (!pdb->bOnlySolvent) bitwiseCalculationFlags |= CALC_ATOMIC_FORMFACTORS;
	if ((RAD_DUMMY_ATOMS_ONLY == pdb->GetRadiusType())
		&& solventED != 0.0
		) bitwiseCalculationFlags |= CALC_DUMMY_SOLVENT;

	if (pdb->haveAnomalousAtoms) bitwiseCalculationFlags |= CALC_ANOMALOUS;

	affCalculator.Initialize(bitwiseCalculationFlags, pdb->sortedIonInd.size(), numUIons, pdb->sortedCoeffs.data(), numberOfIonsPerIndex.data(), isElectron());

	if (bitwiseCalculationFlags & CALC_DUMMY_SOLVENT)
		affCalculator.SetSolventED(solventED, c1, uniIonRads.data() , pdb->bOnlySolvent);
	if (bitwiseCalculationFlags & CALC_ANOMALOUS)
	{
		std::vector<float2> anomfPrimesAsFloat2;

		size_t sz = pdb->sortedAnomfPrimes.size();
		anomfPrimesAsFloat2.resize(sz);
 		Eigen::Map<Eigen::ArrayXf>((float*)anomfPrimesAsFloat2.data(), 2 * sz) =
 			(Eigen::Map<Eigen::ArrayXf>((float*)pdb->sortedAnomfPrimes.data(), 2 * sz)).cast<float>();

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
					pdb->GetRadiusType() != RAD_UNINITIALIZED && pdb->GetRadiusType() != RAD_DUMMY_ATOMS_ONLY
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
		( (solventED != 0.0 && (pdb->GetRadiusType() == RAD_CALC || pdb->GetRadiusType() == RAD_EMP ||
		pdb->GetRadiusType() == RAD_DUMMY_ATOMS_RADII || pdb->GetRadiusType() == RAD_VDW) )
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

	if(RAD_UNINITIALIZED == pdb->GetRadiusType()) {
		// This will load the default (vdw) as a way to make the rad pointer not problematic
		pdb->SetRadiusType(RAD_UNINITIALIZED);
	}
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

bool PDBAmplitude::SetModel( GridWorkspace& workspace ) {
	if (!g_useGPUAndAvailable)
		return false;

	workspace.bSolOnly = pdb->bOnlySolvent;
	workspace.scale = scale;

	std::vector<float4> solCOM, outSolCOM;
	std::vector<int4> solDims, outSolDims;

	PrepareParametersForGPU(solCOM, solDims, outSolCOM, outSolDims);

	workspace.solventED = float(solventED);
	workspace.solventType = int(pdb->GetRadiusType());
	workspace.atomsPerIon = pdb->atomsPerIon.data();
	workspace.numUniqueIons = pdb->atomsPerIon.size();
	
	int comb = (workspace.bSolOnly ? 0 : CALC_ATOMIC_FORMFACTORS);
	comb |= ((solventED != 0.0 && int(pdb->GetRadiusType()) == 4) ? CALC_DUMMY_SOLVENT : 0);
	workspace.kernelComb = comb;

	std::vector<float> affs;
	affs.resize(workspace.qLayers * workspace.numUniqueIons);
	affCalculator.GetQMajorAFFMatrix(affs.data(), workspace.qLayers, workspace.stepSize);

	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return GPUHybrid_SetPDB(workspace, *reinterpret_cast<std::vector<float4>*>(&pdb->atomLocs),
		affs,
		/*pdb->sortedCoeffIonInd,
		pdb->sortedCoeffs, pdb->atomsPerIon,
		int(pdb->GetRadiusType()), pdb->sortedAtmInd, *(pdb->rad),
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
	int nAtoms = pdb->x.size();
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
		sprintf(&occTemp[0], "%6f", pdb->occupancy[pp]);
		sprintf(&occTemp[6], "%6f", pdb->BFactor[pp]);
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
			pdb->pdbAtomSerNo[pp].substr(0, 5).c_str(), pdb->pdbAtomName[pp].substr(0, 4).c_str(),
			pdb->pdbResName[pp].substr(0, 3).c_str(), pdb->pdbChain[pp],
			pdb->pdbResNo[pp].substr(0, 4).c_str(),
			posPlaceHolder.substr(0, 24).c_str(),//pdb->x[pp]*10., pdb->y[pp]*10., pdb->z[pp]*10.,
			occTemp.substr(0,12).c_str(),
			pdb->pdbSegID[pp].substr(0, 4).c_str(), pdb->atom[pp].substr(0, 4).c_str());
		Eigen::Vector3f loc(pdb->x[pp], pdb->y[pp], pdb->z[pp]);
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
	pdb_hash += pdb->fn;
	pdb_hash += pdb->anomalousfn;

	//sortedX, sortedY, sortedZ;
	for (const auto& c : pdb->sortedX)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb->sortedY)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb->sortedZ)
		pdb_hash += std::to_string(c);
	//sortedAtmInd, sortedIonInd
	for (const auto& c : pdb->sortedAtmInd)
		pdb_hash += std::to_string(c);
	for (const auto& c : pdb->sortedIonInd)
		pdb_hash += std::to_string(c);

	// anomIndex, sortedAnomfPrimes
	for (const auto& c : pdb->anomIndex)
		pdb_hash += std::to_string(c.first) + std::to_string(c.second.first) + std::to_string(c.second.second);
	for (const auto& c : pdb->sortedAnomfPrimes)
		pdb_hash += std::to_string(c.real()) + std::to_string(c.imag());

	pdb_hash = md5(pdb_hash);

}
void DebyeCalTester::SetStop(int* stop) {
	pStop = stop;
}

void DebyeCalTester::OrganizeParameters(const VectorXd& p, int nLayers) {
	return;	// For now (during testing), there will be no parameters
	//throw std::exception("The method or operation is not implemented.");
}

void DebyeCalTester::PreCalculate(VectorXd& p, int nLayers) {
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

	if (_aff_calculator)
		delete _aff_calculator;
	_aff_calculator = new atomicFFCalculator(comb, num, numUnIons, fAtmFFcoefs.data(), pdb->atomsPerIon.data());

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

		_aff_calculator->SetSolventED(p(0), p(2), ionRads.data());
	}
}



VectorXd DebyeCalTester::CalculateVector(
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

VectorXd DebyeCalTester::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

double DebyeCalTester::Calculate(double q, int nLayers, VectorXd& p /*= VectorXd( ) */) {
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
	_aff_calculator->GetAllAFFs(atmAmps.data(), q);

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

VectorXd DebyeCalTester::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p /*= VectorXd( ) */) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

PDB_READER_ERRS DebyeCalTester::LoadPDBFile(string filename, int model /*= 0*/) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

void DebyeCalTester::GetHeader(unsigned int depth, JsonWriter& writer)
{
	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");
}

void DebyeCalTester::GetHeader(unsigned int depth, std::string& header) {

	throw backend_exception(ERROR_UNSUPPORTED, "The method or operation is not implemented.");

}

// typedef int(*GPUCalculateDebye_t)(int qValues, F_TYPE *qVals, F_TYPE *outData,
// 	F_TYPE *loc, u8 *ionInd, int numAtoms, F_TYPE *coeffs, int numCoeffs, bool bSol,
// 	u8 * atmInd, F_TYPE *rad, double solvED,
// 	progressFunc progfunc, void *progargs, double progmin, double progmax, int *pStop);
// 
// GPUCalculateDebye_t gpuCalcDebye = NULL;

typedef int(*GPUCalculateDebyeV2_t)(int numQValues, float qMin, float qMax,
	F_TYPE* outData,
	int numAtoms, const int* atomsPerIon,
	float4* loc, u8* ionInd, float2* anomalousVals,
	bool bBfactors, float* BFactors,
	float* coeffs, bool bSol,
	bool bSolOnly, char* atmInd, float* atmRad, float solvED, float c1,
	progressFunc progfunc, void* progargs,
	double progmin, double progmax, int* pStop);
GPUCalculateDebyeV2_t gpuCalcDebyeV2 = NULL;

//GPUCalculateDebyeV2_t gpuCalcDebyeV3MAPS = NULL;

typedef int(*GPUCalcDebyeV4MAPS_t)(
	int numQValues, float qMin, float qMax, F_TYPE* outData, int numAtoms,
	const int* atomsPerIon, float4* atomLocations, float2* anomFactors, bool bBfactors, float* BFactors,
	float* coeffs, bool bSol, bool bSolOnly, char* atmInd, float* atmRad, float solvED, float c1,
	progressFunc progfunc, void* progargs, double progmin, double progmax, int* pStop);
GPUCalcDebyeV4MAPS_t thisSucks = NULL;

VectorXd DebyeCalTester::CalculateVectorGPU(
	const std::vector<double>& q,
	int nLayers,
	VectorXd& p /*= VectorXd( ) */,
	progressFunc progress /*= NULL*/,
	void* progressArgs /*= NULL*/)
{

	if (g_useGPUAndAvailable) {
		if (sizeof(F_TYPE) == sizeof(double))
		{
			gpuCalcDebyeV2 = (GPUCalculateDebyeV2_t)GPUCalculateDebyeDv2;
			thisSucks = (GPUCalcDebyeV4MAPS_t)GPUCalcDebyeV4MAPSD;
		}
	}

	if (g_useGPUAndAvailable && (/*gpuCalcDebye == NULL || */gpuCalcDebyeV2 == NULL || thisSucks == NULL)) {
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
		gpuRes = gpuCalcDebyeV2(ftq.size(), q.front(), q.back(), res.data(),
			pdb->sortedIonInd.size(), pdb->atomsPerIon.data(),
			(float4*)pdb->atomLocs.data(),
			(u8*)pdb->sortedCoeffIonInd.data(),
			pdb->haveAnomalousAtoms ? anomfPrimesAsFloat2.data() : NULL,
			bDW, pdb->sortedBFactor.data(),
			fAtmFFcoefs.data(), bSol, false, (char*)pdb->sortedAtmInd.data(),
			rads.data(), (float)(p[0]), c1, progress, progressArgs, 0.0, 1.0, pStop);
		break;

	case 4:
		gpuRes = thisSucks(ftq.size(), q.front(), q.back(), res.data(),
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


void DebyeCalTester::xrayInitialize() {
	pdb = nullptr;
	_aff_calculator = nullptr;

	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 9);
#pragma region Atomic form factor coefficients
	atmFFcoefs << 0.49300, 10.51090, 0.32290, 26.1257, 0.14020, 3.14240, 0.04080, 57.79980, 0.0030,	// H
		0.87340, 9.10370, 0.63090, 3.35680, 0.31120, 22.9276, 0.17800, 0.98210, 0.0064,	// He
		1.12820, 3.95460, 0.75080, 1.05240, 0.61750, 85.3905, 0.46530, 168.26100, 0.0377,	// Li
		0.69680, 4.62370, 0.78880, 1.95570, 0.34140, 0.63160, 0.15630, 10.09530, 0.0167,	// Li+1
		1.59190, 43.6427, 1.12780, 1.86230, 0.53910, 103.483, 0.70290, 0.54200, 0.0385,	// Be
		6.26030, 0.00270, 0.88490, 0.93130, 0.79930, 2.27580, 0.16470, 5.11460, -6.1092,	// Be+2
		2.05450, 23.2185, 1.33260, 1.02100, 1.09790, 60.3498, 0.70680, 0.14030, -0.1932,	// B
		2.31000, 20.8439, 1.02000, 10.2075, 1.58860, 0.56870, 0.86500, 51.65120, 0.2156, // Carbon
		12.2126, 0.00570, 3.13220, 9.89330, 2.01250, 28.9975, 1.16630, 0.58260, -11.5290,	// N
		3.04850, 13.2771, 2.28680, 5.70110, 1.54630, 0.32390, 0.86700, 32.90890, 0.2508,	// O
		4.19160, 12.8573, 1.63969, 4.17236, 1.52673, 47.0179, -20.307, -0.01404, 21.9412,	// O-1
		3.53920, 10.2825, 2.64120, 4.29440, 1.51700, 0.26150, 1.02430, 26.14760, 0.2776,	// F
		3.63220, 5.27756, 3.51057, 14.7353, 1.26064, 0.442258, 0.940706, 47.34370, 0.653396,	// F-1
		3.95530, 8.40420, 3.11250, 3.42620, 1.45460, 0.23060, 1.12510, 21.71840, 0.3515,	// Ne
		4.76260, 3.28500, 3.17360, 8.84220, 1.26740, 0.31360, 1.11280, 129.42400, 0.676,	// Na
		3.25650, 2.66710, 3.93620, 6.11530, 1.39980, 0.20010, 1.00320, 14.03900, 0.404,	// Na+1
		5.42040, 2.82750, 2.17350, 79.2611, 1.22690, 0.38080, 2.30730, 7.19370, 0.8584,	// Mg
		3.49880, 2.16760, 3.83780, 4.75420, 1.32840, 0.18500, 0.84970, 10.14110, 0.4853,	// Mg+2
		6.42020, 3.03870, 1.90020, 0.74260, 1.59360, 31.5472, 1.96460, 85.08860, 1.1151,	// Al
		4.17448, 1.93816, 3.38760, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786,	// Al+3
		6.29150, 2.43860, 3.03530, 32.3337, 1.98910, 0.67850, 1.54100, 81.69370, 1.1407,	// Si_v
		4.43918, 1.64167, 3.20345, 3.43757, 1.19453, 0.21490, 0.41653, 6.65365, 0.746297,	// Si+4
		6.43450, 1.90670, 4.17910, 27.1570, 1.78000, 0.52600, 1.49080, 68.16450, 1.1149,	// P
		6.90530, 1.46790, 5.20340, 22.2151, 1.43790, 0.25360, 1.58630, 56.1720, 0.86690,	// S
		11.46040, 0.01040, 7.19640, 1.16620, 6.25560, 18.5194, 1.64550, 47.77840, -9.5574,	// Cl
		18.29150, 0.00660, 7.40840, 1.17170, 6.53370, 19.5424, 2.33860, 60.44860, -16.378,	// Cl-1
		7.48450, 0.90720, 6.77230, 14.8407, 0.65390, 43.8983, 1.64420, 33.39290, 1.4445,
		8.21860, 12.79490, 7.43980, 0.77480, 1.05190, 213.187, 0.86590, 41.68410, 1.4228,
		7.95780, 12.63310, 7.49170, 0.76740, 6.35900, -0.0020, 1.19150, 31.91280, -4.9978,
		8.62660, 10.44210, 7.38730, 0.65990, 1.58990, 85.7484, 1.02110, 178.43700, 1.3751,
		15.63480, -0.00740, 7.95180, 0.60890, 8.43720, 10.3116, 0.85370, 25.99050, -14.875,
		9.18900, 9.02130, 7.36790, 0.57290, 1.64090, 136.108, 1.46800, 51.35310, 1.3329,
		13.40080, 0.29854, 8.02730, 7.96290, 1.65943, -0.28604, 1.57936, 16.06620, -6.6667,
		9.75950, 7.85080, 7.35580, 0.50000, 1.69910, 35.6338, 1.90210, 116.10500, 1.2807,
		9.11423, 7.52430, 7.62174, 0.457585, 2.27930, 19.5361, 0.087899, 61.65580, 0.897155,
		17.73440, 0.22061, 8.73816, 7.04716, 5.25691, -0.15762, 1.92134, 15.97680, -14.652,
		19.51140, 0.178847, 8.23473, 6.67018, 2.01341, -0.29263, 1.52080, 12.94640, -13.28,
		10.29710, 6.86570, 7.35110, 0.43850, 2.07030, 26.8938, 2.05710, 102.47800, 1.2199,
		10.10600, 6.88180, 7.35410, 0.44090, 2.28840, 20.3004, 0.02230, 115.12200, 1.2298,
		9.43141, 6.39535, 7.74190, 0.383349, 2.15343, 15.1908, 0.016865, 63.96900, 0.656565,
		15.68870, 0.679003, 8.14208, 5.40135, 2.03081, 9.97278, -9.57600, 0.940464, 1.7143,
		10.64060, 6.10380, 7.35370, 0.39200, 3.32400, 20.26260, 1.49220, 98.73990, 1.1832,
		9.54034, 5.66078, 7.75090, 0.344261, 3.58274, 13.30750, 0.509107, 32.42240, 0.616898,
		9.68090, 5.59463, 7.81136, 0.334393, 2.87603, 12.82880, 0.113575, 32.87610, 0.518275,
		11.28190, 5.34090, 7.35730, 0.34320, 3.01930, 17.86740, 2.24410, 83.75430, 1.0896,
		10.80610, 5.27960, 7.36200, 0.34350, 3.52680, 14.34300, 0.21840, 41.32350, 1.0874,
		9.84521, 4.91797, 7.87194, 0.294393, 3.56531, 10.81710, 0.323613, 24.12810, 0.393974,
		9.96253, 4.84850, 7.97057, 0.283303, 2.76067, 10.48520, 0.054447, 27.57300, 0.251877,
		11.76950, 4.76110, 7.35730, 0.30720, 3.52220, 15.35350, 2.30450, 76.88050, 1.0369,
		11.04240, 4.65380, 7.37400, 0.30530, 4.13460, 12.05460, 0.43990, 31.28090, 1.0097,
		11.17640, 4.61470, 7.38630, 0.30050, 3.39480, 11.67290, 0.07240, 38.55660, 0.9707,
		12.28410, 4.27910, 7.34090, 0.27840, 4.00340, 13.53590, 2.34880, 71.16920, 1.0118,
		11.22960, 4.12310, 7.38830, 0.27260, 4.73930, 10.24430, 0.71080, 25.64660, 0.9324,
		10.33800, 3.90969, 7.88173, 0.238668, 4.76795, 8.35583, 0.725591, 18.34910, 0.286667,
		12.83760, 3.87850, 7.29200, 0.25650, 4.44380, 12.17630, 2.38000, 66.34210, 1.0341,
		11.41660, 3.67660, 7.40050, 0.24490, 5.34420, 8.87300, 0.97730, 22.16260, 0.8614,
		10.78060, 3.54770, 7.75868, 0.22314, 5.22746, 7.64468, 0.847114, 16.96730, 0.386044,
		13.33800, 3.58280, 7.16760, 0.24700, 5.61580, 11.39660, 1.67350, 64.81260, 1.191,
		11.94750, 3.36690, 7.35730, 0.22740, 6.24550, 8.66250, 1.55780, 25.84870, 0.89,
		11.81680, 3.37484, 7.11181, .244078, 5.78135, 7.98760, 1.14523, 19.89700, 1.14431,
		14.07430, 3.26550, 7.03180, 0.23330, 5.16520, 10.31630, 2.41000, 58.70970, 1.3041,
		11.97190, 2.99460, 7.38620, 0.20310, 6.46680, 7.08260, 1.39400, 18.09950, 0.7807,
		15.23540, 3.06690, 6.70060, 0.24120, 4.35910, 10.78050, 2.96230, 61.41350, 1.7189,
		12.69200, 2.81262, 6.69883, 0.22789, 6.06692, 6.36441, 1.00660, 14.41220, 1.53545,
		16.08160, 2.85090, 6.37470, 0.25160, 3.70680, 11.44680, 3.68300, 54.76250, 2.1313,
		12.91720, 2.53718, 6.70003, 0.205855, 6.06791, 5.47913, 0.859041, 11.60300, 1.45572,
		16.67230, 2.63450, 6.07010, 0.26470, 3.43130, 12.94790, 4.27790, 47.79720, 2.531,
		17.00600, 2.40980, 5.81960, 0.27260, 3.97310, 15.23720, 4.35436, 43.81630, 2.8409,
		17.17890, 2.17230, 5.23580, 16.57960, 5.63770, 0.26090, 3.98510, 41.43280, 2.9557,
		17.17180, 2.20590, 6.33380, 19.33450, 5.57540, 0.28710, 3.72720, 58.15350, 3.1776,
		17.35550, 1.93840, 6.72860, 16.56230, 5.54930, 0.22610, 3.53750, 39.39720, 2.825,
		17.17840, 1.78880, 9.64350, 17.31510, 5.13990, 0.27480, 1.52920, 164.93400, 3.4873,
		17.58160, 1.71390, 7.65980, 14.79570, 5.89810, 0.16030, 2.78170, 31.20870, 2.0782,
		17.56630, 1.55640, 9.81840, 14.09880, 5.42200, 0.16640, 2.66940, 132.37600, 2.5064,
		18.08740, 1.49070, 8.13730, 12.69630, 2.56540, 24.56510, -34.19300, -0.01380, 41.4025,
		17.77600, 1.40290, 10.29460, 12.80060, 5.72629, 0.125599, 3.26588, 104.35400, 1.91213,
		17.92680, 1.35417, 9.15310, 11.21450, 1.76795, 22.65990, -33.10800, -0.01319, 40.2602,
		17.87650, 1.27618, 10.94800, 11.91600, 5.41732, 0.117622, 3.65721, 87.66270, 2.06929,
		18.16680, 1.21480, 10.05620, 10.14830, 1.01118, 21.60540, -2.64790, -0.10276, 9.41454,
		17.61420, 1.18865, 12.01440, 11.76600, 4.04183, 0.204785, 3.53346, 69.79570, 3.75591,
		19.88120, 0.019175, 18.06530, 1.13305, 11.01770, 10.16210, 1.94715, 28.33890, -12.912,
		17.91630, 1.12446, 13.34170, 0.028781, 10.79900, 9.28206, 0.337905, 25.72280, -6.3934,
		3.70250, 0.27720, 17.23560, 1.09580, 12.88760, 11.00400, 3.74290, 61.65840, 4.3875,
		21.16640, 0.014734, 18.20170, 1.03031, 11.74230, 9.53659, 2.30951, 26.63070, -14.421,
		21.01490, 0.014345, 18.09920, 1.02238, 11.46320, 8.78809, 0.740625, 23.34520, -14.316,
		17.88710, 1.03649, 11.17500, 8.48061, 6.57891, 0.058881, 0.00000, 0.00000, 0.344941,
		19.13010, 0.864132, 11.09480, 8.14487, 4.64901, 21.57070, 2.71263, 86.84720, 5.40428,
		19.26740, 0.80852, 12.91820, 8.43467, 4.86337, 24.79970, 1.56756, 94.29280, 5.37874,
		18.56380, 0.847329, 13.28850, 8.37164, 9.32602, 0.017662, 3.00964, 22.88700, -3.1892,
		18.50030, 0.844582, 13.17870, 8.12534, 4.71304, 0.036495, 2.18535, 20.85040, 1.42357,
		19.29570, 0.751536, 14.35010, 8.21758, 4.73425, 25.87490, 1.28918, 98.60620, 5.328,
		18.87850, 0.764252, 14.12590, 7.84438, 3.32515, 21.24870, -6.19890, -0.01036, 11.8678,
		18.85450, 0.760825, 13.98060, 7.62436, 2.53464, 19.33170, -5.65260, -0.01020, 11.2835,
		19.33190, 0.69866, 15.50170, 7.98939, 5.29537, 25.20520, 0.60584, 76.89860, 5.26593,
		19.17010, 0.696219, 15.20960, 7.55573, 4.32234, 22.50570, 0.00000, 0.00000, 5.2916,
		19.24930, 0.683839, 14.79000, 7.14833, 2.89289, 17.91440, -7.94920, 0.005127, 13.0174,
		19.28080, 0.64460, 16.68850, 7.47260, 4.80450, 24.66050, 1.04630, 99.81560, 5.179,
		19.18120, 0.646179, 15.97190, 7.19123, 5.27475, 21.73260, 0.357534, 66.11470, 5.21572,
		19.16430, 0.645643, 16.24560, 7.18544, 4.37090, 21.40720, 0.00000, 0.00000, 5.21404,
		19.22140, 0.59460, 17.64440, 6.90890, 4.46100, 24.70080, 1.60290, 87.48250, 5.0694,
		19.15140, 0.597922, 17.25350, 6.80639, 4.47128, 20.25210, 0.00000, 0.00000, 5.11937,
		19.16240, 0.54760, 18.55960, 6.37760, 4.29480, 25.84990, 2.03960, 92.80290, 4.9391,
		19.10450, 0.551522, 18.11080, 6.32470, 3.78897, 17.35950, 0.00000, 0.00000, 4.99635,
		19.18890, 5.83030, 19.10050, 0.50310, 4.45850, 26.89090, 2.46630, 83.95710, 4.7821,
		19.10940, 0.50360, 19.05480, 5.83780, 4.56480, 23.37520, 0.48700, 62.20610, 4.7861,
		18.93330, 5.76400, 19.71310, 0.46550, 3.41820, 14.00490, 0.01930, -0.75830, 3.9182,
		19.64180, 5.30340, 19.04550, 0.46070, 5.03710, 27.90740, 2.68270, 75.28250, 4.5909,
		18.97550, 0.467196, 18.93300, 5.22126, 5.10789, 19.59020, 0.288753, 55.51130, 4.69626,
		19.86850, 5.44853, 19.03020, 0.467973, 2.41253, 14.12590, 0.00000, 0.00000, 4.69263,
		19.96440, 4.81742, 19.01380, 0.420885, 6.14487, 28.52840, 2.52390, 70.84030, 4.352,
		20.14720, 4.34700, 18.99490, 0.23140, 7.51380, 27.76600, 2.27350, 66.87760, 4.07121,
		20.23320, 4.35790, 18.99700, 0.38150, 7.80690, 29.52590, 2.88680, 84.93040, 4.0714,
		20.29330, 3.92820, 19.02980, 0.34400, 8.97670, 26.46590, 1.99000, 64.26580, 3.7118,
		20.38920, 3.56900, 19.10620, 0.31070, 10.66200, 24.38790, 1.49530, 213.90400, 3.3352,
		20.35240, 3.55200, 19.12780, 0.30860, 10.28210, 23.71280, 0.96150, 59.45650, 3.2791,
		20.33610, 3.21600, 19.29700, 0.27560, 10.88800, 20.20730, 2.69590, 167.20200, 2.7731,
		20.18070, 3.21367, 19.11360, 0.28331, 10.90540, 20.05580, 0.77634, 51.74600, 3.02902,
		20.57800, 2.94817, 19.59900, 0.244475, 11.37270, 18.77260, 3.28719, 133.12400, 2.14678,
		20.24890, 2.92070, 19.37630, 0.250698, 11.63230, 17.82110, 0.336048, 54.94530, 2.4086,
		21.16710, 2.81219, 19.76950, 0.226836, 11.85130, 17.60830, 3.33049, 127.11300, 1.86264,
		20.80360, 2.77691, 19.55900, 0.23154, 11.93690, 16.54080, 0.612376, 43.16920, 2.09013,
		20.32350, 2.65941, 19.81860, 0.21885, 12.12330, 15.79920, 0.144583, 62.23550, 1.5918,
		22.04400, 2.77393, 19.66970, 0.222087, 12.38560, 16.76690, 2.82428, 143.64400, 2.0583,
		21.37270, 2.64520, 19.74910, 0.214299, 12.13290, 15.32300, 0.97518, 36.40650, 1.77132,
		20.94130, 2.54467, 20.05390, 0.202481, 12.46680, 14.81370, 0.296689, 45.46430, 1.24285,
		22.68450, 2.66248, 19.68470, 0.210628, 12.77400, 15.88500, 2.85137, 137.90300, 1.98486,
		21.96100, 2.52722, 19.93390, 0.199237, 12.12000, 14.17830, 1.51031, 30.87170, 1.47588,
		23.34050, 2.56270, 19.60950, 0.202088, 13.12350, 15.10090, 2.87516, 132.72100, 2.02876,
		22.55270, 2.41740, 20.11080, 0.185769, 12.06710, 13.12750, 2.07492, 27.44910, 1.19499,
		24.00420, 2.47274, 19.42580, 0.19651, 13.43960, 14.39960, 2.89604, 128.00700, 2.20963,
		23.15040, 2.31641, 20.25990, .174081, 11.92020, 12.15710, 2.71488, 24.82420, .954586,
		24.62740, 2.38790, 19.08860, 0.19420, 13.76030, 17.75460, 2.92270, 123.17400, 2.5745,
		24.00630, 2.27783, 19.95040, 0.17353, 11.80340, 11.60960, 3.87243, 26.51560, 1.36389,
		23.74970, 2.22258, 20.37450, 0.16394, 11.85090, 11.31100, 3.26503, 22.99660, 0.759344,
		25.07090, 2.25341, 19.07980, 0.181951, 13.85180, 12.93310, 3.54545, 101.39800, 2.4196,
		24.34660, 2.15530, 20.42080, 0.15552, 11.87080, 10.57820, 3.71490, 21.70290, 0.64509,
		25.89760, 2.24256, 18.21850, 0.196143, 14.31670, 12.66480, 2.95354, 115.36200, 3.58324,
		24.95590, 2.05601, 20.32710, 0.149525, 12.24710, 10.04990, 3.77300, 21.27730, 0.691967,
		26.50700, 2.18020, 17.63830, 0.202172, 14.55960, 12.18990, 2.96577, 111.87400, 4.29728,
		25.53950, 1.98040, 20.28610, 0.143384, 11.98120, 9.34972, 4.50073, 19.58100, 0.68969,
		26.90490, 2.07051, 17.29400, 0.19794, 14.55830, 11.44070, 3.63837, 92.65660, 4.56796,
		26.12960, 1.91072, 20.09940, 0.139358, 11.97880, 8.80018, 4.93676, 18.59080, 0.852795,
		27.65630, 2.07356, 16.42850, 0.223545, 14.97790, 11.36040, 2.98233, 105.70300, 5.92046,
		26.72200, 1.84659, 19.77480, 0.13729, 12.15060, 8.36225, 5.17379, 17.89740, 1.17613,
		28.18190, 2.02859, 15.88510, 0.238849, 15.15420, 10.99750, 2.98706, 102.96100, 6.75621,
		27.30830, 1.78711, 19.33200, 0.136974, 12.33390, 7.96778, 5.38348, 17.29220, 1.63929,
		28.66410, 1.98890, 15.43450, 0.257119, 15.30870, 10.66470, 2.98963, 100.41700, 7.56672,
		28.12090, 1.78503, 17.68170, 0.15997, 13.33350, 8.18304, 5.14657, 20.39000, 3.70983,
		27.89170, 1.73272, 18.76140, 0.13879, 12.60720, 7.64412, 5.47647, 16.81530, 2.26001,
		28.94760, 1.90182, 15.22080, 9.98519, 15.10000, 0.261033, 3.71601, 84.32980, 7.97628,
		28.46280, 1.68216, 18.12100, 0.142292, 12.84290, 7.33727, 5.59415, 16.35350, 2.97573,
		29.14400, 1.83262, 15.17260, 9.59990, 14.75860, 0.275116, 4.30013, 72.02900, 8.58154,
		28.81310, 1.59136, 18.46010, 0.128903, 12.72850, 6.76232, 5.59927, 14.03660, 2.39699,
		29.20240, 1.77333, 15.22930, 9.37046, 14.51350, 0.295977, 4.76492, 63.36440, 9.24354,
		29.15870, 1.50711, 18.84070, 0.116741, 12.82680, 6.31524, 5.38695, 12.42440, 1.78555,
		29.08180, 1.72029, 15.43000, 9.22590, 14.43270, 0.321703, 5.11982, 57.05600, 9.8875,
		29.49360, 1.42755, 19.37630, 0.104621, 13.05440, 5.93667, 5.06412, 11.19720, 1.01074,
		28.76210, 1.67191, 15.71890, 9.09227, 14.55640, 0.35050, 5.44174, 52.08610, 10.472,
		28.18940, 1.62903, 16.15500, 8.97948, 14.93050, 0.38266, 5.67589, 48.16470, 11.0005,
		30.41900, 1.37113, 15.26370, 6.84706, 14.74580, 0.165191, 5.06795, 18.00300, 6.49804,
		27.30490, 1.59279, 16.72960, 8.86553, 15.61150, 0.41792, 5.83377, 45.00110, 11.4722,
		30.41560, 1.34323, 15.86200, 7.10909, 13.61450, 0.204633, 5.82008, 20.32540, 8.27903,
		30.70580, 1.30923, 15.55120, 6.71983, 14.23260, 0.167252, 5.53672, 17.49110, 6.96824,
		27.00590, 1.51293, 17.76390, 8.81174, 15.71310, .424593, 5.78370, 38.61030, 11.6883,
		29.84290, 1.32927, 16.72240, 7.38979, 13.21530, 0.263297, 6.35234, 22.94260, 9.85329,
		30.96120, 1.24813, 15.98290, 6.60834, 13.73480, 0.16864, 5.92034, 16.93920, 7.39534,
		16.88190, 0.46110, 18.59130, 8.62160, 25.55820, 1.48260, 5.86000, 36.39560, 12.0658,
		28.01090, 1.35321, 17.82040, 7.73950, 14.33590, 0.356752, 6.58077, 26.40430, 11.2299,
		30.68860, 1.21990, 16.90290, 6.82872, 12.78010, 0.212867, 6.52354, 18.65900, 9.0968,
		20.68090, 0.54500, 19.04170, 8.44840, 21.65750, 1.57290, 5.96760, 38.32460, 12.6089,
		25.08530, 1.39507, 18.49730, 7.65105, 16.88830, 0.443378, 6.48216, 28.22620, 12.0205,
		29.56410, 1.21152, 18.06000, 7.05639, 12.83740, .284738, 6.89912, 20.74820, 10.6268,
		27.54460, 0.65515, 19.15840, 8.70751, 15.53800, 1.96347, 5.52593, 45.81490, 13.1746,
		21.39850, 1.47110, 20.47230, 0.517394, 18.74780, 7.43463, 6.82847, 28.84820, 12.5258,
		30.86950, 1.10080, 18.38410, 6.53852, 11.93280, 0.219074, 7.00574, 17.21140, 9.8027,
		31.06170, 0.69020, 13.06370, 2.35760, 18.44200, 8.61800, 5.96960, 47.25790, 13.4118,
		21.78860, 1.33660, 19.56820, 0.48838, 19.14060, 6.77270, 7.01107, 23.81320, 12.4734,
		32.12440, 1.00566, 18.80030, 6.10926, 12.01750, 0.147041, 6.96886, 14.71400, 8.08428,
		33.36890, 0.70400, 12.95100, 2.92380, 16.58770, 8.79370, 6.46920, 48.00930, 13.5782,
		21.80530, 1.23560, 19.50260, 6.24149, 19.10530, 0.469999, 7.10295, 20.31850, 12.4711,
		33.53640, 0.91654, 25.09460, 0.039042, 19.24970, 5.71414, 6.91555, 12.82850, -6.7994,
		34.67260, 0.700999, 15.47330, 3.55078, 13.11380, 9.55642, 7.02588, 47.00450, 13.677,
		35.31630, 0.68587, 19.02110, 3.97458, 9.49887, 11.38240, 7.42518, 45.47150, 13.7108,
		35.56310, 0.66310, 21.28160, 4.06910, 8.00370, 14.04220, 7.44330, 44.24730, 13.6905,
		35.92990, 0.646453, 23.05470, 4.17619, 12.14390, 23.10520, 2.11253, 150.64500, 13.7247,
		35.76300, 0.616341, 22.90640, 3.87135, 12.47390, 19.98870, 3.21097, 142.32500, 13.6211,
		35.21500, 0.604909, 21.67000, 3.57670, 7.91342, 12.60100, 7.65078, 29.84360, 13.5431,
		35.65970, 0.589092, 23.10320, 3.65155, 12.59770, 18.59900, 4.08655, 117.02000, 13.5266,
		35.17360, 0.579689, 22.11120, 3.41437, 8.19216, 12.91870, 7.05545, 25.94430, 13.4637,
		35.56450, 0.563359, 23.42190, 3.46204, 12.74730, 17.83090, 4.80703, 99.17220, 13.4314,
		35.10070, 0.555054, 22.44180, 3.24498, 9.78554, 13.46610, 5.29444, 23.95330, 13.376,
		35.88470, 0.547751, 23.29480, 3.41519, 14.18910, 16.92350, 4.17287, 105.25100, 13.4287,
		36.02280, 0.52930, 23.41280, 3.32530, 14.94910, 16.09270, 4.18800, 100.61300, 13.3966,
		35.57470, 0.52048, 22.52590, 3.12293, 12.21650, 12.71480, 5.37073, 26.33940, 13.3092,
		35.37150, 0.516598, 22.53260, 3.05053, 12.02910, 12.57230, 4.79840, 23.45820, 13.2671,
		34.85090, 0.507079, 22.75840, 2.89030, 14.00990, 13.17670, 1.21457, 25.20170, 13.1665,
		36.18740, 0.511929, 23.59640, 3.25396, 15.64020, 15.36220, 4.18550, 97.49080, 13.3573,
		35.70740, 0.502322, 22.61300, 3.03807, 12.98980, 12.14490, 5.43227, 25.49280, 13.2544,
		35.51030, 0.498626, 22.57870, 2.96627, 12.77660, 11.94840, 4.92159, 22.75020, 13.2116,
		35.01360, 0.48981, 22.72860, 2.81099, 14.38840, 12.33000, 1.75669, 22.65810, 13.113,
		36.52540, 0.499384, 23.80830, 3.26371, 16.77070, 14.94550, 3.47947, 105.9800, 13.3812,
		35.84000, 0.484938, 22.71690, 2.96118, 13.58070, 11.53310, 5.66016, 24.39920, 13.1991,
		35.64930, 0.481422, 22.64600, 2.89020, 13.35950, 11.31600, 5.18831, 21.83010, 13.1555,
		35.17360, 0.473204, 22.71810, 2.73848, 14.76350, 11.55300, 2.28678, 20.93030, 13.0582,
		36.67060, 0.483629, 24.09920, 3.20647, 17.34150, 14.31360, 3.49331, 102.2730, 13.3592,
		36.64880, 0.465154, 24.40960, 3.08997, 17.39900, 13.43460, 4.21665, 88.48340, 13.2887,
		36.78810, 0.451018, 24.77360, 3.04619, 17.89190, 12.89460, 4.23284, 86.00300, 13.2754,
		36.91850, 0.437533, 25.19950, 3.00775, 18.33170, 12.40440, 4.24391, 83.78810, 13.2674,
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

void DebyeCalTester::electronInitialize()
{
	pdb = nullptr;
	_aff_calculator = nullptr;

	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
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

F_TYPE DebyeCalTester::xrayAtomicFF(F_TYPE q, int elem) {
	F_TYPE res = 0.0;
	F_TYPE sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

#pragma unroll 4
	for (int i = 0; i < 4; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	res += (atmFFcoefs(elem, 8));
	return res;
}

F_TYPE DebyeCalTester::electronAtomicFF(F_TYPE q, int elem) {
	//This code is not being used in the DC
	F_TYPE res = 0.0;
	F_TYPE sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

#pragma unroll 5
	for (int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	return res;
}

bool DebyeCalTester::GetHasAnomalousScattering()
{
	return pdb->getHasAnomalous();
}

DebyeCalTester::~DebyeCalTester()
{
	if (pdb) delete pdb;
	if (_aff_calculator) delete _aff_calculator;
}


double AnomDebyeCalTester::Calculate(double q, int nLayers, VectorXd& p)
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
	_aff_calculator->GetAllAFFs((float2*)atmAmps.data(), q);
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

VectorXd AnomDebyeCalTester::CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress /*= NULL*/, void* progressArgs /*= NULL*/)
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

void AnomDebyeCalTester::PreCalculate(VectorXd& p, int nLayers)
{
	DebyeCalTester::PreCalculate(p, nLayers);
}

std::complex<F_TYPE> AnomDebyeCalTester::anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime, bool electron)
{
	if (electron)
		return DebyeCalTester::electronAtomicFF(q, elem) + fPrime + std::complex<F_TYPE>(0, 1) * fPrimePrime;
	else
		return DebyeCalTester::xrayAtomicFF(q, elem) + fPrime + std::complex<F_TYPE>(0, 1) * fPrimePrime;
}

bool PDBAmplitude::GetHasAnomalousScattering()
{
	return pdb->getHasAnomalous();
}

#pragma endregion	// CPDB Reader class

#pragma region XRayPDBAmplitude
// Consider moving this region to a different file
XRayPDBAmplitude::XRayPDBAmplitude(string filename, bool bCenter, string anomalousFilename, int model /*= 0*/) : PDBAmplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb = new PDBReader::XRayPDBReaderOb<float>(filename, bCenter, model, anomalousFilename);

	bCentered = bCenter;

	SetPDBHash();
}

XRayPDBAmplitude::XRayPDBAmplitude(const char* buffer, size_t buffSize, const char* filenm, size_t fnSize, bool bCenter, const char* anomalousFilename, size_t anomBuffSize, int model) : PDBAmplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb->fn.assign(filenm, fnSize);

	try
	{
		if (anomalousFilename && anomBuffSize > 0)
			status = pdb->readAnomalousbuffer(anomalousFilename, anomBuffSize);

		if (PDB_OK == status)
			status = pdb->readPDBbuffer(buffer, buffSize, bCenter, model);
	}
	catch (PDBReader::pdbReader_exception& e)
	{
		status = PDB_READER_ERRS::ERROR_IN_PDB_FILE;
		throw backend_exception(e.GetErrorCode(), e.GetErrorMessage().c_str());
	}
	bCentered = bCenter;

	SetPDBHash();
}

XRayPDBAmplitude::XRayPDBAmplitude() {
	gridStatus = AMP_UNINITIALIZED;

	status = UNINITIALIZED;
	initialize();
}

FACC XRayPDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

	for (int i = 0; i < 4; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);
	res += (atmFFcoefs(elem, 8));
	return res;
}

void XRayPDBAmplitude::initialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 9);
#pragma region Atomic form factor coefficients
	atmFFcoefs << 0.49300, 10.51090, 0.32290, 26.1257, 0.14020, 3.14240, 0.04080, 57.79980, 0.0030,
		0.87340, 9.10370, 0.63090, 3.35680, 0.31120, 22.9276, 0.17800, 0.98210, 0.0064,
		1.12820, 3.95460, 0.75080, 1.05240, 0.61750, 85.3905, 0.46530, 168.26100, 0.0377,
		0.69680, 4.62370, 0.78880, 1.95570, 0.34140, 0.63160, 0.15630, 10.09530, 0.0167,
		1.59190, 43.6427, 1.12780, 1.86230, 0.53910, 103.483, 0.70290, 0.54200, 0.0385,
		6.26030, 0.00270, 0.88490, 0.93130, 0.79930, 2.27580, 0.16470, 5.11460, -6.1092,
		2.05450, 23.2185, 1.33260, 1.02100, 1.09790, 60.3498, 0.70680, 0.14030, -0.1932,
		2.31000, 20.8439, 1.02000, 10.2075, 1.58860, 0.56870, 0.86500, 51.65120, 0.2156, // Carbon
		12.2126, 0.00570, 3.13220, 9.89330, 2.01250, 28.9975, 1.16630, 0.58260, -11.5290,
		3.04850, 13.2771, 2.28680, 5.70110, 1.54630, 0.32390, 0.86700, 32.90890, 0.2508,
		4.19160, 12.8573, 1.63969, 4.17236, 1.52673, 47.0179, -20.307, -0.01404, 21.9412,
		3.53920, 10.2825, 2.64120, 4.29440, 1.51700, 0.26150, 1.02430, 26.14760, 0.2776,	// F
		3.63220, 5.27756, 3.51057, 14.7353, 1.26064, 0.442258, 0.940706, 47.34370, 0.653396,
		3.95530, 8.40420, 3.11250, 3.42620, 1.45460, 0.23060, 1.12510, 21.71840, 0.3515,
		4.76260, 3.28500, 3.17360, 8.84220, 1.26740, 0.31360, 1.11280, 129.42400, 0.676,
		3.25650, 2.66710, 3.93620, 6.11530, 1.39980, 0.20010, 1.00320, 14.03900, 0.404,
		5.42040, 2.82750, 2.17350, 79.2611, 1.22690, 0.38080, 2.30730, 7.19370, 0.8584,
		3.49880, 2.16760, 3.83780, 4.75420, 1.32840, 0.18500, 0.84970, 10.14110, 0.4853,
		6.42020, 3.03870, 1.90020, 0.74260, 1.59360, 31.5472, 1.96460, 85.08860, 1.1151,
		4.17448, 1.93816, 3.38760, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786,
		6.29150, 2.43860, 3.03530, 32.3337, 1.98910, 0.67850, 1.54100, 81.69370, 1.1407,
		4.43918, 1.64167, 3.20345, 3.43757, 1.19453, 0.21490, 0.41653, 6.65365, 0.746297,
		6.43450, 1.90670, 4.17910, 27.1570, 1.78000, 0.52600, 1.49080, 68.16450, 1.1149,
		6.29150, 2.43860, 3.03530, 32.3337, 1.98910, 0.67850, 1.54100, 81.69370, 1.1407,
		11.46040, 0.01040, 7.19640, 1.16620, 6.25560, 18.5194, 1.64550, 47.77840, -9.5574,
		18.29150, 0.00660, 7.40840, 1.17170, 6.53370, 19.5424, 2.33860, 60.44860, -16.378,
		7.48450, 0.90720, 6.77230, 14.8407, 0.65390, 43.8983, 1.64420, 33.39290, 1.4445,
		8.21860, 12.79490, 7.43980, 0.77480, 1.05190, 213.187, 0.86590, 41.68410, 1.4228,
		7.95780, 12.63310, 7.49170, 0.76740, 6.35900, -0.0020, 1.19150, 31.91280, -4.9978,
		8.62660, 10.44210, 7.38730, 0.65990, 1.58990, 85.7484, 1.02110, 178.43700, 1.3751,
		15.63480, -0.00740, 7.95180, 0.60890, 8.43720, 10.3116, 0.85370, 25.99050, -14.875,
		9.18900, 9.02130, 7.36790, 0.57290, 1.64090, 136.108, 1.46800, 51.35310, 1.3329,
		13.40080, 0.29854, 8.02730, 7.96290, 1.65943, -0.28604, 1.57936, 16.06620, -6.6667,
		9.75950, 7.85080, 7.35580, 0.50000, 1.69910, 35.6338, 1.90210, 116.10500, 1.2807,
		9.11423, 7.52430, 7.62174, 0.457585, 2.27930, 19.5361, 0.087899, 61.65580, 0.897155,
		17.73440, 0.22061, 8.73816, 7.04716, 5.25691, -0.15762, 1.92134, 15.97680, -14.652,
		19.51140, 0.178847, 8.23473, 6.67018, 2.01341, -0.29263, 1.52080, 12.94640, -13.28,
		10.29710, 6.86570, 7.35110, 0.43850, 2.07030, 26.8938, 2.05710, 102.47800, 1.2199,
		10.10600, 6.88180, 7.35410, 0.44090, 2.28840, 20.3004, 0.02230, 115.12200, 1.2298,
		9.43141, 6.39535, 7.74190, 0.383349, 2.15343, 15.1908, 0.016865, 63.96900, 0.656565,
		15.68870, 0.679003, 8.14208, 5.40135, 2.03081, 9.97278, -9.57600, 0.940464, 1.7143,
		10.64060, 6.10380, 7.35370, 0.39200, 3.32400, 20.26260, 1.49220, 98.73990, 1.1832,
		9.54034, 5.66078, 7.75090, 0.344261, 3.58274, 13.30750, 0.509107, 32.42240, 0.616898,
		9.68090, 5.59463, 7.81136, 0.334393, 2.87603, 12.82880, 0.113575, 32.87610, 0.518275,
		11.28190, 5.34090, 7.35730, 0.34320, 3.01930, 17.86740, 2.24410, 83.75430, 1.0896,
		10.80610, 5.27960, 7.36200, 0.34350, 3.52680, 14.34300, 0.21840, 41.32350, 1.0874,
		9.84521, 4.91797, 7.87194, 0.294393, 3.56531, 10.81710, 0.323613, 24.12810, 0.393974,
		9.96253, 4.84850, 7.97057, 0.283303, 2.76067, 10.48520, 0.054447, 27.57300, 0.251877,
		11.76950, 4.76110, 7.35730, 0.30720, 3.52220, 15.35350, 2.30450, 76.88050, 1.0369,
		11.04240, 4.65380, 7.37400, 0.30530, 4.13460, 12.05460, 0.43990, 31.28090, 1.0097,
		11.17640, 4.61470, 7.38630, 0.30050, 3.39480, 11.67290, 0.07240, 38.55660, 0.9707,
		12.28410, 4.27910, 7.34090, 0.27840, 4.00340, 13.53590, 2.34880, 71.16920, 1.0118,
		11.22960, 4.12310, 7.38830, 0.27260, 4.73930, 10.24430, 0.71080, 25.64660, 0.9324,
		10.33800, 3.90969, 7.88173, 0.238668, 4.76795, 8.35583, 0.725591, 18.34910, 0.286667,
		12.83760, 3.87850, 7.29200, 0.25650, 4.44380, 12.17630, 2.38000, 66.34210, 1.0341,
		11.41660, 3.67660, 7.40050, 0.24490, 5.34420, 8.87300, 0.97730, 22.16260, 0.8614,
		10.78060, 3.54770, 7.75868, 0.22314, 5.22746, 7.64468, 0.847114, 16.96730, 0.386044,
		13.33800, 3.58280, 7.16760, 0.24700, 5.61580, 11.39660, 1.67350, 64.81260, 1.191,
		11.94750, 3.36690, 7.35730, 0.22740, 6.24550, 8.66250, 1.55780, 25.84870, 0.89,
		11.81680, 3.37484, 7.11181, .244078, 5.78135, 7.98760, 1.14523, 19.89700, 1.14431,
		14.07430, 3.26550, 7.03180, 0.23330, 5.16520, 10.31630, 2.41000, 58.70970, 1.3041,
		11.97190, 2.99460, 7.38620, 0.20310, 6.46680, 7.08260, 1.39400, 18.09950, 0.7807,
		15.23540, 3.06690, 6.70060, 0.24120, 4.35910, 10.78050, 2.96230, 61.41350, 1.7189,
		12.69200, 2.81262, 6.69883, 0.22789, 6.06692, 6.36441, 1.00660, 14.41220, 1.53545,
		16.08160, 2.85090, 6.37470, 0.25160, 3.70680, 11.44680, 3.68300, 54.76250, 2.1313,
		12.91720, 2.53718, 6.70003, 0.205855, 6.06791, 5.47913, 0.859041, 11.60300, 1.45572,
		16.67230, 2.63450, 6.07010, 0.26470, 3.43130, 12.94790, 4.27790, 47.79720, 2.531,
		17.00600, 2.40980, 5.81960, 0.27260, 3.97310, 15.23720, 4.35436, 43.81630, 2.8409,
		17.17890, 2.17230, 5.23580, 16.57960, 5.63770, 0.26090, 3.98510, 41.43280, 2.9557,
		17.17180, 2.20590, 6.33380, 19.33450, 5.57540, 0.28710, 3.72720, 58.15350, 3.1776,
		17.35550, 1.93840, 6.72860, 16.56230, 5.54930, 0.22610, 3.53750, 39.39720, 2.825,
		17.17840, 1.78880, 9.64350, 17.31510, 5.13990, 0.27480, 1.52920, 164.93400, 3.4873,
		17.58160, 1.71390, 7.65980, 14.79570, 5.89810, 0.16030, 2.78170, 31.20870, 2.0782,
		17.56630, 1.55640, 9.81840, 14.09880, 5.42200, 0.16640, 2.66940, 132.37600, 2.5064,
		18.08740, 1.49070, 8.13730, 12.69630, 2.56540, 24.56510, -34.19300, -0.01380, 41.4025,
		17.77600, 1.40290, 10.29460, 12.80060, 5.72629, 0.125599, 3.26588, 104.35400, 1.91213,
		17.92680, 1.35417, 9.15310, 11.21450, 1.76795, 22.65990, -33.10800, -0.01319, 40.2602,
		17.87650, 1.27618, 10.94800, 11.91600, 5.41732, 0.117622, 3.65721, 87.66270, 2.06929,
		18.16680, 1.21480, 10.05620, 10.14830, 1.01118, 21.60540, -2.64790, -0.10276, 9.41454,
		17.61420, 1.18865, 12.01440, 11.76600, 4.04183, 0.204785, 3.53346, 69.79570, 3.75591,
		19.88120, 0.019175, 18.06530, 1.13305, 11.01770, 10.16210, 1.94715, 28.33890, -12.912,
		17.91630, 1.12446, 13.34170, 0.028781, 10.79900, 9.28206, 0.337905, 25.72280, -6.3934,
		3.70250, 0.27720, 17.23560, 1.09580, 12.88760, 11.00400, 3.74290, 61.65840, 4.3875,
		21.16640, 0.014734, 18.20170, 1.03031, 11.74230, 9.53659, 2.30951, 26.63070, -14.421,
		21.01490, 0.014345, 18.09920, 1.02238, 11.46320, 8.78809, 0.740625, 23.34520, -14.316,
		17.88710, 1.03649, 11.17500, 8.48061, 6.57891, 0.058881, 0.00000, 0.00000, 0.344941,
		19.13010, 0.864132, 11.09480, 8.14487, 4.64901, 21.57070, 2.71263, 86.84720, 5.40428,
		19.26740, 0.80852, 12.91820, 8.43467, 4.86337, 24.79970, 1.56756, 94.29280, 5.37874,
		18.56380, 0.847329, 13.28850, 8.37164, 9.32602, 0.017662, 3.00964, 22.88700, -3.1892,
		18.50030, 0.844582, 13.17870, 8.12534, 4.71304, 0.036495, 2.18535, 20.85040, 1.42357,
		19.29570, 0.751536, 14.35010, 8.21758, 4.73425, 25.87490, 1.28918, 98.60620, 5.328,
		18.87850, 0.764252, 14.12590, 7.84438, 3.32515, 21.24870, -6.19890, -0.01036, 11.8678,
		18.85450, 0.760825, 13.98060, 7.62436, 2.53464, 19.33170, -5.65260, -0.01020, 11.2835,
		19.33190, 0.69866, 15.50170, 7.98939, 5.29537, 25.20520, 0.60584, 76.89860, 5.26593,
		19.17010, 0.696219, 15.20960, 7.55573, 4.32234, 22.50570, 0.00000, 0.00000, 5.2916,
		19.24930, 0.683839, 14.79000, 7.14833, 2.89289, 17.91440, -7.94920, 0.005127, 13.0174,
		19.28080, 0.64460, 16.68850, 7.47260, 4.80450, 24.66050, 1.04630, 99.81560, 5.179,
		19.18120, 0.646179, 15.97190, 7.19123, 5.27475, 21.73260, 0.357534, 66.11470, 5.21572,
		19.16430, 0.645643, 16.24560, 7.18544, 4.37090, 21.40720, 0.00000, 0.00000, 5.21404,
		19.22140, 0.59460, 17.64440, 6.90890, 4.46100, 24.70080, 1.60290, 87.48250, 5.0694,
		19.15140, 0.597922, 17.25350, 6.80639, 4.47128, 20.25210, 0.00000, 0.00000, 5.11937,
		19.16240, 0.54760, 18.55960, 6.37760, 4.29480, 25.84990, 2.03960, 92.80290, 4.9391,
		19.10450, 0.551522, 18.11080, 6.32470, 3.78897, 17.35950, 0.00000, 0.00000, 4.99635,
		19.18890, 5.83030, 19.10050, 0.50310, 4.45850, 26.89090, 2.46630, 83.95710, 4.7821,
		19.10940, 0.50360, 19.05480, 5.83780, 4.56480, 23.37520, 0.48700, 62.20610, 4.7861,
		18.93330, 5.76400, 19.71310, 0.46550, 3.41820, 14.00490, 0.01930, -0.75830, 3.9182,
		19.64180, 5.30340, 19.04550, 0.46070, 5.03710, 27.90740, 2.68270, 75.28250, 4.5909,
		18.97550, 0.467196, 18.93300, 5.22126, 5.10789, 19.59020, 0.288753, 55.51130, 4.69626,
		19.86850, 5.44853, 19.03020, 0.467973, 2.41253, 14.12590, 0.00000, 0.00000, 4.69263,
		19.96440, 4.81742, 19.01380, 0.420885, 6.14487, 28.52840, 2.52390, 70.84030, 4.352,
		20.14720, 4.34700, 18.99490, 0.23140, 7.51380, 27.76600, 2.27350, 66.87760, 4.07121,
		20.23320, 4.35790, 18.99700, 0.38150, 7.80690, 29.52590, 2.88680, 84.93040, 4.0714,
		20.29330, 3.92820, 19.02980, 0.34400, 8.97670, 26.46590, 1.99000, 64.26580, 3.7118,
		20.38920, 3.56900, 19.10620, 0.31070, 10.66200, 24.38790, 1.49530, 213.90400, 3.3352,
		20.35240, 3.55200, 19.12780, 0.30860, 10.28210, 23.71280, 0.96150, 59.45650, 3.2791,
		20.33610, 3.21600, 19.29700, 0.27560, 10.88800, 20.20730, 2.69590, 167.20200, 2.7731,
		20.18070, 3.21367, 19.11360, 0.28331, 10.90540, 20.05580, 0.77634, 51.74600, 3.02902,
		20.57800, 2.94817, 19.59900, 0.244475, 11.37270, 18.77260, 3.28719, 133.12400, 2.14678,
		20.24890, 2.92070, 19.37630, 0.250698, 11.63230, 17.82110, 0.336048, 54.94530, 2.4086,
		21.16710, 2.81219, 19.76950, 0.226836, 11.85130, 17.60830, 3.33049, 127.11300, 1.86264,
		20.80360, 2.77691, 19.55900, 0.23154, 11.93690, 16.54080, 0.612376, 43.16920, 2.09013,
		20.32350, 2.65941, 19.81860, 0.21885, 12.12330, 15.79920, 0.144583, 62.23550, 1.5918,
		22.04400, 2.77393, 19.66970, 0.222087, 12.38560, 16.76690, 2.82428, 143.64400, 2.0583,
		21.37270, 2.64520, 19.74910, 0.214299, 12.13290, 15.32300, 0.97518, 36.40650, 1.77132,
		20.94130, 2.54467, 20.05390, 0.202481, 12.46680, 14.81370, 0.296689, 45.46430, 1.24285,
		22.68450, 2.66248, 19.68470, 0.210628, 12.77400, 15.88500, 2.85137, 137.90300, 1.98486,
		21.96100, 2.52722, 19.93390, 0.199237, 12.12000, 14.17830, 1.51031, 30.87170, 1.47588,
		23.34050, 2.56270, 19.60950, 0.202088, 13.12350, 15.10090, 2.87516, 132.72100, 2.02876,
		22.55270, 2.41740, 20.11080, 0.185769, 12.06710, 13.12750, 2.07492, 27.44910, 1.19499,
		24.00420, 2.47274, 19.42580, 0.19651, 13.43960, 14.39960, 2.89604, 128.00700, 2.20963,
		23.15040, 2.31641, 20.25990, .174081, 11.92020, 12.15710, 2.71488, 24.82420, .954586,
		24.62740, 2.38790, 19.08860, 0.19420, 13.76030, 17.75460, 2.92270, 123.17400, 2.5745,
		24.00630, 2.27783, 19.95040, 0.17353, 11.80340, 11.60960, 3.87243, 26.51560, 1.36389,
		23.74970, 2.22258, 20.37450, 0.16394, 11.85090, 11.31100, 3.26503, 22.99660, 0.759344,
		25.07090, 2.25341, 19.07980, 0.181951, 13.85180, 12.93310, 3.54545, 101.39800, 2.4196,
		24.34660, 2.15530, 20.42080, 0.15552, 11.87080, 10.57820, 3.71490, 21.70290, 0.64509,
		25.89760, 2.24256, 18.21850, 0.196143, 14.31670, 12.66480, 2.95354, 115.36200, 3.58324,
		24.95590, 2.05601, 20.32710, 0.149525, 12.24710, 10.04990, 3.77300, 21.27730, 0.691967,
		26.50700, 2.18020, 17.63830, 0.202172, 14.55960, 12.18990, 2.96577, 111.87400, 4.29728,
		25.53950, 1.98040, 20.28610, 0.143384, 11.98120, 9.34972, 4.50073, 19.58100, 0.68969,
		26.90490, 2.07051, 17.29400, 0.19794, 14.55830, 11.44070, 3.63837, 92.65660, 4.56796,
		26.12960, 1.91072, 20.09940, 0.139358, 11.97880, 8.80018, 4.93676, 18.59080, 0.852795,
		27.65630, 2.07356, 16.42850, 0.223545, 14.97790, 11.36040, 2.98233, 105.70300, 5.92046,
		26.72200, 1.84659, 19.77480, 0.13729, 12.15060, 8.36225, 5.17379, 17.89740, 1.17613,
		28.18190, 2.02859, 15.88510, 0.238849, 15.15420, 10.99750, 2.98706, 102.96100, 6.75621,
		27.30830, 1.78711, 19.33200, 0.136974, 12.33390, 7.96778, 5.38348, 17.29220, 1.63929,
		28.66410, 1.98890, 15.43450, 0.257119, 15.30870, 10.66470, 2.98963, 100.41700, 7.56672,
		28.12090, 1.78503, 17.68170, 0.15997, 13.33350, 8.18304, 5.14657, 20.39000, 3.70983,
		27.89170, 1.73272, 18.76140, 0.13879, 12.60720, 7.64412, 5.47647, 16.81530, 2.26001,
		28.94760, 1.90182, 15.22080, 9.98519, 15.10000, 0.261033, 3.71601, 84.32980, 7.97628,
		28.46280, 1.68216, 18.12100, 0.142292, 12.84290, 7.33727, 5.59415, 16.35350, 2.97573,
		29.14400, 1.83262, 15.17260, 9.59990, 14.75860, 0.275116, 4.30013, 72.02900, 8.58154,
		28.81310, 1.59136, 18.46010, 0.128903, 12.72850, 6.76232, 5.59927, 14.03660, 2.39699,
		29.20240, 1.77333, 15.22930, 9.37046, 14.51350, 0.295977, 4.76492, 63.36440, 9.24354,
		29.15870, 1.50711, 18.84070, 0.116741, 12.82680, 6.31524, 5.38695, 12.42440, 1.78555,
		29.08180, 1.72029, 15.43000, 9.22590, 14.43270, 0.321703, 5.11982, 57.05600, 9.8875,
		29.49360, 1.42755, 19.37630, 0.104621, 13.05440, 5.93667, 5.06412, 11.19720, 1.01074,
		28.76210, 1.67191, 15.71890, 9.09227, 14.55640, 0.35050, 5.44174, 52.08610, 10.472,
		28.18940, 1.62903, 16.15500, 8.97948, 14.93050, 0.38266, 5.67589, 48.16470, 11.0005,
		30.41900, 1.37113, 15.26370, 6.84706, 14.74580, 0.165191, 5.06795, 18.00300, 6.49804,
		27.30490, 1.59279, 16.72960, 8.86553, 15.61150, 0.41792, 5.83377, 45.00110, 11.4722,
		30.41560, 1.34323, 15.86200, 7.10909, 13.61450, 0.204633, 5.82008, 20.32540, 8.27903,
		30.70580, 1.30923, 15.55120, 6.71983, 14.23260, 0.167252, 5.53672, 17.49110, 6.96824,
		27.00590, 1.51293, 17.76390, 8.81174, 15.71310, .424593, 5.78370, 38.61030, 11.6883,
		29.84290, 1.32927, 16.72240, 7.38979, 13.21530, 0.263297, 6.35234, 22.94260, 9.85329,
		30.96120, 1.24813, 15.98290, 6.60834, 13.73480, 0.16864, 5.92034, 16.93920, 7.39534,
		16.88190, 0.46110, 18.59130, 8.62160, 25.55820, 1.48260, 5.86000, 36.39560, 12.0658,
		28.01090, 1.35321, 17.82040, 7.73950, 14.33590, 0.356752, 6.58077, 26.40430, 11.2299,
		30.68860, 1.21990, 16.90290, 6.82872, 12.78010, 0.212867, 6.52354, 18.65900, 9.0968,
		20.68090, 0.54500, 19.04170, 8.44840, 21.65750, 1.57290, 5.96760, 38.32460, 12.6089,
		25.08530, 1.39507, 18.49730, 7.65105, 16.88830, 0.443378, 6.48216, 28.22620, 12.0205,
		29.56410, 1.21152, 18.06000, 7.05639, 12.83740, .284738, 6.89912, 20.74820, 10.6268,
		27.54460, 0.65515, 19.15840, 8.70751, 15.53800, 1.96347, 5.52593, 45.81490, 13.1746,
		21.39850, 1.47110, 20.47230, 0.517394, 18.74780, 7.43463, 6.82847, 28.84820, 12.5258,
		30.86950, 1.10080, 18.38410, 6.53852, 11.93280, 0.219074, 7.00574, 17.21140, 9.8027,
		31.06170, 0.69020, 13.06370, 2.35760, 18.44200, 8.61800, 5.96960, 47.25790, 13.4118,
		21.78860, 1.33660, 19.56820, 0.48838, 19.14060, 6.77270, 7.01107, 23.81320, 12.4734,
		32.12440, 1.00566, 18.80030, 6.10926, 12.01750, 0.147041, 6.96886, 14.71400, 8.08428,
		33.36890, 0.70400, 12.95100, 2.92380, 16.58770, 8.79370, 6.46920, 48.00930, 13.5782,
		21.80530, 1.23560, 19.50260, 6.24149, 19.10530, 0.469999, 7.10295, 20.31850, 12.4711,
		33.53640, 0.91654, 25.09460, 0.039042, 19.24970, 5.71414, 6.91555, 12.82850, -6.7994,
		34.67260, 0.700999, 15.47330, 3.55078, 13.11380, 9.55642, 7.02588, 47.00450, 13.677,
		35.31630, 0.68587, 19.02110, 3.97458, 9.49887, 11.38240, 7.42518, 45.47150, 13.7108,
		35.56310, 0.66310, 21.28160, 4.06910, 8.00370, 14.04220, 7.44330, 44.24730, 13.6905,
		35.92990, 0.646453, 23.05470, 4.17619, 12.14390, 23.10520, 2.11253, 150.64500, 13.7247,
		35.76300, 0.616341, 22.90640, 3.87135, 12.47390, 19.98870, 3.21097, 142.32500, 13.6211,
		35.21500, 0.604909, 21.67000, 3.57670, 7.91342, 12.60100, 7.65078, 29.84360, 13.5431,
		35.65970, 0.589092, 23.10320, 3.65155, 12.59770, 18.59900, 4.08655, 117.02000, 13.5266,
		35.17360, 0.579689, 22.11120, 3.41437, 8.19216, 12.91870, 7.05545, 25.94430, 13.4637,
		35.56450, 0.563359, 23.42190, 3.46204, 12.74730, 17.83090, 4.80703, 99.17220, 13.4314,
		35.10070, 0.555054, 22.44180, 3.24498, 9.78554, 13.46610, 5.29444, 23.95330, 13.376,
		35.88470, 0.547751, 23.29480, 3.41519, 14.18910, 16.92350, 4.17287, 105.25100, 13.4287,
		36.02280, 0.52930, 23.41280, 3.32530, 14.94910, 16.09270, 4.18800, 100.61300, 13.3966,
		35.57470, 0.52048, 22.52590, 3.12293, 12.21650, 12.71480, 5.37073, 26.33940, 13.3092,
		35.37150, 0.516598, 22.53260, 3.05053, 12.02910, 12.57230, 4.79840, 23.45820, 13.2671,
		34.85090, 0.507079, 22.75840, 2.89030, 14.00990, 13.17670, 1.21457, 25.20170, 13.1665,
		36.18740, 0.511929, 23.59640, 3.25396, 15.64020, 15.36220, 4.18550, 97.49080, 13.3573,
		35.70740, 0.502322, 22.61300, 3.03807, 12.98980, 12.14490, 5.43227, 25.49280, 13.2544,
		35.51030, 0.498626, 22.57870, 2.96627, 12.77660, 11.94840, 4.92159, 22.75020, 13.2116,
		35.01360, 0.48981, 22.72860, 2.81099, 14.38840, 12.33000, 1.75669, 22.65810, 13.113,
		36.52540, 0.499384, 23.80830, 3.26371, 16.77070, 14.94550, 3.47947, 105.9800, 13.3812,
		35.84000, 0.484938, 22.71690, 2.96118, 13.58070, 11.53310, 5.66016, 24.39920, 13.1991,
		35.64930, 0.481422, 22.64600, 2.89020, 13.35950, 11.31600, 5.18831, 21.83010, 13.1555,
		35.17360, 0.473204, 22.71810, 2.73848, 14.76350, 11.55300, 2.28678, 20.93030, 13.0582,
		36.67060, 0.483629, 24.09920, 3.20647, 17.34150, 14.31360, 3.49331, 102.2730, 13.3592,
		36.64880, 0.465154, 24.40960, 3.08997, 17.39900, 13.43460, 4.21665, 88.48340, 13.2887,
		36.78810, 0.451018, 24.77360, 3.04619, 17.89190, 12.89460, 4.23284, 86.00300, 13.2754,
		36.91850, 0.437533, 25.19950, 3.00775, 18.33170, 12.40440, 4.24391, 83.78810, 13.2674,
		//////////////////////////////////////////////////////////////////////////
		// Modified atomic form factors
		0.894937, 55.7145, 0.894429, 4.03158, 3.78824, 24.8323, 3.14683e-6, 956.628, 1.42149,	// CH
		1.61908, 52.1451, 2.27205, 24.6589, 2.1815, 24.6587, 0.0019254, 152.165, 1.92445,		// CH2
		12.5735, 38.7341, -0.456658, -6.28167, 5.71547, 54.955, -11.711, 47.898, 2.87762,		// CH3
		0.00506991, 108.256, 2.03147, 14.6199, 1.82122, 14.628, 2.06506, 35.4102, 2.07168,		// NH
		3.00872, 28.3717, 0.288137, 63.9637, 3.39248, 3.51866, 2.03511, 28.3675, 0.269952,		// NH2
		0.294613, 67.4408, 6.48379, 29.1576, 5.67182, 0.54735, 6.57164, 0.547493, -9.02757,		// NH3
		-2.73406, 22.1288, 0.00966263, 94.3428, 6.64439, 13.9044, 2.67949, 32.7607, 2.39981,	// OH
		-127.811, 7.19935, 62.5514, 12.1591, 160.747, 1.88979, 2.34822, 55.952, -80.836,		// SH

		// Ions that had no x-ray form factors:
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0
		;

#pragma endregion
}

std::string XRayPDBAmplitude::Hash() const
{
	std::string str = BACKEND_VERSION "PDB: ";
	str += std::to_string(voxelStep) + std::to_string(solventED) + std::to_string(c1)
		+ std::to_string(solventRad) + std::to_string(solvationThickness) + std::to_string(outerSolventED);

	str += pdb_hash;

	return md5(str);
}

bool XRayPDBAmplitude::SetModel(Workspace& workspace) {
	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuSetPDB)
		gpuSetPDB = (GPUDirectSetPDB_t)GPUDirect_SetPDBDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_SetPDBDLL");
	if (!gpuSetPDB)
		return false;

	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return gpuSetPDB(workspace, (const float4*)&pdb->atomLocs[0], &pdb->sortedCoeffIonInd[0],
		pdb->atomLocs.size(), &pdb->sortedCoeffs[0],
		&pdb->atomsPerIon[0],
		pdb->sortedCoeffs.size() / 9);
}

std::string XRayPDBAmplitude::GetName() const {
	return "PDB: " + pdb->fn;
}
#pragma endregion // XRayPDBAmplitude

#pragma region ElectronPDBAmplitude

ElectronPDBAmplitude::ElectronPDBAmplitude(string filename, bool bCenter, string anomalousFilename, int model) : PDBAmplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb = new PDBReader::ElectronPDBReaderOb<float>(filename, bCenter, model, anomalousFilename);

	bCentered = bCenter;

	SetPDBHash();
}

ElectronPDBAmplitude::ElectronPDBAmplitude(const char* buffer, size_t buffSize, const char* filenm, size_t fnSize, bool bCenter, const char* anomalousFilename, size_t anomBuffSize, int model) : PDBAmplitude() {
	initialize();
	gridStatus = AMP_UNINITIALIZED;

	pdb->fn.assign(filenm, fnSize);

	try
	{
		if (anomalousFilename && anomBuffSize > 0)
			status = pdb->readAnomalousbuffer(anomalousFilename, anomBuffSize);

		if (PDB_OK == status)
			status = pdb->readPDBbuffer(buffer, buffSize, bCenter, model);
	}
	catch (PDBReader::pdbReader_exception& e)
	{
		status = PDB_READER_ERRS::ERROR_IN_PDB_FILE;
		throw backend_exception(e.GetErrorCode(), e.GetErrorMessage().c_str());
	}
	bCentered = bCenter;

	SetPDBHash();
}

ElectronPDBAmplitude::ElectronPDBAmplitude() {
	gridStatus = AMP_UNINITIALIZED;

	status = UNINITIALIZED;
	initialize();
}

FACC ElectronPDBAmplitude::atomicFF(FACC q, int elem) {
	// NOTE: Units are (inverse) Angstroms
	FACC res = 0.0;
	FACC sqq = q * q / (157.913670417429737901351855998);
	// Should the number be 100*pi/2 = 157.07963267948966192313216916397514420985846996876

	for (int i = 0; i < 5; i++)
		res += (atmFFcoefs(elem, 2 * i)) * exp(-atmFFcoefs(elem, (2 * i) + 1) * sqq);

	return res;
}

void ElectronPDBAmplitude::initialize() {
	bCentered = false;
	this->bUseGrid = true;	// As default
	this->bSolventLoaded = false;
	this->voxelStep = 15.4e99;
	this->solventED = 0.0;
	this->outerSolventED = 0.0;
	atmFFcoefs.resize(NUMBER_OF_ATOMIC_FORM_FACTORS, 10);
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

std::string ElectronPDBAmplitude::Hash() const
{
	std::string str = BACKEND_VERSION "EPDB: ";
	str += std::to_string(voxelStep) + std::to_string(solventED) + std::to_string(c1)
		+ std::to_string(solventRad) + std::to_string(solvationThickness) + std::to_string(outerSolventED);

	str += pdb_hash;

	return md5(str);
}

bool ElectronPDBAmplitude::SetModel(Workspace& workspace) {
	// !!!!This has to changed appropriatly when converting to E+!!!!
	bool isE = true;
	int divider = 9;

	if (isE) { divider = 10; }

	if (!g_useGPUAndAvailable)
		return false;

	if (!gpuSetPDB)
		gpuSetPDB = (GPUDirectSetPDB_t)GPUDirect_SetPDBDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUDirect_SetPDBDLL");
	if (!gpuSetPDB)
		return false;

	// Set the PDB coefficients	--> NOTE: The atomic form factors are calculated here. IF the user selects solvent subtraction or solvent only, the AFFs will be wrong. The parameters are only sent in a more inner loop...
	return gpuSetPDB(workspace, (const float4*)&pdb->atomLocs[0], &pdb->sortedCoeffIonInd[0],
		pdb->atomLocs.size(), &pdb->sortedCoeffs[0],
		&pdb->atomsPerIon[0],
		pdb->sortedCoeffs.size() / divider);
}

std::string ElectronPDBAmplitude::GetName() const {
	return "EPDB: " + pdb->fn;
}

#pragma endregion	// ElectronPDBAmplitude
