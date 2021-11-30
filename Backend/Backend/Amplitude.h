#ifndef __AMPLITUDE_H
#define __AMPLITUDE_H
#define BOOST_SYSTEM_NO_DEPRECATED
#pragma warning( disable : 4251 )

#pragma once

#pragma managed(push)
#pragma unmanaged

//#include "gitrev.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "../GPU/GPUInterface.h" // Make sure that CUDA defines __host__ etc. (otherwise we get macro redefinition warnings)
#include "Geometry.h"
#include "mathfuncs.h"
#include "../PDBReaderLib/PDBReaderLib.h"
#include "../GPU/Atomic Form Factor.h"
#include "LocalBackend.h"
#include "Model.h"
#pragma managed(pop)

using std::vector;
using std::string;

typedef double FACC;

typedef Eigen::Array<std::complex<FACC>, Eigen::Dynamic, 1> ArrayXcX;
typedef Eigen::Array<FACC, Eigen::Dynamic, 1> ArrayXX;


//#ifdef _DEBUG	// Options to turn on and off in the code
//#define SYM_TREE_CHECK
//#endif

template <typename T>
struct idxtemplate {
	T x, y, z;
	T& operator[](int index)
	{
		switch (index) {
		default:
			//was out_of_range
			throw backend_exception(ERROR_INVALIDARGS, "Index must be in the range [0,3)");
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
	}
	T operator[](int index) const
	{
		switch (index) {
		default:
			////was out_of_range
			throw backend_exception(ERROR_INVALIDARGS, "Index must be in the range [0,3)");
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
	}
};

typedef idxtemplate<uint64_t> idx;          // 3D index for array
typedef idxtemplate<int> sIdx;				// 3D index for array
typedef idxtemplate<float> fIdx;			// 3D index for array

// Forward declarations
namespace boost {
	template <typename T, std::size_t NumDims, typename Allocator>
	class multi_array;
}
typedef boost::multi_array<unsigned char, 3, std::allocator<unsigned char> > array3;	// 3D array

// Various defines to change the behavior of the calculations
//#define NO_GRID
#define WRITE_READ_FILE
//#define PRESENT_TO_TOM
//#define GAUSS_LEGENDRE_INTEGRATION
#define SPHERICAL_MATRIX
// #define COMPRESSED_FILE
//#define CF4_CHEAT
//#define CF4_QVEC

// #if defined(PRESENT_TO_TOM) && !defined(NO_GRID)
// #define NO_GRID
// #endif

using Eigen::MatrixXd;
using Eigen::Vector3d;

// Forward declaration
class Grid;
class JsonWriter;

class EXPORTED_BE Amplitude {

protected:
	friend class DomainModel;
	friend class tests;	///< For the "unit test" class

	/// Flag indicating whether or not a grid should be used
	bool bUseGrid;

	/// Flag to indicate whether or not the amplitude is allowed to be calculated
	///  using a GPU. Does not mean that the internal model has a GPU implementation.
	bool bUseGPU;

	PDB_READER_ERRS status;
	AMPLITUDE_GRID_STATUS gridStatus;

	VectorXd previousParameters;

	Eigen::Matrix<FACC, 3, 3, 0, 3, 3> RotMat;

	/// The 3D array of the scattering amplitudes
#ifdef _DEBUG
public:
#endif // _DEBUG
	Grid *grid;

	std::wstring rndPath;

protected:
	Amplitude();

public:
	void setrndPath(std::wstring w){ rndPath = w; }
	std::wstring getrndPath(){ return rndPath;}
	bool ampiscached();
	/// Translation and rotation variables
	FACC tx, ty, tz;
	Radian ra, rb, rg;
	double scale;

	virtual ~Amplitude();

	virtual std::string Hash() const = 0;

	virtual std::string GetName() const = 0;

	virtual void SetLocationData(FACC x, FACC y, FACC z, Radian alpha, Radian beta, Radian gamma);

	/// Organize parameters from the parameter vector into the matrix and vector defined earlier.
	virtual void OrganizeParameters(const VectorXd& p, int nLayers) {}

	/// Called before each series of q-calculations
	/// If implemented and dependent on parameters, should set status to UNINITIALIZED
	virtual void PreCalculate(VectorXd& p, int nLayers);

	virtual bool GridIsReadyToBeUsed();
	virtual void ResetGrid();
	bool ampWasReset;

	/**
	* GetHeader Fills the header parameter with a header to be written to a file
	* @param depth The depth from the root calling instance. Each header line should be
	*				preceded by (depth + 1) '#'
	* @param header The header to be filled
	* @return void
	**/
	virtual void GetHeader(unsigned int depth, std::string &header) = 0;
	virtual void GetHeader(unsigned int depth, JsonWriter &writer) = 0;

	virtual bool InitializeGrid(double qMax, int sections);

	virtual Grid *GetInternalGridPointer();

public:
	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz) = 0;
	virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL,
		void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);
	virtual std::complex<FACC> getAmplitude(FACC qx, FACC qy, FACC qz);

	virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi);

	ArrayXcX getAmplitudesAtPointsWithoutGrid(double newTheta, double newPhi, const std::vector<FACC> &relevantQs, Eigen::Ref<ArrayXcX> phases);


	virtual PDB_READER_ERRS getError() const;

	/**
	* @name	GetUseGridWithChildren
	* @brief	This method determines if from this point down in the model tree all models are set to use a grid.
	* @ret True iff this model and all children are set to use a grid/lookup table (bUseGrid == true)
	*/
	virtual bool GetUseGridWithChildren() const;
	/**
	* @name	GetUseGridWithChildren
	* @brief	This method determines if any descendant in the model tree uses a grid.
	* @ret True iff this model or any descendants are set to use a grid/lookup table (bUseGrid == true)
	*/
	virtual bool GetUseGridAnyChildren() const;

	virtual bool GetUseGrid() const;
	virtual void SetUseGrid(bool bUse);

	virtual bool GetUseGPU() const;
	virtual void SetUseGPU(bool bUse);

	virtual void GetTranslationRotationVariables(FACC& x, FACC& y, FACC& z, FACC& a, FACC& b, FACC& g);

	virtual double *GetDataPointer();
	virtual u64 GetDataPointerSize();

	virtual int GetGridSize() const;
	virtual double GetGridStepSize() const;


	virtual PDB_READER_ERRS WriteAmplitudeToFile(const std::wstring& fileName);
	virtual PDB_READER_ERRS WriteAmplitudeToFile(const std::wstring& fileName, std::string header);
	virtual PDB_READER_ERRS WriteAmplitudeToFile(const std::string& fileName);
	virtual PDB_READER_ERRS WriteAmplitudeToFile(const std::string& fileName, std::string header);

	virtual PDB_READER_ERRS WriteAmplitudeToStream(std::ostream & stream);
	virtual PDB_READER_ERRS WriteAmplitudeToStream(std::ostream & stream, std::string header);

	virtual PDB_READER_ERRS WriteAmplitudeToCache();
	virtual PDB_READER_ERRS ReadAmplitudeFromCache();

	virtual bool GetHasAnomalousScattering();

	virtual Eigen::VectorXd GetPreviousParameters();
};

// Forward declaration
class SolventSpace
{
public:
	typedef char ScalarType;
	typedef Eigen::Array< size_t, 3, 1 > Dimensions;
	typedef Eigen::Array< ScalarType, -1, 1 > vect1d;
	typedef Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic> array_t;
	ScalarType& operator()(size_t x, size_t y, size_t z);
	void allocate(size_t x, size_t y, size_t z, float voxel_length);
	void deallocate();
	Eigen::Map<array_t, Eigen::AlignmentType::Aligned> SliceX(size_t x);
	Eigen::Map<array_t, 0, Eigen::Stride<Eigen::Dynamic, 1>> SurroundingVoxels(size_t x, size_t y, size_t z);
	Dimensions dimensions();

	array_t& SpaceRef();

protected:
	float _voxel_length;
	array_t _solvent_space;
	size_t _zy_plane;
	size_t _x_size, _y_size, _z_size;
};


class EXPORTED_BE PDBAmplitude : public Amplitude, public IGPUCalculable,
	public IGPUGridCalculable {
protected:
	friend class DomainModel;

	PDBReader::PDBReaderOb<float> pdb;
	/// Values of Table 2.2B (International Tables of X-ray Crystallography Vol IV)
	Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> atmFFcoefs;
	/// Contains the MD5 hash of the pdb object so that we don't have to collect it over and over again.
	std::string pdb_hash;
	void SetPDBHash();

	// Variables to describe the displaced solvent volume
	FACC xMin, xMax, yMin, yMax, zMin, zMax;	// The range of the PDB
	FACC voxelStep;		///< The size (in nm) of a step in any direction along any axis
	FACC solventED, c1, solventRad, solvationThickness, outerSolventED;

	/** solventSpace
	* This variable is a 3D grid of the real space that the protein occupies with a voxel
	* size of voxelStep. The space is initialized as zeros and changes as follows:
	*
	* Every voxel within (rad[atmInd[i]] + solventRad) of an atom is changed to 1. The outer
	* solvent is then changed from 0 to 2 (floodfill). All remaining 0's are thus trapped
	* volume and considered excluded volume for the solvent and changed to 1. The outer
	* solventRad voxels are then identified and changed from 1 to 3. The space is subsequently
	* reduced to larger boxes whose dimensions and locations are saved to sparse matrices and
	* changed from 1 to 5.
	*
	* Summary:
	* 0 (Blue)		- Solvent or Hole in first step
	* 1 (Green)	- Atoms
	* 2 (Red)		- Solvent sufficiently distant from the atoms
	* 3 (Lt. Blue)	- Solvent within solventRad of an outer atom (outer solvent)
	* 4 (Hot Pink)	- Holes
	* 5 (Yellow)	- Space filled by an atom that has already been taken into account when reducing
	*		the solvent space
	* 6 (Gray)		- Space filled by outer solvent that has already been taken into account when
	*		reducing the solvent space
	*/
	SolventSpace _solvent_space;
	bool bSolventLoaded;
	std::vector<fIdx> solventBoxCOM;
	std::vector<idx>  solventBoxDims;
	std::vector<fIdx> outerSolventBoxCOM;
	std::vector<idx>  outerSolventBoxDims;

	Eigen::ArrayX3f solventBoxCOM_array, outerSolventBoxCOM_array;
	Eigen::ArrayX3f solventBoxDims_array, outerSolventBoxDims_array;

	bool bCentered;
public:
	virtual ~PDBAmplitude();
	PDBAmplitude();
	PDBAmplitude(string filename, bool bCenter, string anomalousFilename = "", int model = 0);
	PDBAmplitude(const char *buffer, size_t buffSize, const char *filenm, size_t fnSize,
		bool bCenter, const char *anomalousFilename = NULL, size_t anomBuffSize = 0, int model = 0);

	virtual std::string Hash() const;
	virtual std::string GetName() const;

	virtual FACC GetVoxelStepSize();
	virtual void SetVoxelStepSize(FACC stepSize);
	/**
	* @name	SetOutputSlices
	* @brief	Sets the output path for pictures of the voxelized solvent.
	*
	*
	* @param[in]	bool bOutput True if file will be written, false otherwise.
	* @param[in]	std::string outPath The output path where files will be written. If it does not exist,
	it will be created.
	*/
	virtual void SetOutputSlices(bool bOutput, std::string outPath = "");
	/**
	* @name	SetSolventOnlyCalculation
	* @brief	Sets a flag that determines whether to calculate the atomic form factor contribution of the PDB.
	* @param[in]	bool bOnlySol If set to true, the amplitude contribution of the atoms will not be calculated.
	*/
	virtual void SetSolventOnlyCalculation(bool bOnlySol);
	virtual void SetFillHoles(bool bFillHole);

	virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL,
		void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);

	/**
	* Fills and calculates the solvent space based on the step size and the solvent electron density
	* @param stepSize The size of the steps to be taken in nm
	* @param solED The electron density of the solvent whose amplitude will be subtracted
	* @param outerSolvED The electron density of the solvation layer whose amplitude will be added
	* @param soRad The radius (in nm) of the solvent whose amplitude will be subtracted
	* @return PDB_ERRS OK, sans errors
	**/
	virtual PDB_READER_ERRS CalculateSolventAmp(FACC stepSize, FACC solED, FACC outerSolvED, FACC solRad, FACC atmRadDiffIn = 0.0);

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	void PreCalculate(VectorXd& p, int nLayers);

	void dummyprogress();

	// GPU calculating functions
	virtual bool SetModel(Workspace& workspace);

	virtual bool SetParameters(Workspace& workspace, const double *params, unsigned int numParams);

	virtual bool ComputeOrientation(Workspace& workspace, float3 rotation);

	virtual bool CalculateGridGPU(GridWorkspace& workspace);

	virtual bool SetModel(GridWorkspace& workspace);

	virtual bool ImplementedHybridGPU();

	virtual bool SavePDBFile(std::ostream &output);

	virtual bool AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs);

	virtual bool GetHasAnomalousScattering();
protected:
	/**
	* Finds the farthest coordinates (plus atom radius) of the PDB
	* @return PDB_ERRS OK, sans errors (NO_ATOMS_IN_FILE)
	**/
	virtual PDB_READER_ERRS FindPDBRanges();
	/**
	* Using found range, create real space solvent volume
	* @return PDB_ERRS OK, sans errors
	**/
	virtual	PDB_READER_ERRS AllocateSolventSpace();

	/**
	* Method that returns a scalar representing the atomic form factor based on
	* the tables of the International Tables of X-ray Crystallography Vol IV, Table 2.2B.
	* @param q The scaler q for which the form factor is to be calculated. Units: A^{-1}
	* @param elem An index representing the atom/ion for which the form factor is to be
	*			calculated. Range: [0-207]
	* @return FACC The amplitude of the atomic form factor of elem at q
	**/
	virtual FACC atomicFF(FACC q, int elem);
	void WriteEigenSlicesToFile(string filebase);

	virtual void PrepareParametersForGPU(
		std::vector<float4>& solCOM, std::vector<int4>& solDims,
		std::vector<float4>& oSolCOM, std::vector<int4>& outSolDims);

protected:

	Eigen::Matrix<float, Eigen::Dynamic, 3> atomLocs;
	Eigen::Matrix<float, Eigen::Dynamic, 1> uniqueAffs;
	Eigen::ArrayXi uniqueIonsIndices;
	Eigen::ArrayXf uniqueIonRads;
	Eigen::ArrayXi numberOfIonsPerIndex;

	int bitwiseCalculationFlags;

	atomicFFCalculator affCalculator;

	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz);
	virtual std::complex<FACC> calcAmplitude(int indqx, int indqy, int indqz);
	virtual void initialize();
	void CalculateSolventSpace();
	void ReduceSolventSpaceToIrregularBoxes(std::vector<fIdx>& boxCOM, std::vector<idx>& boxDims, int designation, int mark_used_as);
	void MarkVoxelsNeighboringAtoms(FACC solventRad, SolventSpace::ScalarType from, SolventSpace::ScalarType to, int ignoreIndex = -1);
	void Floodfill3D(int i, int j, int k, int from, int to);
	void MarkLayerOfSolvent(FACC radius, SolventSpace::ScalarType type, SolventSpace::ScalarType neighbor, SolventSpace::ScalarType totype, SolventSpace::ScalarType fromType);

	/**
	* Method that returns a scalar representing the atomic form factor based on
	* the tables of the International Tables of X-ray Crystallography Vol IV, Table 2.2B.
	* @param q The scaler q for which the form factor is to be calculated. Units: A^{-1}
	* @param elem An index representing the atom/ion for which the form factor is to be
	*			calculated. Range: [0-207]
	* @return FACC The amplitude of the atomic form factor of elem at q
	**/
	//virtual FACC atomicFF(FACC q, int elem, double background = 0.0);
};

class EXPORTED_BE AmpGridAmplitude : public Amplitude, public IGPUGridCalculable {
protected:
	std::string fileHeader;
	std::string hash;
	
	Grid* originalGrid;

	virtual PDB_READER_ERRS ReadAmplitudeHeaderFromFile(std::string fileName, std::stringstream& header);

	virtual	PDB_READER_ERRS ReadAmplitudeFromStream(std::istream& readFile);
	virtual PDB_READER_ERRS ReadAmplitudeFromFile(std::string fileName);
	virtual PDB_READER_ERRS ReadAmplitudeFromAmpjFile(std::string fileName);
	virtual PDB_READER_ERRS ReadAmplitudeFromBuffer(const char *buffer, size_t buffSize);

public:
	AmpGridAmplitude(string filename);
	AmpGridAmplitude(const char *buffer, size_t buffSize);

	virtual ~AmpGridAmplitude();

	virtual std::string Hash() const;

	virtual std::string GetName() const;

	void PreCalculate(VectorXd& p, int nLayers);

	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz){
		if (status != PDB_OK)
			return std::complex<FACC>(-double(status), -double(status));

		return std::complex<FACC>(-999.0, -999.0);
	}

	virtual std::complex<FACC> getAmplitude(FACC qx, FACC qy, FACC qz);

	virtual void SetUseGrid(bool bUse) { bUseGrid = true; }

	// Nothing to calculate
	virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL,
		void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	virtual bool CalculateGridGPU(GridWorkspace& workspace);

	virtual bool SetModel(GridWorkspace& workspace);

	virtual bool ImplementedHybridGPU();
};

class EXPORTED_BE GeometricAmplitude : public Amplitude, public IGPUCalculable,
	public IGPUGridCalculable
{
protected:
#ifdef SYM_TREE_CHECK
	std::vector<Parameter> storedParams;
#endif
	FFModel *model;
	int modelLayers;
public:
	GeometricAmplitude(FFModel *mod);
	virtual ~GeometricAmplitude();
	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz) {
		Eigen::Vector3d vec(qx, qy, qz);
		return (*model).CalculateFF(vec, modelLayers);
	}

	virtual std::string Hash() const;
	virtual std::string GetName() const;

	/*virtual int GetNumberOfParameters();
	virtual std::vector<double> GetParameterValues();
	virtual void SetParameterValues(std::vector<Parameter>& pars);*/

	// Organize parameters from the parameter vector into the matrix and vector defined earlier.
	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	// Called before each series of q-calculations
	virtual void PreCalculate(VectorXd& p, int nLayers);

	virtual FFModel *GetGeometry() { return model; }

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	virtual bool SetModel(Workspace& workspace);

	virtual bool CalculateGridGPU(GridWorkspace& workspace);

	virtual bool SetModel(GridWorkspace& workspace);

	virtual bool ImplementedHybridGPU();

	virtual bool SetParameters(Workspace& workspace, const double *params, unsigned int numParams);

	virtual bool ComputeOrientation(Workspace& workspace, float3 rotation);

	virtual void CorrectLocationRotation(double& x, double& y, double& z,
		double& alpha, double& beta, double& gamma);
};

typedef double F_TYPE;
class atomicFFCalculator; // Forward declaration
/**
A test class that reads a pdb file, calculates the intensity using debye's
formula and reports performance
**/
class EXPORTED_BE DebyeCalTester : public IModel {
public:
	PDBReader::PDBReaderOb<F_TYPE> *pdb;
	Eigen::Array<F_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> atmFFcoefs;

	//Eigen::ArrayXXd distances; //< The 0.5 * N^2 - N distances of all the atoms. Is of N^2 dimension.

	atomicFFCalculator* _aff_calculator;

	bool bUseGPU;
	int kernelVersion;

	progressFunc progFunc;
	void *progArgs;

	Eigen::Matrix3Xf sortedLocations;

	//////////////////////////////////////////////////////////////////////////
	// Methods
	//////////////////////////////////////////////////////////////////////////

	DebyeCalTester() { bUseGPU = true; kernelVersion = 2; initialize(); }
	DebyeCalTester(bool bGPU, int kernelVersion_ = 2) { bUseGPU = bGPU;  kernelVersion = kernelVersion_;  initialize(); }
	virtual ~DebyeCalTester();

	virtual void initialize();

	virtual F_TYPE atomicFF(F_TYPE q, int elem);

	virtual PDB_READER_ERRS LoadPDBFile(string filename, int model = 0);

	virtual void SetStop(int *stop);

	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	virtual void PreCalculate(VectorXd& p, int nLayers);



	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void *progressArgs = NULL);

	virtual VectorXd CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void *progressArgs = NULL);

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai);

	virtual double Calculate(double q, int nLayers, VectorXd& p);


	virtual VectorXd GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p);

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	virtual bool GetHasAnomalousScattering();


};

class EXPORTED_BE AnomDebyeCalTester : public DebyeCalTester {
public:
	AnomDebyeCalTester() : DebyeCalTester() {}
	AnomDebyeCalTester(bool bGPU) : DebyeCalTester(bGPU, 4) { }
	AnomDebyeCalTester(bool bGPU, int kernelVersion) : DebyeCalTester(bGPU, kernelVersion) { }

	virtual std::complex<F_TYPE> anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime);

	virtual void PreCalculate(VectorXd& p, int nLayers);

	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual VectorXd CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void *progressArgs = NULL);
};



/**
Parameters:
Orientation iterations: uint64
Default grid size:	uint
Use grid by default: 1.0 iff true
Epsilon size: double
qMax: double
**/
class EXPORTED_BE DomainModel : public IModel, public ICeresModel, public ISymmetry /* For the sub-amplitude methods */ {
protected:
	std::vector<Amplitude*> _amps;
	std::vector<VectorXd> _ampParams;
	std::vector<int> _ampLayers;

	bool only_scale_changed;
	VectorXd previousParameters;
	std::string _previous_hash;
	ArrayXd _previous_intensity;
	ArrayXd _previous_q_values;

	bool bDefUseGrid, bDefUseGPU;
	int gridSize;
	unsigned long long oIters;
	OAMethod_Enum orientationMethod;

	double eps;
	double qMax;
	double qMin;
	progressFunc progFunc;
	void *progArgs;

#ifdef GAUSS_LEGENDRE_INTEGRATION
	ArrayXd theta_, wTheta, phi_, wPhi;
#endif

	PDB_READER_ERRS state;

	FACC CalculateIntensity(FACC q, FACC epsi, unsigned int seed, uint64_t iterations);

	virtual double Calculate(double q, int nLayers, VectorXd& p);

	// For adaptive 2D Gauss-Legendre integration
	double GaussLegPhi(double q, double st, double ct, double pMin, double pMax, double epsilon, int64_t maxDepth, int64_t minDepth);
	double GaussLegTheta(double q, double tMin, double tMax, double epsilon, int64_t maxDepth, int64_t minDepth);
	double GaussKron2DSphereRecurs(double q, double epsilon, int64_t maxDepth, int64_t minDepth);

	// Closed 2D Cubic Spline
	// 	PDB_READER_ERRS CubicSplineSphereVector(const std::vector<double>& q, std::vector<FACC>& res);
	// 	double CubicSplineSphere(double q);
	// 	void CubicSplineSphere(double qMin, double qMax, VectorXd &qPoints);

	// The global "should we stop" variable
	int *pStop;

public:
	DomainModel();
	~DomainModel();

	// So as to let pdbgen access it directly
	template <typename T>
	PDB_READER_ERRS CalculateIntensityVector(const std::vector<T>& Q,
		std::vector<T>& res,
		T epsi, uint64_t iterations);

	template<typename T>
	void setPreviousValues(std::vector<T> & res, const std::vector<T> & Q);

	template<typename T>
	PDB_READER_ERRS DefaultCPUCalculation(clock_t &aveBeg, const std::vector<T> & Q, std::vector<T> & res, T &epsi, std::vector<unsigned int> &seeds, const uint64_t &iterations, const double &cProgMax, const double &cProgMin, int &prog, clock_t &aveEnd, const clock_t &gridBegin);

	PDB_READER_ERRS gridComputation();

	template<typename T>
	PDB_READER_ERRS IntegrateLayersTogether(std::vector<unsigned int> &seeds, std::uniform_int_distribution<unsigned int> &sRan, std::mt19937 &seedGen, const std::vector<T> & Q, std::vector<T> & res, T &epsi, uint64_t &iterations, const double &cProgMax, const double &cProgMin, int &prog, clock_t &aveEnd, const clock_t &aveBeg, const clock_t &gridBegin);

	template <typename T>
	PDB_READER_ERRS PerformgGPUAllGridsMCOACalculations(const std::vector<T> &Q, std::vector<T> &res, uint64_t iterations, T epsi, clock_t &aveEnd, clock_t aveBeg, clock_t gridBegin);

	template <typename T>
	PDB_READER_ERRS PerformGPUHybridComputation(clock_t &gridBegin, const std::vector<T> &Q, clock_t &aveBeg, std::vector<T> &res, T epsi, uint64_t iterations, clock_t &aveEnd);



	std::string Hash() const;
	/**
	* @name	GetUseGridWithChildren
	* @brief	This method determines if from this point down in the model tree all models are set to use a grid.
	* @ret True iff this model and all children are set to use a grid/lookup table (bUseGrid == true)
	*/
	bool GetUseGridWithChildren() const;

	// ISymmetry implementers
	virtual int GetNumSubAmplitudes();
	virtual Amplitude *GetSubAmplitude(int index);
	virtual void SetSubAmplitude(int index, Amplitude *subAmp);
	virtual void AddSubAmplitude(Amplitude *subAmp);
	virtual void RemoveSubAmplitude(int index);
	virtual void ClearSubAmplitudes();
	virtual void GetSubAmplitudeParams(int index, VectorXd& params, int& nLayers);
	virtual bool Populate(const VectorXd& p, int nLayers);
	virtual unsigned int GetNumSubLocations();
	virtual LocationRotation GetSubLocation(int index);

	// IModel implementers
	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	virtual void SetStop(int *stop);

	virtual void PreCalculate(VectorXd& p, int nLayers);


	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void *progressArgs = NULL);

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai);

	virtual VectorXd GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p);

	// ICeresModel implementers
	virtual bool CalculateVectorForCeres(const double* qs, double const* const* p,
		double* residual, int points);
	virtual void SetInitialParamVecForCeres(VectorXd* p, std::vector<int> &mutIndices);
	virtual void GetMutatedParamVecForCeres(VectorXd* p);
	virtual void SetMutatedParamVecForCeres(const VectorXd& p);

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	virtual bool SavePDBFile(std::ostream &output);
	virtual bool AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs);
	
	virtual bool GetHasAnomalousScattering();

	template <typename T>
	void AverageIntensitiesBetweenLayers(const std::vector<T> &relevantQs, std::vector<T> &reses, size_t layerInd, FACC epsi, unsigned int seed, uint64_t iterations);
};

template<class FLOAT_TYPE>
Eigen::Matrix<FLOAT_TYPE, 3, 3, 0, 3, 3> EulerD(Radian theta, Radian phi, Radian psi) {
	FLOAT_TYPE ax, ay, az, c1, c2, c3, s1, s2, s3;
	ax = theta;
	ay = phi;
	az = psi;
	c1 = cos(ax); s1 = sin(ax);
	c2 = cos(ay); s2 = sin(ay);
	c3 = cos(az); s3 = sin(az);

	// Tait-Bryan angles X1Y2Z3 (x-alpha, y-beta, z-gamma)
	Eigen::Matrix<FLOAT_TYPE, 3, 3, 0, 3, 3> rot;
	rot << c2*c3, -c2*s3, s2,
		c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1,
		s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2;
	return rot;
}

template<class FLOAT_TYPE>
Eigen::Matrix<FLOAT_TYPE, 3, 3, 0, 3, 3> EulerD(Degree theta, Degree phi, Degree psi) {
	return EulerD<FLOAT_TYPE>(Radian(theta), Radian(phi), Radian(psi));
}


template<class FLOAT_TYPE>
void GetEulerAngles(Eigen::Quaternion<FLOAT_TYPE>& qt, Radian& theta, Radian& phi, Radian& psi) {

	qt.normalize();
	Eigen::Vector3f euAng = qt.toRotationMatrix().eval().eulerAngles(0, 1, 2);
	theta = Radian(euAng(0));
	phi = Radian(euAng(1));
	psi = Radian(euAng(2));
}

#endif	//__AMPLITUDE_H