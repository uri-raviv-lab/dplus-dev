#ifndef __ELECTRONPDBAMPLITUDE_H
#define __ELECTRONPDBAMPLITUDE_H


#include "../PDBReaderLib/ElectronPDBReaderLib.h"
#include "../GPU/electron Atomic Form Factor.h"
#include "Amplitude.h"

class EXPORTED_BE electronPDBAmplitude : public Amplitude, public IGPUCalculable,
	public IGPUGridCalculable {
protected:
	friend class DomainModel;

	ElectronPDBReader::electronPDBReaderOb<float> pdb;
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
	virtual ~electronPDBAmplitude();
	electronPDBAmplitude();
	electronPDBAmplitude(string filename, bool bCenter, string anomalousFilename = "", int model = 0);
	electronPDBAmplitude(const char* buffer, size_t buffSize, const char* filenm, size_t fnSize,
		bool bCenter, const char* anomalousFilename = NULL, size_t anomBuffSize = 0, int model = 0);

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
		void* progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int* pStop = NULL);

	/**
	* Fills and calculates the solvent space based on the step size and the solvent electron density
	* @param stepSize The size of the steps to be taken in nm
	* @param solED The electron density of the solvent whose amplitude will be subtracted
	* @param outerSolvED The electron density of the solvation layer whose amplitude will be added
	* @param soRad The radius (in nm) of the solvent whose amplitude will be subtracted
	* @return PDB_ERRS OK, sans errors
	**/
	virtual PDB_READER_ERRS CalculateSolventAmp(FACC stepSize, FACC solED, FACC outerSolvED, FACC solRad, FACC atmRadDiffIn = 0.0);

	virtual void GetHeader(unsigned int depth, std::string& header);
	virtual void GetHeader(unsigned int depth, JsonWriter& writer);

	void PreCalculate(VectorXd& p, int nLayers);

	void dummyprogress();

	// GPU calculating functions
	virtual bool SetModel(Workspace& workspace);

	virtual bool SetParameters(Workspace& workspace, const double* params, unsigned int numParams);

	virtual bool ComputeOrientation(Workspace& workspace, float3 rotation);

	virtual bool CalculateGridGPU(GridWorkspace& workspace);

	virtual bool SetModel(GridWorkspace& workspace);

	virtual bool ImplementedHybridGPU();

	virtual bool SavePDBFile(std::ostream& output);

	virtual bool AssemblePDBFile(std::vector<std::string>& lines, std::vector<Eigen::Vector3f>& locs);

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
	virtual FACC electronAtomicFF(FACC q, int elem);
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

	electronAtomicFFCalculator electronAffCalculator;

	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz);
	virtual std::complex<FACC> calcAmplitude(int indqx, int indqy, int indqz);
	virtual void electronInitialize();
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

class EXPORTED_BE electronDebyeCalTester : public IModel {
public:
	ElectronPDBReader::electronPDBReaderOb<F_TYPE>* pdb;
	Eigen::Array<F_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> atmFFcoefs;

	//Eigen::ArrayXXd distances; //< The 0.5 * N^2 - N distances of all the atoms. Is of N^2 dimension.

	electronAtomicFFCalculator* _electronAff_calculator;

	bool bUseGPU;
	int kernelVersion;

	progressFunc progFunc;
	void* progArgs;

	Eigen::Matrix3Xf sortedLocations;

	//////////////////////////////////////////////////////////////////////////
	// Methods
	//////////////////////////////////////////////////////////////////////////

	electronDebyeCalTester() { bUseGPU = true; kernelVersion = 2; electronInitialize(); }
	electronDebyeCalTester(bool bGPU, int kernelVersion_ = 2) { bUseGPU = bGPU;  kernelVersion = kernelVersion_;  electronInitialize(); }
	virtual ~electronDebyeCalTester();

	virtual void electronInitialize();

	virtual F_TYPE electronAtomicFF(F_TYPE q, int elem);

	virtual PDB_READER_ERRS LoadPDBFile(string filename, int model = 0);

	virtual void SetStop(int* stop);

	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	virtual void PreCalculate(VectorXd& p, int nLayers);



	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void* progressArgs = NULL);

	virtual VectorXd CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void* progressArgs = NULL);

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai);

	virtual double Calculate(double q, int nLayers, VectorXd& p);


	virtual VectorXd GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p);

	virtual void GetHeader(unsigned int depth, std::string& header);
	virtual void GetHeader(unsigned int depth, JsonWriter& writer);

	virtual bool GetHasAnomalousScattering();


};

/**
A test class that reads a pdb file, calculates the intensity using debye's
formula and reports performance
**/

class EXPORTED_BE electronAnomDebyeCalTester : public electronDebyeCalTester {
public:
	electronAnomDebyeCalTester() : electronDebyeCalTester() {}
	electronAnomDebyeCalTester(bool bGPU) : electronDebyeCalTester(bGPU, 4) { }
	electronAnomDebyeCalTester(bool bGPU, int kernelVersion) : electronDebyeCalTester(bGPU, kernelVersion) { }

	virtual std::complex<F_TYPE> anomAtomicFF(F_TYPE q, int elem, F_TYPE fPrime, F_TYPE fPrimePrime);

	virtual void PreCalculate(VectorXd& p, int nLayers);

	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual VectorXd CalculateVectorGPU(const std::vector<double>& q, int nLayers, VectorXd& p, progressFunc progress = NULL, void* progressArgs = NULL);
};

#endif