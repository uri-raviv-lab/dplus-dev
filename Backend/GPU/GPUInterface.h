#ifndef __GPUINTERFACE
#define __GPUINTERFACE

#include <vector_types.h>
#include <vector>

#include "Common.h" // After vector_types.h to minimize warnings


// Forward declaration
class IGPUCalculator;

struct Workspace
{
	IGPUCalculator *parent;

	int gpuID;

	void *stream;	


	// Inputs
	float2 *d_angles;  // 1D, size: numAngles
	float  *d_qPoints; // 1D, size: numQ
	unsigned int numAngles, numQ;
	size_t anglePitch;
	float4 *d_translations, *h_translations;
	size_t maxTranslations;

	// Actual workspace
	double2 *d_work;   // 2D, size: numAngles x numQ (pitched)
	double2 *d_amp;
	size_t workPitch;

	// PDB stuff
	float4 *d_atomLocs, *d_rotAtomLocs;
	float *d_affs;
	unsigned int *h_atomsPerIon;
	size_t numAtoms,  maxNumAtoms;
	size_t numCoeffs, maxNumCoeffs;
	float solED;

	// Processed data and outputs
	double *d_transamp, *d_intensity;
	int *d_intensityIndices; // Necessary for thrust
	size_t ampPitch;


	Workspace() : parent(NULL), gpuID(0), stream(NULL), d_angles(NULL), 
		d_qPoints(NULL), numAngles(0), numQ(0), d_translations(NULL), h_translations(NULL), maxTranslations(0), d_amp(NULL), d_work(NULL),
		d_atomLocs(NULL), d_rotAtomLocs(NULL), d_affs(NULL), h_atomsPerIon(NULL),
		numAtoms(0), maxNumAtoms(0), numCoeffs(0), maxNumCoeffs(0), solED(0.0), d_transamp(NULL),
		d_intensity(NULL), d_intensityIndices(NULL), ampPitch(0) {}
};

// This class is kept in order to keep the state of the GPU
// calculator (so that several computational jobs can be done
// concurrently without running over each other's memory)

class IGPUCalculator
{
 public:
    virtual ~IGPUCalculator() {}

    // Should be called once per workspace
    virtual bool Initialize(int gpuID, const float2 *angles, size_t numAngles,
							const float *qPoints, size_t numQ,
							size_t maxNumAtoms, size_t maxNumCoeffs, 
							size_t maxTranslations, Workspace& res) = 0;

	
	virtual bool TranslateWorkspace(Workspace& workspace, float3 *translations, unsigned int numTrans) = 0;
    
    virtual bool ComputeIntensity(Workspace *workspaces, unsigned int numWorkspaces, double *outData) = 0;

	virtual bool FreeWorkspace(Workspace& workspace) = 0;
};

class IGPUCalculable
{
public:
	virtual bool SetModel(Workspace& workspace) = 0;

	virtual bool SetParameters(Workspace& workspace, const double *params, 
							   unsigned int numParams) = 0;
	
	// Function should fill workspace.d_work
	virtual bool ComputeOrientation(Workspace& workspace, float3 rotation) = 0;

	virtual void CorrectLocationRotation(double& x, double& y, double& z, 
										 double& alpha, double& beta, double& gamma) {}
};

typedef IGPUCalculator *(*gpucalculator_t)();

static inline int ComputeGridSize(int total, int blockSize)
{
	return ((total % blockSize == 0) ? (total / blockSize) : (total / blockSize + 1));
}

// Forward declaration
class IGPUGridCalculator;

typedef IGPUGridCalculator *(*gpuGridcalculator_t)();

// TODO Change the pointer based arrays to std::vector<T>
struct GridWorkspace
{
	IGPUGridCalculator *calculator;	///< Momma!
	
	GridWorkspace *parent;	///< Momma!

	OAMethod_Enum intMethod; ///< The integration method used for orientation averaging

	int gpuID;				///< The ID of the GPU card kernels will be run on

	void *memoryStream;		///< The cuda stream used for memory transfers
	void *computeStream;	///< The cuda stream used for calculation kernels
	
	// Actual workspace
	double *d_amp;	///< The amplitude of the node
	double *d_int;	///< The interpolation coefficients of d_amp. Not necessarily existent
	
	//GridWorkspace *parentWorkspace;	///< The workspace of a parent. Relevant for the children of symmetries.
	GridWorkspace *children;		///< The workspaces of the children. Relevant for symmetries.
	int numChildren;				///< The number of children of a symmetry.

	// Geometric model parameters
	int nParams;		///< The number of parameters in params. NLP is nParams / nLayers.
	int nLayers;		///< The number of layers of a layer based geometric model. NLP is nParams / nLayers.
	float *d_params;	///< The parameters of the model. 
	int nExtras;		///< The number of extra parameters in a geometric model
	float *d_extraPrm;	///< The extra parameters of a geometric model

	// Symmetry parameters
	int		symSize  ;	///< The size of d_symLocs and d_symRots. For a Manual symmetry, the number of layers; for a space filling symmetry, 1.
	float4 *d_symLocs;	///< The locations of a symmetry
	float4 *d_symRots;	///< The rotations of a symmetry
	double scale;		///< The scale by which to multiply the amplitude.

/*
	float* d_constantMemory;		///< A pointer to the devices constant memory
	int beginConstantMemoryOffset;	///< The offset from which point the model can use global memory. Will be non-zero if another model has used some constant memory
	int totalConstantMemoryOffset;	///< The total offset of the used constant memory. The models used constant memory will be (totalConstantMemoryOffset - beginConstantMemoryOffset)
*/

	// Parameters needed for the calculation of PDB amplitudes
	float4* d_pdbLocs;	///< The locations (Cartesian) of all the atoms
	int numAtoms;		///< The total number of atoms (e.g. Acetic acid [CH3COOH] would be 8)
	int numUniqueIons;	///< The total number of unique ions (e.g. Acetic acid assuming [A-][H+] would be 5 (H, C, O, O-, H+) if we assume the entire charge is on one oxygen)
	float* d_affCoeffs;	///< The coefficients (a_i, b_i) used to calculate the atomic form factors
	float* d_affs;		///< The calculated atomic form factors. Should be of dimensions [numUniquIons*qLayers]
	const int* atomsPerIon;	///< DO NOT ALLOCATE!! Should point to the PDBReaderObj member atomsPerIon.
	int solventType;	///< The index of the solvent type. Is casted to an int from enum ATOM_RADIUS_TYPE. Valid values are: 1, VDW; 2, EMP; 3, CALC; 4, Dummy atom.
	float solventED;	///< The electron density of the bulk solvent
	float* d_atmRad;	///< The radii used for calculating the solvent amplitude. Of length numUniqueIons.
	float voxSize;		///< The length in [UNITS? nm? Angstrom?] of one real-space voxel
	float4* d_SolCOM;	///< The coordinates of the centers of mass of the voxels of the displaced solvent.
	int4* d_SolDims;	///< The dimensions of the voxels of the displaced solvent. The number is the number of step sizes per dimension.
	int numSolVoxels;	///< The total number of displaced solvent voxels.
	float outSolED;		///< The electron density of the solvation layer
	float4* d_OSolCOM;	///< The coordinates of the centers of mass of the voxels of the outer solvent.
	int4* d_OSolDims;	///< The dimensions of the voxels of the outer solvent. The number is the number of step sizes per dimension.
	int numOSolVoxels;	///< The total number of outer solvent voxels.
	bool bSolOnly;		///< A flag that when raised, disables the contribution from the atoms themselves.
	int kernelComb;		///< The bitwise combination of calculation kernels that need to be run. Values: 0x01, atomic formfactors; 0x02, dummy atom solvent; 0x04, voxelized solvent; 0x08, voxelized solvation.

	// Rotation and translation parameters
	int numRotations;	///< The number of rotations of the model.
	float4 *d_rots;		///< The actual rotations of the model. Should be allocated to be numRotations long.
	int *numTrans;		///< The list of translations lengths for each translation.
	int *d_nTrans;		///< The device pointer that should be copied from *numTrans once ready to be used.
	float4 **trans;		///< The actual translations of each rotation.
	float4 *d_trns;		///< The device pointer that should contain all the translations. Effectively the double pointer (**) that will require a map (*d_nTrans) to determine which translations belong to which rotations.

	// Jacobian Grid parameters
	long long totalsz;	///< Twice the total number of voxels in the JacobianGrid (one for Real, one for Imag)
	int phiDivs, thetaDivs;
	int qLayers;		///< The number of layers in the Jacobian grid. Includes the extra layers.
	float qMax, stepSize;
	float qMin;	///< Not to be used yet...

	float *qVec;	///< The values of q for which the intensity is to be calculated.
	int numQ;		///< The number of q values in qVec.


	GridWorkspace() : parent(NULL), intMethod(OA_MC), gpuID(0), memoryStream(NULL),
		computeStream(NULL), d_amp(NULL), d_int(NULL), children(NULL), numChildren(0),
		nParams(0), nLayers(0), d_params(NULL), nExtras(0), d_extraPrm(NULL),
		symSize(0), d_symLocs(NULL), d_symRots(NULL), scale(1.0),
		//d_constantMemory(NULL), beginConstantMemoryOffset(0), totalConstantMemoryOffset(0),
		d_pdbLocs(NULL), numAtoms(0), d_affCoeffs(NULL), d_affs(NULL), atomsPerIon(0), d_atmRad(NULL),
		d_SolCOM(NULL), d_SolDims(NULL), numSolVoxels(0), outSolED(0.0), d_OSolCOM(NULL),
		d_OSolDims(NULL), numOSolVoxels(0), bSolOnly(false), kernelComb(0),
		numRotations(0), d_rots(NULL), numTrans(NULL), d_nTrans(NULL), trans(NULL), d_trns(NULL),
		totalsz(0), phiDivs(0), thetaDivs(0), qLayers(0), qMax(0.0), stepSize(0.0), qMin(0.0),
		qVec(NULL), numQ(0)
	{}

	void setNumberOfAmplitudes(int num) {}
};

class IGPUGridCalculator
{
public:
	// Should be called once per workspace
	virtual bool Initialize(int gpuID, const std::vector<float>& qPoints,
		long long totalSize, int thetaDivisions, int phiDivisions, int qLayers,
		double qMax, double stepSize, GridWorkspace& res) = 0;

	virtual bool ComputeIntensity(std::vector<GridWorkspace> &workspaces,
									double *outData, double epsi, long long iterations,
									progressFunc progfunc = NULL, void *progargs = NULL, float progmin = 0., float progmax = 0., int *pStop = NULL) = 0;

	virtual bool SetNumChildren(GridWorkspace &workspace, int numChildren) = 0;

	virtual bool AddRotations(GridWorkspace &workspace, std::vector<float4> &rotations) = 0;

	virtual bool AddTranslations(GridWorkspace &workspace, int rotationIndex, std::vector<float4> &translations) = 0;

	virtual bool FreeWorkspace(GridWorkspace& workspace) = 0;
};

class IGPUGridCalculable
{
public:
	virtual bool CalculateGridGPU(GridWorkspace& workspace) = 0;

	virtual bool SetModel(GridWorkspace& workspace) = 0;

	/// Returns true iff the model has been fully implemented to run as hybrid.
	/// Should be true for all models, except GeometricModels that have not been
	/// implemented.
	virtual bool ImplementedHybridGPU() = 0;
};

#endif