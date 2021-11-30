#ifndef ATOMIC_FORM_FACTOR_H__
#define ATOMIC_FORM_FACTOR_H__


//#include <math_constants.h>
#include <vector_functions.h>
//#include <Eigen/Core>

//#define SOLVENT_ONLY					(0x0000)
#define CALC_ATOMIC_FORMFACTORS			(1 << 0)
#define CALC_DUMMY_SOLVENT				(1 << 1)
#define CALC_ANOMALOUS					(1 << 2)
#define CALC_VOXELIZED_SOLVENT			(1 << 3)
#define CALC_VOXELIZED_OUTER_SOLVENT	(1 << 4)
class internalAtomicFF;

/**
Calculates the atomic form factors for the set of ions/atoms initialized with.

TODO: Add bfactors(Debye Waller) so that we can delete DW factors from kernels
      and other functions. The DW calculation belongs here anyway.
**/
class atomicFFCalculator
{
public:
	atomicFFCalculator();
	atomicFFCalculator(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int *atomsPerIon);
	~atomicFFCalculator();
	void Initialize(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int *atomsPerIon);
	void SetSolventED(float solED, float c1, float *ionRads, bool solventOnly = false);
	void SetAnomalousFactors(float2* anomFacs);

	/**
	q is in nm^{-1}
	**/
	void GetAllAFFs(float* allAffs, float q, void* anoms = NULL);
	/**
	q is in nm^{-1}
	**/
	void GetAllAFFs(float2* allAffs, float q, void* anoms = NULL);

	void GetSparseAnomalousFactors(float2* allAffs);
	
	/// Returns the number of indices (to be allocated by the caller).
	int GetAnomalousIndices(int* indices = NULL);

	// Fills theMatrix with (numberOfQValues X #uniqueIons) in a q-major memory layout
	void GetQMajorAFFMatrix(float* theMatrix, int numberOfQValues, float stepSize, float qMin = 0.f);

	int GetNumUniqueIon();

	void GetAllUniqueAFFs(float* uniqueAffs, float q);

	int GetNumAtomsPerIon(int index);

	bool HasSomethingToCalculate();

	int GetBitCombination();

protected:
	internalAtomicFF* intern;
};

#endif
