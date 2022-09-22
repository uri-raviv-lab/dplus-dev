#ifndef ELECTRON_ATOMIC_FORM_FACTOR_H__
#define ELECTRON_ATOMIC_FORM_FACTOR_H__


//#include <math_constants.h>
#include <vector_functions.h>
//#include <Eigen/Core>
#include "atomicFormDefines.h"
class electronInternalAtomicFF;

/**
Calculates the atomic form factors for the set of ions/atoms initialized with.

TODO: Add bfactors(Debye Waller) so that we can delete DW factors from kernels
	  and other functions. The DW calculation belongs here anyway.
**/
class electronAtomicFFCalculator
{
public:
	electronAtomicFFCalculator();
	electronAtomicFFCalculator(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int* atomsPerIon);
	~electronAtomicFFCalculator();
	void Initialize(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int* atomsPerIon);
	void SetSolventED(float solED, float c1, float* ionRads, bool solventOnly = false);
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
	electronInternalAtomicFF* intern;
};

#endif
