#ifndef ATOMIC_FORMS_DEFINES
#define ATOMIC_FORMS_DEFINES

//#define SOLVENT_ONLY					(0x0000)
#define CALC_ATOMIC_FORMFACTORS			(1 << 0)
#define CALC_DUMMY_SOLVENT				(1 << 1)
#define CALC_ANOMALOUS					(1 << 2)
#define CALC_VOXELIZED_SOLVENT			(1 << 3)
#define CALC_VOXELIZED_OUTER_SOLVENT	(1 << 4)

class baseInternalAtomicFF
{
public:

	virtual void GetAllUniqueAFFs(Eigen::Ref<Eigen::ArrayXf, 0, Eigen::InnerStride<> > mapToAffs, float q) = 0;

	virtual void GetAllAFFs(float* allAffs, float q) = 0;

	virtual void GetAllAFFs(float2* allAffs, float q) = 0;


	void SetAnomalousFactors(float2* anomFacs);
	void SetSolventED(float solED, float c1, float* ionRads, bool solventOnly = false);

	Eigen::ArrayXi GetAnomalousIndices();

	Eigen::ArrayXcf GetSparseAnomalousFactors();


	void GetQMajorAFFMatrix(float* theMatrix, int numberOfQValues, float stepSize, float qMin = 0.f);

	int GetNumUniqueIon();

	int GetNumAtomsPerIon(int index);

	bool HasSomethingToCalculate();

	int GetBitCombination();

protected:
	Eigen::ArrayXf solventContribution(float q);

protected:
	bool m_solventOnly;
	float m_solED;
	float m_c1;
	float m_rm;
	int m_numUnIons, m_numAtoms;
	const float* m_coeffs;
	const int* m_atomsPerIon;
	int m_bitCombination;
	Eigen::ArrayXXf as, bs;
	//Eigen::ArrayXf cs;
	Eigen::ArrayXf m_ionRads;
	Eigen::ArrayXcf m_anomFactors;

};



#endif 