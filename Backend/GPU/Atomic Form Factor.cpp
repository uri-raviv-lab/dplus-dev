#include "Atomic Form Factor.h"

#include <Eigen/Core>

class internalAtomicFF
{
public:
	internalAtomicFF(
		int bitCombination, int numAtoms, int numUnIons,
		const float* coeffs, const int *atomsPerIon)
			: m_bitCombination(bitCombination), m_numAtoms(numAtoms), m_numUnIons(numUnIons),
			m_coeffs(coeffs), m_atomsPerIon(atomsPerIon),
			as(4, numUnIons), bs(4, numUnIons), cs(numUnIons),
			m_solED(0), m_solventOnly(false)
	{

		for (size_t i = 0; i < numUnIons; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				as(j, i) = coeffs[i + 2 * j * numUnIons];
				bs(j, i) = coeffs[i + (2 * j + 1) * numUnIons];
			}
			cs(i) = coeffs[i + (9 - 1) * numUnIons];
		}

	}

	void SetAnomalousFactors(float2* anomFacs)
	{
		m_anomFactors = Eigen::Map<Eigen::ArrayXcf>((std::complex<float>*)(anomFacs), m_numAtoms);
	}
	void SetSolventED(float solED, float c1, float *ionRads, bool solventOnly = false)
	{
		m_solED = solED;
		m_c1 = c1;
		m_solventOnly = solventOnly;
		
		m_ionRads = Eigen::Map<Eigen::ArrayXf>(ionRads, m_numUnIons);
		auto numberOfIonsEach = Eigen::Map<const Eigen::ArrayXi>(m_atomsPerIon, m_numUnIons);

#ifdef USE_SINGLE_ATOM_VOLUMES
		m_rm = (m_ionRads * numberOfIonsEach.cast<float>()).sum() / float(numberOfIonsEach.sum());
#else
		m_rm = cbrt((m_ionRads.cube() * numberOfIonsEach.cast<float>()).sum() / float(numberOfIonsEach.sum()));
#endif
		printf("rm = %f\n", m_rm);

	}

	void GetAllUniqueAFFs(Eigen::Ref<Eigen::ArrayXf, 0, Eigen::InnerStride<> > mapToAffs, float q)
	{
		const float sqq = (q * q / (100.0f * 157.913670417429737901351855998f));
		Eigen::ArrayXf uniqueAffsArr = ((-sqq * bs).exp() * as).colwise().sum().transpose() + cs;
		//Eigen::Map<Eigen::ArrayXf> mapToAffs(uniqueAffs, m_numAtoms);
		Eigen::ArrayXf solventContrast = solventContribution(q);

		mapToAffs = ((m_bitCombination & CALC_ATOMIC_FORMFACTORS) ? uniqueAffsArr : Eigen::ArrayXf::Constant(uniqueAffsArr.size(), 0.f))
			- solventContrast;
	}

	void GetAllAFFs(float* allAffs, float q)
	{
		const float sqq = (q * q / (100.0f * 157.913670417429737901351855998f));
		Eigen::ArrayXf uniqueAffs = ((-sqq * bs).exp() * as).colwise().sum().transpose() + cs;
		Eigen::Map<Eigen::ArrayXf> mapToAffs(allAffs, m_numAtoms);
		Eigen::ArrayXf solventContrast = solventContribution(q);

		int initialPos = 0;
		for (size_t j = 0; j < m_numUnIons; j++)
		{
			mapToAffs.segment(initialPos, m_atomsPerIon[j]).setConstant(
				((m_bitCombination & CALC_ATOMIC_FORMFACTORS) ? uniqueAffs(j) : 0)
				- solventContrast(j));
			initialPos += m_atomsPerIon[j];
		}

	}

	void GetAllAFFs(float2* allAffs, float q)
	{
		const float sqq = (q * q / (100.0f * 157.913670417429737901351855998f));
		Eigen::ArrayXf uniqueAffs = ((-sqq * bs).exp() * as).colwise().sum().transpose() + cs;
		Eigen::Map<Eigen::ArrayXcf> mapToAffs((std::complex<float>*)allAffs, m_numAtoms);
		Eigen::ArrayXf solventContrast = solventContribution(q);


		int initialPos = 0;
		for (size_t j = 0; j < m_numUnIons; j++)
		{
			mapToAffs.segment(initialPos, m_atomsPerIon[j]).setConstant(
				((m_bitCombination & CALC_ATOMIC_FORMFACTORS) ? uniqueAffs(j) : 0)
				- solventContrast(j));
			initialPos += m_atomsPerIon[j];
		}

		if ((m_bitCombination & CALC_ANOMALOUS) && m_anomFactors.size() == m_numAtoms)
		{
			mapToAffs += m_anomFactors;
		}

	}

	Eigen::ArrayXi GetAnomalousIndices()
	{
		Eigen::ArrayXi nonZeros = (m_anomFactors != std::complex<float>(0, 0)).cast<int>();
		int numNonZeros = nonZeros.sum();

		Eigen::ArrayXi indices(numNonZeros);
		int nextIndex = 0;
		for (int i = 0; i < numNonZeros; i++)
		{
			while (nonZeros(nextIndex) == 0)
				nextIndex++;
			indices(i) = nextIndex;
			nextIndex++;
		}

		return indices;
	}

	Eigen::ArrayXcf GetSparseAnomalousFactors()
	{
		const Eigen::ArrayXi indices = GetAnomalousIndices();
		Eigen::ArrayXcf factors(indices.size());

		for (int i = 0; i < indices.size(); i++)
			factors(i) = m_anomFactors(indices(i));

		return factors;
	}


	void GetQMajorAFFMatrix(float* theMatrix, int numberOfQValues, float stepSize, float qMin = 0.f)
	{
		Eigen::Map<Eigen::ArrayXXf> affMatrix(theMatrix, numberOfQValues, m_numUnIons);

		for (int i = 0; i < numberOfQValues; i++)
			GetAllUniqueAFFs(affMatrix.row(i), qMin + stepSize * i);
	}

	int GetNumUniqueIon()
	{
		return m_numUnIons;
	}

	int GetNumAtomsPerIon(int index)
	{
		if (index >= m_numUnIons)
			return -1;
		return m_atomsPerIon[index];
	}

	bool HasSomethingToCalculate()
	{
		return
			m_bitCombination & CALC_ATOMIC_FORMFACTORS ||
			m_bitCombination & CALC_DUMMY_SOLVENT
			;
	}

	int GetBitCombination()
	{
		return m_bitCombination;
	}

protected:
	Eigen::ArrayXf solventContribution(float q)
	{
		Eigen::ArrayXf solventContrast;
		if (m_bitCombination & CALC_DUMMY_SOLVENT)
		{
#ifdef USE_FRASER
			solventContrast = 5.5683279968317084528 * m_ionRads.cube() * (-(m_ionRads.square() * (q*q) / 4.)).exp() * m_solED;
#elif defined(USE_SINGLE_ATOM_VOLUMES)
			solventContrast = 4.1887902047863909846 * m_ionRads.cube() * (-(m_ionRads.square() * (q*q) * 0.20678349696647)).exp() * m_solED;
			solventContrast *= m_c1 *m_c1 *m_c1 * exp(
				(/*-(4\pi/3)^1.5 * (4\pi)^-1*/ -0.6822178052976590 * q * q * m_rm * m_rm * (m_c1 * m_c1 - 1.)) );
#else
			//Cdummyatom(q) = ro*Vc*exp(-(q ^ 2 * Vaverage ^ (2 / 3) / 4Pi));
			solventContrast = 4.1887902047863909846 * m_ionRads.cube() * exp(-((m_rm*m_rm) * (q*q) * 0.2067834969664667))* m_solED;
			solventContrast *= m_c1 *m_c1 *m_c1 * exp(
				(/*-(4\pi/3)^1.5 * (4\pi)^-1*/ -0.6822178052976590 * q * q * m_rm * m_rm * (m_c1 * m_c1 - 1.)));
#endif
		}
		else
		{
			solventContrast.setZero(m_numUnIons);
		}

		return solventContrast;
	}


protected:
	bool m_solventOnly;
	float m_solED;
	float m_c1;
	float m_rm;
	int m_numUnIons, m_numAtoms;
	const float* m_coeffs;
	const int *m_atomsPerIon;
	int m_bitCombination;
	Eigen::ArrayXXf as, bs;
	Eigen::ArrayXf cs;
	Eigen::ArrayXf m_ionRads;
	Eigen::ArrayXcf m_anomFactors;

};

atomicFFCalculator::~atomicFFCalculator()
{
	delete intern;
}

atomicFFCalculator::atomicFFCalculator(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int *atomsPerIon)
{
	intern = NULL;
	Initialize(bitCombination, numAtoms, numUnIons, coeffs, atomsPerIon);
}

atomicFFCalculator::atomicFFCalculator()
{
	intern = NULL;
}

void atomicFFCalculator::GetAllAFFs(float* allAffs, float q, void* anoms)
{
	if (!intern) return;

	if (anoms)
		intern->SetAnomalousFactors((float2*)anoms);
	intern->GetAllAFFs(allAffs, q);
}

void atomicFFCalculator::GetAllAFFs(float2* allAffs, float q, void* anoms)
{
	if (!intern) return;

	if (anoms)
		intern->SetAnomalousFactors((float2*)anoms);
	intern->GetAllAFFs(allAffs, q);
}

void atomicFFCalculator::SetSolventED(float solED, float c1, float *ionRads, bool solventOnly /*= false*/)
{
	if (!intern) return;
	intern->SetSolventED(solED, c1, ionRads, solventOnly);
}

void atomicFFCalculator::SetAnomalousFactors(float2* anomFacs)
{
	if (!intern) return;
	intern->SetAnomalousFactors(anomFacs);
}

void atomicFFCalculator::Initialize(int bitCombination, int numAtoms, int numUnIons, const float* coeffs, const int *atomsPerIon)
{
	if (intern) delete intern;

	intern = new internalAtomicFF(bitCombination, numAtoms, numUnIons, coeffs, atomsPerIon);
}

int atomicFFCalculator::GetAnomalousIndices(int* indices /*= NULL*/)
{
	if (indices)
	{
		Eigen::ArrayXi tmp = intern->GetAnomalousIndices();
		Eigen::Map<Eigen::ArrayXi>(indices, tmp.size()) = tmp;
		return tmp.size();
	}

	return intern->GetAnomalousIndices().size();
}

void atomicFFCalculator::GetSparseAnomalousFactors(float2* allAffs)
{
	if (allAffs)
	{
		Eigen::ArrayXcf tmp = intern->GetSparseAnomalousFactors();
		Eigen::Map<Eigen::ArrayXcf>(((std::complex<float>*)(allAffs)), tmp.size()) = tmp;
	}
}

void atomicFFCalculator::GetQMajorAFFMatrix(float* theMatrix, int numberOfQValues, float stepSize, float qMin /*= 0.f*/)
{
	intern->GetQMajorAFFMatrix(theMatrix, numberOfQValues, stepSize, qMin);
}

int atomicFFCalculator::GetNumUniqueIon()
{
	return intern->GetNumUniqueIon();
}

void atomicFFCalculator::GetAllUniqueAFFs(float* uniqueAffs, float q)
{
	intern->GetAllUniqueAFFs(Eigen::Map<Eigen::ArrayXf>(uniqueAffs, intern->GetNumUniqueIon()), q);
}

int atomicFFCalculator::GetNumAtomsPerIon(int index)
{
	return intern->GetNumAtomsPerIon(index);
}

bool atomicFFCalculator::HasSomethingToCalculate()
{
	return intern->HasSomethingToCalculate();
}

int atomicFFCalculator::GetBitCombination()
{
	return intern->GetBitCombination();
}

