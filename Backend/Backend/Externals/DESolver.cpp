#include <memory.h>
#include "DESolver.h"

#define RowVector(a,b)  (&a[b*nDim])
#define CopyVector(a,b) memcpy((a),(b),nDim*sizeof(double))

DESolver::DESolver(int dim,int popSize, VectorXd& initialGuess) :
					nDim(dim), nPop(popSize),
						generations(0), strategy(stRand2Exp),
					scale(0.7), probability(0.5), bestEnergy(0.0),
					currentGeneration(0)
{
	initialSolution = initialGuess;
	trialSolution = VectorXd::Zero(nDim);
	bestSolution  = VectorXd::Zero(nDim);
	popEnergy     = VectorXd::Zero(nPop);
	population	  = MatrixXd::Zero(nPop, nDim);

	return;
}

DESolver::~DESolver(void)
{
}

void DESolver::Setup(const VectorXd& min, const VectorXd& max,
					int deStrategy,double diffScale,double crossoverProb)
{
	int i;

	strategy	= deStrategy;
	scale		= diffScale;
	probability = crossoverProb;
	
	for (i=0; i < nPop; i++)
	{
		for (int j=0; j < nDim; j++)
			population(i,j) = RandomUniform(min[j],max[j]);

		popEnergy[i] = 1.0E20;
	}

	for (i=0; i < nDim; i++)
		bestSolution[i] = 0.0;

	switch (strategy)
	{
		case stBest1Exp:
			calcTrialSolution = &DESolver::Best1Exp;
			break;

		case stRand1Exp:
			calcTrialSolution = &DESolver::Rand1Exp;
			break;

		case stRandToBest1Exp:
			calcTrialSolution = &DESolver::RandToBest1Exp;
			break;

		case stBest2Exp:
			calcTrialSolution = &DESolver::Best2Exp;
			break;

		case stRand2Exp:
			calcTrialSolution = &DESolver::Rand2Exp;
			break;

		case stBest1Bin:
			calcTrialSolution = &DESolver::Best1Bin;
			break;

		case stRand1Bin:
			calcTrialSolution = &DESolver::Rand1Bin;
			break;

		case stRandToBest1Bin:
			calcTrialSolution = &DESolver::RandToBest1Bin;
			break;

		case stBest2Bin:
			calcTrialSolution = &DESolver::Best2Bin;
			break;

		case stRand2Bin:
			calcTrialSolution = &DESolver::Rand2Bin;
			break;
	}

	return;
}

bool DESolver::Solve(int maxGenerations)
{
	int generation;
	int candidate;

	bestEnergy = 1.0E20;

	for (generation=0;(generation < maxGenerations);generation++)
		for (candidate=0; candidate < nPop; candidate++)
		{
			(this->*calcTrialSolution)(candidate);
			trialEnergy = EnergyFunction(trialSolution);

			if (trialEnergy < popEnergy[candidate])
			{
				// New low for this candidate
				popEnergy[candidate] = trialEnergy;
				population.row(candidate) = trialSolution;

				// Check if all-time low
				if (trialEnergy < bestEnergy)
				{
					bestEnergy = trialEnergy;
					bestSolution = trialSolution;
				}
			}
		}

	generations = generation;
	return true;
}



void DESolver::IncrementalSolveBegin() {
	currentGeneration = 0;
	bestSolution = initialSolution;
	bestEnergy = EnergyFunction(bestSolution);
}

double DESolver::IncrementalSolveGeneration() {
	int candidate;
	bool bAtSolution;

	bAtSolution = false;

	for (candidate=0; candidate < nPop; candidate++)
	{
		// TODO::Fitters Fix
		(this->*calcTrialSolution)(candidate);
		//trialSolution(0) = population(candidate, 0);
		trialEnergy = EnergyFunction(trialSolution);

		if (trialEnergy < popEnergy[candidate])
		{
			// New low for this candidate
			popEnergy[candidate] = trialEnergy;
			population.row(candidate) = trialSolution;

			// Check if all-time low
			if (trialEnergy < bestEnergy)
			{
				bestEnergy = trialEnergy;
				bestSolution = trialSolution;
			}
		}
	}
	currentGeneration++;

	generations = currentGeneration;
	return bestEnergy;
}

void DESolver::Best1Exp(int candidate)
{
	int r1, r2;
	int n;

	SelectSamples(candidate,&r1,&r2);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; (RandomUniform(0.0,1.0) < probability) && (i < nDim); i++) 
	{
		trialSolution[n] = bestSolution[n]
							+ scale * (population(r1,n)
							- (population(r2,n)));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Rand1Exp(int candidate)
{
	int r1, r2, r3;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; (RandomUniform(0.0,1.0) < probability) && (i < nDim); i++) 
	{
		trialSolution[n] = population(r1,n)
							+ scale * (population(r2,n)
							- (population(r3,n)));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::RandToBest1Exp(int candidate)
{
	int r1, r2;
	int n;

	SelectSamples(candidate,&r1,&r2);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; (RandomUniform(0.0,1.0) < probability) && (i < nDim); i++) 
	{
		trialSolution[n] += scale * (bestSolution[n] - trialSolution[n])
							 + scale * (population(r1,n)
							 - population(r2,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Best2Exp(int candidate)
{
	int r1, r2, r3, r4;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3,&r4);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; (RandomUniform(0.0,1.0) < probability) && (i < nDim); i++) 
	{
		trialSolution[n] = bestSolution[n] +
							scale * (population(r1,n)
										+ population(r2,n)
										- population(r3,n)
										- population(r4,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Rand2Exp(int candidate)
{
	int r1, r2, r3, r4, r5;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3,&r4,&r5);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; (RandomUniform(0.0,1.0) < probability) && (i < nDim); i++) 
	{
		trialSolution[n] = population(r1,n)
							+ scale * (population(r2,n)
										+ population(r3,n)
										- population(r4,n)
										- population(r5,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Best1Bin(int candidate)
{
	int r1, r2;
	int n;

	SelectSamples(candidate,&r1,&r2);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; i < nDim; i++) 
	{
		if ((RandomUniform(0.0,1.0) < probability) || (i == (nDim - 1)))
			trialSolution[n] = bestSolution[n]
								+ scale * (population(r1,n)
											- population(r2,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Rand1Bin(int candidate)
{
	int r1, r2, r3;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; i < nDim; i++) 
	{
		if ((RandomUniform(0.0,1.0) < probability) || (i  == (nDim - 1)))
			trialSolution[n] = population(r1,n)
								+ scale * (population(r2,n)
												- population(r3,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::RandToBest1Bin(int candidate)
{
	int r1, r2;
	int n;

	SelectSamples(candidate,&r1,&r2);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; i < nDim; i++) 
	{
		if ((RandomUniform(0.0,1.0) < probability) || (i  == (nDim - 1)))
			trialSolution[n] += scale * (bestSolution[n] - trialSolution[n])
									+ scale * (population(r1,n)
												- population(r2,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Best2Bin(int candidate)
{
	int r1, r2, r3, r4;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3,&r4);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; i < nDim; i++) 
	{
		if ((RandomUniform(0.0,1.0) < probability) || (i  == (nDim - 1)))
			trialSolution[n] = bestSolution[n]
								+ scale * (population(r1,n)
											+ population(r2,n)
											- population(r3,n)
											- population(r4,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::Rand2Bin(int candidate)
{
	int r1, r2, r3, r4, r5;
	int n;

	SelectSamples(candidate,&r1,&r2,&r3,&r4,&r5);
	n = (int)RandomUniform(0.0,(double)nDim);

	trialSolution = population.row(candidate);
	for (int i=0; i < nDim; i++) 
	{
		if ((RandomUniform(0.0,1.0) < probability) || (i  == (nDim - 1)))
			trialSolution[n] = population(r1,n)
								+ scale * (population(r2,n)
											+ population(r3,n)
											- population(r4,n)
											- population(r5,n));
		n = (n + 1) % nDim;
	}

	return;
}

void DESolver::SelectSamples(int candidate,int *r1,int *r2,
										int *r3,int *r4,int *r5)
{
	if (r1)
	{
		do
		{
			*r1 = (int)RandomUniform(0.0,(double)nPop);
		}
		while (*r1 == candidate);
	}

	if (r2)
	{
		do
		{
			*r2 = (int)RandomUniform(0.0,(double)nPop);
		}
		while ((*r2 == candidate) || (*r2 == *r1));
	}

	if (r3)
	{
		do
		{
			*r3 = (int)RandomUniform(0.0,(double)nPop);
		}
		while ((*r3 == candidate) || (*r3 == *r2) || (*r3 == *r1));
	}

	if (r4)
	{
		do
		{
			*r4 = (int)RandomUniform(0.0,(double)nPop);
		}
		while ((*r4 == candidate) || (*r4 == *r3) || (*r4 == *r2) || (*r4 == *r1));
	}

	if (r5)
	{
		do
		{
			*r5 = (int)RandomUniform(0.0,(double)nPop);
		}
		while ((*r5 == candidate) || (*r5 == *r4) || (*r5 == *r3)
													|| (*r5 == *r2) || (*r5 == *r1));
	}

	return;
}

/*------Constants for RandomUniform()---------------------------------------*/
#define SEED 3
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double DESolver::RandomUniform(double minValue,double maxValue)
{
	long j;
	long k;
	static long idum;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	double result;

	if (iy == 0)
		idum = SEED;

	if (idum <= 0)
	{
		if (-idum < 1)
			idum = 1;
		else
			idum = -idum;

		idum2 = idum;

		for (j=NTAB+7; j>=0; j--)
		{
			k = idum / IQ1;
			idum = IA1 * (idum - k*IQ1) - k*IR1;
			if (idum < 0) idum += IM1;
			if (j < NTAB) iv[j] = idum;
		}

		iy = iv[0];
	}

	k = idum / IQ1;
	idum = IA1 * (idum - k*IQ1) - k*IR1;

	if (idum < 0)
		idum += IM1;

	k = idum2 / IQ2;
	idum2 = IA2 * (idum2 - k*IQ2) - k*IR2;

	if (idum2 < 0)
		idum2 += IM2;

	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = idum;

	if (iy < 1)
		iy += IMM1;

	result = AM * iy;

	if (result > RNMX)
		result = RNMX;

	result = minValue + result * (maxValue - minValue);
	return(result);
}
