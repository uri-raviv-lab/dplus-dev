// Differential Evolution Solver Class
// Based on algorithms developed by Dr. Rainer Storn & Kenneth Price
// Written By: Lester E. Godwin
//             PushCorp, Inc.
//             Dallas, Texas
//             972-840-0208 x102
//             godwin@pushcorp.com
// Created: 6/8/98
// Last Modified: 6/8/98
// Tal Ben-Nun, 11/9/09:
// -Converted to Eigen
// -Added immutable parameters
// Revision: 1.0

#if !defined(_DESOLVER_H)
#define _DESOLVER_H

#include "Eigen/Core"
using namespace Eigen;

#define stBest1Exp			0
#define stRand1Exp			1
#define stRandToBest1Exp	2
#define stBest2Exp			3
#define stRand2Exp			4
#define stBest1Bin			5
#define stRand1Bin			6
#define stRandToBest1Bin	7
#define stBest2Bin			8
#define stRand2Bin			9

class DESolver;

typedef void (DESolver::*StrategyFunction)(int);

class DESolver
{
public:
	DESolver(int dim,int popSize, VectorXd& initialGuess);
	~DESolver(void);
	
	// Setup() must be called before solve to set min, max, strategy etc.
	void Setup(const VectorXd& min, const VectorXd& max,int deStrategy,
				double diffScale,double crossoverProb);

	// Solve() returns true if EnergyFunction() returns true.
	// Otherwise it runs maxGenerations generations and returns false.
	virtual bool Solve(int maxGenerations);

	// New functions for interactive fitting by Tal
	// SolveGeneration returns the current best energy
	virtual void IncrementalSolveBegin();
	virtual double IncrementalSolveGeneration();

	// EnergyFunction must be overridden for problem to solve
	// testSolution[] is nDim array for a candidate solution
	// setting bAtSolution = true indicates solution is found
	// and Solve() immediately returns true.
	virtual double EnergyFunction(VectorXd& testSolution) = 0;
	
	int Dimension(void) { return(nDim); }
	int Population(void) { return(nPop); }

	// Call these functions after Solve() to get results.
	double Energy(void) { return(bestEnergy); }
	VectorXd Solution(void) { return(bestSolution); }

	int Generations(void) { return(generations); }

	
protected:
	void SelectSamples(int candidate,int *r1,int *r2=0,int *r3=0,
												int *r4=0,int *r5=0);
	double RandomUniform(double min,double max);

	int nDim;
	int nPop;
	int generations;

	int currentGeneration;

	int strategy;
	StrategyFunction calcTrialSolution;
	double scale;
	double probability;

	double trialEnergy;
	double bestEnergy;

	VectorXd trialSolution;
	VectorXd bestSolution, initialSolution;
	VectorXd popEnergy;
	MatrixXd population;

private:
	void Best1Exp(int candidate);
	void Rand1Exp(int candidate);
	void RandToBest1Exp(int candidate);
	void Best2Exp(int candidate);
	void Rand2Exp(int candidate);
	void Best1Bin(int candidate);
	void Rand1Bin(int candidate);
	void RandToBest1Bin(int candidate);
	void Best2Bin(int candidate);
	void Rand2Bin(int candidate);
};

#endif // _DESOLVER_H