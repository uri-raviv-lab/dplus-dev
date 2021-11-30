#ifndef __DEFITTING_H
#define __DEFITTING_H

#include "fitting.h"
#include "Externals/DESolver.h"

// TODO::Fitters Make a GUI setting
// TODO::Fitters Make the strategy a setting too
#define FIT_POPULATION 100

// Differential Evolution Solver Class
class DEModelSolver : public DESolver {
protected:
	int dataSize, nLayers;
	const std::vector<double> &x, &y;
	IModel *FitModel;
	FittingProperties props;
public:
	// Creates and initializes a new Differential Evolution model fitter
	DEModelSolver(const FittingProperties& fp, int nParams, int layers, VectorXd& params, int fitPopulation, 
				  IModel *model, const std::vector<double>& datax, 
				  const std::vector<double>& datay) : DESolver(nParams, fitPopulation, params),
					props(fp), nLayers(layers), FitModel(model), x(datax), y(datay) { 
		dataSize = (int)x.size();
	}

	double EnergyFunction(VectorXd& trial);

	//virtual void calcErrors();

private:
	// Will not assign to other DEModelSolvers
	void operator=(DEModelSolver rhs) {}
};

// Differential Evolution Geometry Fitting Class
class DEFitter : public ModelFitter {
protected:
	DEModelSolver *solver;

	friend class DEModelSolver;
public:
	// Creates and initializes a new Differential Evolution fitter
	DEFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
                 const std::vector<double>& datay,
                 const std::vector<double>& factor, 
				 const std::vector<double>& bg,
                 const std::vector<double>& fitWeights, VectorXd& p,
                 const VectorXi& pmut, cons *pMin, cons *pMax,
                int layers) : 
	ModelFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, pMin, pMax, layers), solver(NULL) {
            
			if(mutables == 0) return;


			// TODO::Fitters Fit only mutable parameters
			VectorXd pmin = VectorXd::Zero(nParams);
			VectorXd pmax = VectorXd::Zero(nParams);

			if(pMin)
				pmin = pMin->num;
			if(pMax)
				pmax = pMax->num;
			else {
				for(int i = 0; i < pmax.size(); i++)
					pmax[i] = 1.0e7;
			}

			solver = new DEModelSolver(props, nParams, nLayers, params, FIT_POPULATION, model, datax, datay);
			solver->Setup(pmin, pmax, stRand2Exp, 0.9, 1.0);

			solver->IncrementalSolveBegin();
		}

	virtual double FitIteration() { return solver->IncrementalSolveGeneration(); }

	virtual VectorXd GetResult() const { return solver->Solution(); }

	virtual void GetFittingErrors(std::vector<double>& paramErrors, std::vector<double>& modelErrors) {};

	virtual ~DEFitter() { delete solver; }
};


#endif
