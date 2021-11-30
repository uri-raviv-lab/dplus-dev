#ifndef __FITTING_H
#define __FITTING_H

#include "Common.h"

#include "Eigen/Core"
#include "Eigen/Geometry"

using namespace Eigen;

#include <vector>

class IModel;

// Cons: Constraint vector description
//       index - The layer index to compare to (-1 if doesn't exist)
//		 link  - The layer index to link a value to (-1 if doesn't exist)
//       num   - The value to compare to (0.0 if doesn't exist)
typedef struct ConsStruct {
	VectorXi index, link;
	VectorXd num;

	ConsStruct(int m) : index(VectorXi::Constant(m, -1)), link(VectorXi::Constant(m, -1)), num(VectorXd::Zero(m)) {}
	ConsStruct()  {}  
} cons;

class ModelFitter {
protected:
    // Model to fit
	IModel *FitModel;
        
	// Parameters
	FittingProperties props;
	const std::vector<double> &x, &y, &mult, &add;
	VectorXd interimResY;
	VectorXd sqWeights;
	VectorXd params;
	MatrixXd J;
	double mse;
	const VectorXi& paramMut;
	int mutables, nParams, nLayers;
	cons *p_min, *p_max;

	// Should we stop the fitting process now?
	int bStop;

	bool error;

public:
	/** 
	 * Creates and initializes a new fitter.
	 * Parameters:
	 * FitModel - The corresponding model function to fit to
	 * datax, datay - The data to fit to (must have same size)
	 * factor - Multiplicative factor to multiply the model by (must have the same size as datax)
	 * bg - Additive factor to add to the model (must have the same size as datax)
	 * fitWeights - Weights for the Chi-Squared error function
	 * p - Parameter vector
	 * pmut - Parameter mutability vector (0 = immutable, 1 = mutable)
	 * pMin, pMax - The constraint vector pointers
	 * layers - The number of layers in the data (input for the FitModel function)
	 *
	 * Output: The 'p' parameter vector is modified with each fitting iteration
	 */
 ModelFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
             const std::vector<double>& datay,
             const std::vector<double>& factor, 
             const std::vector<double>& bg,
             const std::vector<double>& fitWeights, VectorXd& p,
             const VectorXi& pmut, cons *pMin, cons *pMax,
             int layers);

	bool GetError() const { return error; }

	virtual ~ModelFitter() {}

	// Evaluates the optimization function (WSSR/R^2)
	virtual double Evaluate(const VectorXd& params);

	// Returns true if constraints are met, false if they don't. Also modifies parameter
	// vector for linked parameters.
	virtual bool EnforceConstraints(VectorXd& inoutParams);

	// Performs one fitting iteration, modifies the parameter vector and returns the current WSSR
	virtual double FitIteration() = 0;

	virtual void GetFittingErrors(std::vector<double>& paramErrors, std::vector<double>& modelErrors) = 0;

	virtual VectorXd GetResult() const { return params; }

	virtual VectorXd GetInterimRes() {return interimResY;}

	/************************************************************************/
	/* Stops the fitting process                                            */
	/************************************************************************/
	virtual void Stop() { bStop = 1; }

private:
	// Will not assign to other ModelFitters
	void operator=(ModelFitter& rhs) {}
};

// Levenberg-Marquardt Nonlinear Geometry Fitting Class
class LMFitter : public ModelFitter {
protected:
	// LM Coefficients
	MatrixXd alpha;
	VectorXd beta;

	double lambda, curWssr;

	// Calculates the current fitting coefficients and returns the WSSR
	double CalculateCoefficients(VectorXd& p, MatrixXd& alphaMat, VectorXd& betaVec);

public:
	// Creates and initializes a new Levenberg-Marquardt fitter
	LMFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
                 const std::vector<double>& datay,
                 const std::vector<double>& factor, 
				 const std::vector<double>& bg,
                 const std::vector<double>& fitWeights, VectorXd& p,
                 const VectorXi& pmut, cons *pMin, cons *pMax,
                 int layers) : 
		ModelFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, pMin, pMax, layers) {            
			if(mutables == 0) {
				error = true; 
				return;
			}

			alpha = MatrixXd::Zero(mutables, mutables);
			beta  = VectorXd::Zero(mutables);
            lambda = 0.001;
            
            curWssr = CalculateCoefficients(params, alpha, beta);
	}

	// Performs one fitting iteration, modifies the parameter vector and returns the current WSSR
	virtual double FitIteration();

	virtual void GetFittingErrors(std::vector<double>& paramErrors, std::vector<double>& modelErrors);

	virtual ~LMFitter() { }
};

#endif // __FITTING_H
