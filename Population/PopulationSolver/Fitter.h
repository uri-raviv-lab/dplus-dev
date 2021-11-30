#ifndef __FITTING_H
#define __FITTING_H

#pragma once
#include "PopModel.h"
#include "annoying.h"
#include "Eigen/Core"
#include "Eigen/Geometry"

using namespace Eigen;

#include <vector>

// Cons: Constraint vector description
//       index - The layer index to compare to (-1 if doesn't exist)
//		 link  - The layer index to link a value to (-1 if doesn't exist)
//       num   - The value to compare to (0.0 if doesn't exist)
//typedef struct ConsStruct {
//	ArrayXi index, link;
//	ArrayXd num;
//
//	ConsStruct(int m) : index(ArrayXi::Constant(m, -1)), link(ArrayXi::Constant(m, -1)), num(ArrayXd::Zero(m)) {}
//	ConsStruct()  {}  
//} cons;

class EXPORTED PopulationFitter {
protected:
	ArrayXd x, y, resX, resY;
	std::vector<double> tmpX;
	ArrayXXd intensities;

	ArrayXd sqWeights;
	VectorXd params;
	ArrayXXd J;
	ArrayXd *modelEr;
	ArrayXd *err, *errY;
	double mse, curRsq;
	VectorXi paramMut;
	int mutables, nParams, nLayers;
	cons *p_min, *p_max;

	PopModel *pm;

	bool error;

	
public:
	/** 
	 * Creates and initializes a new fitter.
	 * Parameters:
	 * FitModel - The corresponding model function to fit to
	 * datay - The data to fit to
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
 PopulationFitter(
             const ArrayXd& datay,
             const ArrayXd& datax,
			 const string expression,
			 const ArrayXXd& curvesX,
			 const ArrayXXd& curvesY,
			 ArrayXd& p,	// This will end up being the population sizes
             const ArrayXi& pmut, cons *pMin, cons *pMax,
			 ArrayXd& paramErrors,
			 ArrayXd& modelErrors) :
        y(datay), x(datax), intensities(curvesY), paramMut(pmut),
            nParams(p.size()), p_min(pMin), p_max(pMax), params(p)
        {
			error = false;

			modelEr = &modelErrors;
			(*modelEr).resize(2);

            mutables = 0;
            for(int i = 0; i < nParams; i++)
                if(paramMut[i])
                    mutables++;

			pm = new PopModel(curvesX, curvesY, expression, p);

			pm->GetModXY(x, y);
			sqWeights = y.abs().sqrt() + 1.0;

			for(int i =0; i < x.size(); i++)
				tmpX.push_back(x(i));
	}

	bool GetError() const { return error; }

	virtual ~PopulationFitter() {
	if(pm)
		delete pm;
	}

	// Performs one fitting iteration, modifies the parameter vector and returns the current WSSR
	virtual double FitIteration() = 0;

	virtual void calcErrors() = 0;

	virtual ArrayXd GetResult() const { return params; }

protected:
	double Calc(int ind);

	ArrayXd Calc();

	ArrayXd Derivative(VectorXd param, int ai);
	
private:
	// Will not assign to other PopulationFitter
	void operator=(PopulationFitter& rhs) {}
};

class EXPORTED LMPopFitter : public PopulationFitter {
protected:
	// LM Coefficients
	MatrixXd alpha;
	VectorXd beta;

	double lambda, curWssr;

	// Calculates the current fitting coefficients and returns the WSSR
	double CalculateCoefficients(VectorXd& p, MatrixXd& alphaMat, VectorXd& betaVec);
	
public:
	// Creates and initializes a new Levenberg-Marquardt fitter
	LMPopFitter(const ArrayXd& datay,
				 const ArrayXd& datax,
				 const string expression,
				 const ArrayXXd& curvesX, const ArrayXXd& curvesY, ArrayXd& p,
                 const ArrayXi& pmut, cons *pMin, cons *pMax,
				 ArrayXd& paramErrors,
				 ArrayXd& modelErrors,
                 int layers) : 
		PopulationFitter(datay, datax, expression, curvesX, curvesY, p, pmut, pMin, pMax, paramErrors, modelErrors) {
            // TODO: Modify CreateFitter, Parameter and Fitter constructor to support PD
			if(mutables == 0) {
				error = true; 
				return;
			}

			alpha = ArrayXXd::Zero(mutables, mutables);
			beta  = ArrayXd::Zero(mutables);
			err = &paramErrors;
			errY = &modelErrors;

            lambda = 0.001;
            
            curWssr = CalculateCoefficients(params, alpha, beta);
	}

	// Performs one fitting iteration, modifies the parameter vector and returns the current WSSR
	virtual double FitIteration();

	virtual void calcErrors();

	virtual ~LMPopFitter() { }
};


#endif // __FITTING_H
