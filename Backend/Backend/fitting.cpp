#include <cstdlib>
#include <limits>
#include <sstream>
#include <iostream>	// cout
#include <stdio.h>	// fprintf

#include "fitting.h"

#include <Eigen/LU>
#include "Eigen/SVD"

using namespace Eigen;

#include "Model.h"
#include "mathfuncs.h"
#include "Statistics.h"

// To fix the std::numeric_limits<double>::min/max
#undef min
#undef max

/*#define NOMINMAX
#include <windows.h>*/ //USE FOR MESSAGEBOX TESTING

/**
 * @file fitting.cpp
 * @brief Implements the core fitting algorithms and classes for nonlinear least-squares model fitting
 *        and error estimation in the backend.
 *
 * The fitting module is responsible for:
 *  - Providing the main fitting engine (ModelFitter, LMFitter) for optimizing model parameters
 *    to best fit experimental data.
 *  - Supporting weighted and unweighted fitting, log-scale fitting, and various constraint types.
 *  - Calculating parameter and model errors using the Jacobian and covariance matrices.
 *  - Enforcing parameter constraints and supporting parameter linking during optimization.
 *  - Supporting both accurate (SVD-based) and fast (LU-based) linear system solutions for fitting.
 *
 * Key Concepts:
 *  - ModelFitter: Base class for fitting a model to data, managing parameters, constraints, and evaluation.
 *  - LMFitter: Implements the Levenberg-Marquardt algorithm for nonlinear least-squares fitting,
 *    including error estimation and iterative parameter updates.
 *  - Constraints: Supports parameter bounds, linking, and custom constraints via the cons structure.
 *  - FittingProperties: Controls fitting options such as accuracy, weighting, log-scale, and iteration limits.
 *  - Uses Eigen for efficient linear algebra (SVD, LU, matrix/vector operations).
 *
 * Fields:
 *  - IModel* FitModel:
 *      Pointer to the model being fitted.
 *  - FittingProperties props:
 *      Fitting options (accuracy, weighting, log-scale, iteration limits, etc.).
 *  - std::vector<double> x, y:
 *      Experimental data (x: independent variable, y: dependent variable).
 *  - std::vector<double> mult, add:
 *      Multiplicative and additive background factors for the model.
 *  - VectorXd params:
 *      Current parameter vector for the model.
 *  - VectorXi paramMut:
 *      Flags indicating which parameters are mutable during fitting.
 *  - int nParams:
 *      Number of parameters in the model.
 *  - int nLayers:
 *      Number of layers (for multilayer models).
 *  - cons *p_min, *p_max:
 *      Structures holding parameter minimum and maximum constraints, including linking.
 *  - int mutables:
 *      Number of mutable parameters.
 *  - VectorXd sqWeights:
 *      Squared weights for weighted fitting.
 *  - volatile int bStop:
 *      Stop flag for interrupting fitting (thread-safety).
 *  - bool error:
 *      Error state flag for the fitter.
 *  - MatrixXd J:
 *      Jacobian matrix of partial derivatives.
 *  - MatrixXd alpha:
 *      Alpha matrix (J'J) for the Levenberg-Marquardt algorithm.
 *  - VectorXd beta:
 *      Beta vector (J'(y-f(b))) for the Levenberg-Marquardt algorithm.
 *  - double lambda:
 *      Damping parameter for the Levenberg-Marquardt algorithm.
 *  - double curWssr:
 *      Current weighted sum of squared residuals.
 *  - double mse:
 *      Mean squared error of the fit.
 *  - VectorXd interimResY:
 *      Stores the most recent model evaluation result.
 *
 * Main Methods:
 *  - ModelFitter (constructor): Initializes the fitter with model, data, weights, constraints, and properties.
 *  - Evaluate: Computes the fit quality (WSSR or 1 - R²) for a given parameter set.
 *  - EnforceConstraints: Applies parameter bounds and linking, returning false if violated.
 *  - LMFitter::FitIteration: Performs a single Levenberg-Marquardt iteration, updating parameters and damping.
 *  - LMFitter::GetFittingErrors: Computes parameter and model errors from the covariance matrix and Jacobian.
 *  - LMFitter::CalculateCoefficients: Calculates the Jacobian, alpha/beta matrices, and fit residuals.
 *
 * Threading and Safety:
 *  - Not inherently thread-safe; designed for per-job, per-fit use.
 *  - Supports external stop signals for interruptible fitting.
 *
 * Error Handling:
 *  - Sets internal error flags on matrix failures or constraint violations.
 *  - Returns negative values or sets error state on failure.
 *
 * Dependencies:
 *  - Relies on Eigen for matrix operations.
 *  - Uses Model, Statistics, and mathfuncs for model evaluation and statistics.
 *
 * See fitting.h for class and method declarations.
 */

ModelFitter::ModelFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
			const std::vector<double>& datay,
			const std::vector<double>& factor, 
			const std::vector<double>& bg,
			const std::vector<double>& fitWeights, VectorXd& p,
			const VectorXi& pmut, cons *pMin, cons *pMax,
			int layers) :
				FitModel(model), props(fp), x(datax), y(datay),
				mult(factor), add(bg), params(p), paramMut(pmut),
				nParams(params.size()), nLayers(layers), p_min(pMin), 
				p_max(pMax), bStop(0), error(false) { 			

	mutables = 0;
	for(int i = 0; i < nParams; i++)
		if(paramMut[i])
			mutables++;

	if(FitModel)
		FitModel->SetStop(&bStop);

	sqWeights = VectorXd::Zero(fitWeights.size());
	for(int i = 0; i < (int)fitWeights.size(); i++)
		sqWeights[i] = fitWeights[i] * fitWeights[i];
}

double ModelFitter::Evaluate(const VectorXd& params)
{
	std::vector<double> model (x.size(), 0.0);
	VectorXd pcopy = params;

	interimResY = FitModel->CalculateVector(x, nLayers, pcopy);
	for(int i = 0; i < interimResY.size(); i++)
		model[i] = interimResY[i];

	if(props.wssrFitting)
		return WSSR(y, model, props.logScaleFitting);
	else
		return 1.0 - RSquared(y, model, props.logScaleFitting);
}

bool ModelFitter::EnforceConstraints(VectorXd& p)
{
	// Link the Linked parameters 
	if (p_max) {
		for(int b = 0; b < p_max->link.size(); b++) {
			if(p_max->link[b] >= 0 && p_max->link[b] != b)
				p[b] = p[p_max->link[b]];
		}
	}

	// Constraints ("Penalty Method")
	for (int k = 0; k < nParams; k++) {
		if(p_min && p_max) {
			if(!(fabs(p_min->num[k] - p_max->num[k]) <= 1e-12)) {
				if (((p[k] < 0.0 && p_min->num[k] >= 0) || 
					(p_min->num[k] > p[k]) ||
					(p_max->num[k] < p[k]) ||  
					(p_min->index[k] >= 0 && p[p_min->index[k]] > p[k])   ||
					(p_max->index[k] >= 0 && p[p_max->index[k]] < p[k]) )  && paramMut[k]  ) {
						return false;
				}
			}

		}
	}

	return true;
}

void LMFitter::GetFittingErrors(std::vector<double>& paramErrors, std::vector<double>& modelErrors) {
	
	//VectorXd tmp = alpha.inverse().diagonal();
	lambda = 0.0;

	FitIteration();

	MatrixXd bl, bltmp(J.rows(), mutables);
	VectorXd blah;
	VectorXd tmp = alpha.diagonal();

	paramErrors.clear();

	for(int i = 0; i < tmp.size(); i++) 
		paramErrors.push_back(sqrt(fabs(tmp[i])));



	int j = 0;
	for (int i = 0; i < J.cols(); i++) {
		if(!J.col(i).isZero())
			bltmp.col(j++) = J.col(i);
	}

	modelErrors.clear();

	bl = bltmp * alpha * bltmp.transpose();
	blah = bl.diagonal();

	for(int i = 0; i < blah.size(); i++)
		modelErrors.push_back(sqrt(fabs(blah[i])));
	
}

double LMFitter::FitIteration() {
	VectorXd curParams, curBeta;
	MatrixXd curAlpha;

	if(mutables == 0 || GetError())
		return 0.0;

	// We need to solve the linear equation system curAlpha*delta = curBeta, where:
	// alpha = J'J                       where J is the Jacobian Matrix and J' is its transpose
	// curAlpha = J'J + lambda*diag(J'J)
	// curBeta  = J'(y-f(b))             where b is the initial guess vector, and y is the data

    curAlpha = alpha;
	curAlpha.diagonal() = alpha.diagonal() * (1.0 + lambda);
       
	curBeta = beta;


	// Solving the equations using SVD and Moore-Penrose Pseudoinverse
	if(props.accurateFitting) {
		try {
			// I don't know if using ComputeFullU/V would be better 
			JacobiSVD<MatrixXd> svd(curAlpha, ComputeThinU  | ComputeThinV );

			MatrixXd sigma = MatrixXd::Zero(mutables, mutables);
			sigma.diagonal() = svd.singularValues();
			
			for(int i = 0; i < mutables; i++)
				if(sigma(i, i) != 0.0)
					sigma(i, i) = 1.0 / sigma(i, i);

			curAlpha = svd.matrixV() * sigma * svd.matrixU().transpose();
			curBeta = curAlpha * curBeta;
		} catch(...) {
			// Invalid matrix values
			error = true;
			return -1.0;
		}

	} else { // Solving the equations using LU decomposition
                Eigen::FullPivLU<MatrixXd> lu(curAlpha);
		VectorXd sol = curAlpha.lu().solve(curBeta);
		
		if(sol.size() > 0) {
			curAlpha = lu.inverse();
			curBeta = sol;
		} else {
			lambda *= 10.0;
			return curWssr;
		}
	}

	// Testing the results of the equation solution
	curParams = params;
	
	int j = 0;
    for (int i = 0; i < nParams; i++)
        if (paramMut[i]) 
			curParams[i] = params[i] + curBeta[j++];

	if(fabs(lambda) < 1e-14)
		alpha = curAlpha;

	double wssr = CalculateCoefficients(curParams, curAlpha, curBeta);

	// If better, keep them, otherwise, don't, and increase damping factor
    if (wssr < curWssr && wssr >= 0.0) {
        lambda *= 0.1;
        curWssr = wssr;

		alpha = curAlpha;
        beta = curBeta;
        params = curParams;
    } 
    else
		lambda *= 10.0;

	return curWssr;
}

double LMFitter::CalculateCoefficients(VectorXd& p, MatrixXd& alphaMat,
                                       VectorXd& betaVec) {
    double wssr = 0.0, mean = 0.0, sstotal = 0.0;
    
	VectorXd dy = VectorXd::Zero(x.size());
    MatrixXd dyda = MatrixXd::Zero(x.size(), nParams);

    betaVec.setZero();
    
    for (int i = 0; i < mutables; i++) 
        for (int j = 0; j <= i; j++) 
            alphaMat(i, j) = 0.0;

	// Enforce the constraints
	if(!EnforceConstraints(p))
		return -1.0;

	std::vector <double> ResY = MachineResolution(x, y, props.resolution);

    // Calculating the R-Squared coefficients
    if(!props.wssrFitting) {
		mean = Mean(ResY);	

		for(int i = 0; i < (int)y.size(); i++)
			sstotal += (ResY[i] - mean) * (ResY[i] - mean);
	
    }

	// Create copies for the parameter vector and the number of layers for
	// this iteration
	VectorXd guess = p;
	int guessLayers = nLayers;

    /*if(edp)
		MessageBoxA(NULL, FitModel->debugModelParams().c_str(), "Hey!", NULL);*/
    
    int size = (int)x.size();

    // 1st tier of parallelization

	VectorXd pDummy = FitModel->CalculateVector(x, guessLayers, guess);

    for (int i = 0; i < size; i++) {
        double cury;
		if(error)
			continue;
        
        if(bStop) {
			error = true;
			continue;
		}
		
		cury = pDummy(i) * mult[i] + add[i];

		if(props.logScaleFitting)
			cury = log10(cury);
        
		dy[i] = ResY[i] - cury;
    }

    // Calculate trial alpha and beta coefficients according to the
    // gradient
    int j = 0;
    for (int l = 0; l < nParams; l++) {
        if (paramMut[l]) {
			// Partial derivative calculation
			dyda.col(l) = FitModel->Derivative(x, p, nLayers, l);

            for(int i = 0; i < size; i++) {
                
				double wt = dyda(i, l) / sqWeights[i];
                
                int k = 0;
                for (int m = 0; m <= l; m++) {
                    if(paramMut[m]) { 
                        alphaMat(j, k) += wt * dyda(i, m); 
                        k++;
                    }
                }
                betaVec[j] += dy[i] * wt;
            }
            j++;
        }
    }
    
	J = dyda;

    // Makes matrix symmetric
    for (int i = 1; i < mutables; i++)
        for (int k = 0; k < i; k++) 
            alphaMat(k, i) = alphaMat(i, k);

    // Calculate WSSR
	mse = 0.0;
	for(int i = 0; i < size; i++) {
		mse += (dy[i] * dy[i]);
		if(props.wssrFitting)
			wssr += (dy[i] * dy[i]) / sqWeights[i];
        else
			wssr += (dy[i] * dy[i])  / sstotal;
	}
    mse /= double(size);

	interimResY = VectorXd::Zero(ResY.size());
	for(int w = 0; w < int(ResY.size()); w++)
		interimResY(w) = ResY[w] - dy[w];
    return wssr;
}
