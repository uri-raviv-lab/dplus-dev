#include "MCFitting.h"
#include <ctime>

static bool randInit = false;

/**
 * @file MCFitting.cpp
 * @brief Implements Monte Carlo initialization and Levenberg-Marquardt fitting for model parameter optimization.
 *
 * The MCFitting module is responsible for:
 *  - Providing a hybrid Monte Carlo and Levenberg-Marquardt (LM) fitting algorithm for model parameter optimization.
 *  - Randomly initializing mutable parameters within specified bounds before performing LM fitting.
 *  - Tracking and retaining the best fit parameters and weighted sum of squared residuals (WSSR) found during the fitting process.
 *
 * Key Concepts:
 *  - Monte Carlo Initialization: Randomizes mutable parameters within their allowed ranges to escape local minima and improve global search.
 *  - Levenberg-Marquardt Fitting: Uses LM optimization to refine parameters and minimize the WSSR for the model fit.
 *  - Parameter Bounds: Supports user-specified or default parameter bounds for random initialization.
 *  - Best Fit Tracking: Remembers the best parameter set and WSSR encountered during the fitting iterations.
 *
 * Fields:
 *  - Inherited from LMFitter:
 *      - Model pointer, fitting properties, data vectors, parameter vectors, mutability masks, parameter bounds, and internal LM state.
 *  - double bestWssr: Best (lowest) WSSR found during fitting.
 *  - VectorXd bestParams: Parameter vector corresponding to bestWssr.
 *  - cons* p_min, p_max: Parameter bounds for Monte Carlo initialization.
 *  - bool delPmin, delPmax: Flags for ownership and cleanup of parameter bounds.
 *
 * Main Methods:
 *  - MCLMFitter::MCLMFitter(...):
 *      Constructor. Initializes random seed, parameter bounds, and best fit tracking. Inherits from LMFitter.
 *  - double MCLMFitter::FitIteration():
 *      Performs a Monte Carlo randomization of mutable parameters, then runs LM fitting. Tracks and returns the best WSSR.
 *
 * Error Handling:
 *  - Returns 0.0 if no mutable parameters or if an error is detected.
 *  - Ensures parameter bounds are allocated if not provided.
 *  - Handles random seed initialization only once per process.
 *
 * Dependencies:
 *  - MCFitting.h for class and method declarations.
 *  - LMFitter base class for LM optimization logic.
 *  - Standard C++ libraries for randomization and time.
 *  - Eigen library for vector/matrix operations (VectorXd, VectorXi, MatrixXd).
 *
 * See MCFitting.h and LMFitter for class and method declarations and base functionality.
 */

MCLMFitter::MCLMFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
           const std::vector<double>& datay,
           const std::vector<double>& factor, 
           const std::vector<double>& bg,
           const std::vector<double>& fitWeights, VectorXd& p,
           const VectorXi& pmut, cons *pMin, cons *pMax,
		   int layers)  : LMFitter(model, fp, datax, datay, factor, bg,
                                    fitWeights, p, pmut, pMin, pMax, layers) {
    if(!randInit) {
        srand((unsigned int)time(NULL));
        randInit = true;
    }

    if(!pMin) {
        p_min = new cons(nParams);
        delPmin = true;
    } else
        delPmin = false;
    
    if(!pMax) {
        p_max = new cons(nParams);
        for(int i = 0; i < nParams; i++)
            p_max->num[i] = 1000.0;
        
        delPmax = true;
    } else
        delPmax = false;

	bestWssr = LMFitter::FitIteration();
	bestParams = params;
}

double MCLMFitter::FitIteration() {
    VectorXd curParams = params;
    double curWssr = 0.0, lastWssr;

	if(mutables == 0 || GetError())
		return 0.0;

	for(int i = 0; i < nParams; i++) {
		if(paramMut[i])
			curParams[i] = ((float)rand() / (float)RAND_MAX) *
						   (p_max->num[i] - p_min->num[i]) + p_min->num[i];
	}

    params = curParams;

	// Constructing a new LMFitter
	alpha = MatrixXd::Zero(mutables, mutables);
    beta  = VectorXd::Zero(mutables);
            
    lambda = 0.001;
           
    curWssr = CalculateCoefficients(params, alpha, beta);
    
	// Fitting
    do {
		lastWssr = curWssr;
        curWssr = LMFitter::FitIteration();
		if(mutables == 0 || GetError())
			return 0.0;

        if(curWssr < bestWssr) {
            bestWssr = curWssr;
            bestParams = params;
        }

    } while(lambda <= 1e4 && fabs(lastWssr - curWssr) >= 1e-7);

    return bestWssr;
}
