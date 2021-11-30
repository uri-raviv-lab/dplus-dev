#include "MCFitting.h"
#include <ctime>

static bool randInit = false;

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
