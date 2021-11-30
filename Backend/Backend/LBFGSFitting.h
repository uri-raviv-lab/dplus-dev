#ifndef __LBFGSFITTING_H
#define __LBFGSFITTING_H

#include "fitting.h"

class LBFGSFitter : public ModelFitter {
protected:
	bool delPmin, delPmax;

	void *h_lbfgs;
	int maxIters;

	double curEval;

	
public:

	// Initializes a new LBFGS fitter
	LBFGSFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
		const std::vector<double>& datay,
		const std::vector<double>& factor, 
		const std::vector<double>& bg,
		const std::vector<double>& fitWeights, VectorXd& p,
		const VectorXi& pmut, cons *pMin, cons *pMax,
		int layers);

	virtual double FitIteration();

	virtual void GetFittingErrors(std::vector<double>& paramErrors, std::vector<double>& modelErrors) {}

	static double LBEvaluate(void *instance, const double *x, double *g, const int n,
							 const double step);

	virtual ~LBFGSFitter();
};


#endif
