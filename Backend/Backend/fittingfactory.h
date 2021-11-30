#ifndef __FITTINGFACTORY_H
#define __FITTINGFACTORY_H

#include "fitting.h"
#include "DEFitting.h"
#include "MCFitting.h"
#include "LBFGSFitting.h"

#include "Common.h"

inline static ModelFitter *CreateFitter(FitMethod method, IModel *model, 
										const FittingProperties& fp,
										const std::vector<double>& datax, 
										const std::vector<double>& datay,
										const std::vector<double>& factor, 
										const std::vector<double>& bg,
										const std::vector<double>& fitWeights, VectorXd& p,
										const VectorXi& pmut, cons& pMin, cons& pMax,
										int layers) {
	switch(method) {
		default:
		case FIT_LM:
			return new LMFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, &pMin, &pMax, layers);
		case FIT_DE:
			return new DEFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, &pMin, &pMax, layers);
		case FIT_RAINDROP:
			return new MCLMFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, &pMin, &pMax, layers);
		case FIT_LBFGS:
			return new LBFGSFitter(model, fp, datax, datay, factor, bg, fitWeights, p, pmut, &pMin, &pMax, layers);
	}
}

#endif
