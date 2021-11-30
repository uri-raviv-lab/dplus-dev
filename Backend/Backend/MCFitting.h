#ifndef __MCFITTING_H
#define __MCFITTING_H

#include "fitting.h"

class MCLMFitter : public LMFitter {
protected:
    double bestWssr;
    VectorXd bestParams;
    bool delPmin, delPmax;
    
public:
    
    // Initializes a new Monte-Carlo Levenberg-Marquardt (raindropping) fitter
    MCLMFitter(IModel *model, const FittingProperties& fp, const std::vector<double>& datax, 
             const std::vector<double>& datay,
             const std::vector<double>& factor, 
             const std::vector<double>& bg,
             const std::vector<double>& fitWeights, VectorXd& p,
             const VectorXi& pmut, cons *pMin, cons *pMax,
			 int layers);


    virtual double FitIteration();

    virtual VectorXd GetResult() const { return bestParams; }
    
    virtual ~MCLMFitter() {
        if(delPmin)
            delete p_min;
        if(delPmax)
            delete p_max;
    }
};


#endif
