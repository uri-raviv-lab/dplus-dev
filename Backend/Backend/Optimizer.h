#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

#include "Eigen/Core"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class IOptimizationMethod {
protected:
	unsigned int nEvals, nGradEvals;

	IOptimizationMethod() : nEvals(0), nGradEvals(0) {}

public:
	virtual ~IOptimizationMethod() {}

	virtual double Iterate(const VectorXd& p, VectorXd& pnew) = 0;

	virtual double GetEval() const = 0;
	virtual void GetParams(VectorXd& params) const = 0;

	virtual unsigned int GetNumEvaluations() { return nEvals; }
	virtual unsigned int GetNumGradientEvaluations() { return nGradEvals; }
	virtual bool Convergence() const { return false; }

	virtual IOptimizationMethod *Clone() = 0;
};

#endif //__OPTIMIZER_H
