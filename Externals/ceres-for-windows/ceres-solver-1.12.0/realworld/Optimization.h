// Copyright Tal Ben-Nun, 2013. All rights reserved.

#ifndef __OPTIMIZATION_H
#define __OPTIMIZATION_H

#include "Eigen/Core"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class LConstraintEnforcer;

// An abstract class representing an objective function
class IObjective {
protected:
	IObjective() {}
public:
	virtual ~IObjective() {}

	virtual double Evaluate(const double *p, int nParams) = 0;
	virtual double Evaluate(const VectorXd& p) { return Evaluate(p.data(), p.size()); }

	virtual void EvaluateGradient(const double *p, int nParams, double *res) = 0;	
	virtual void EvaluateGradient(const VectorXd& p, VectorXd& res) { 
		res = VectorXd::Zero(p.size());
		EvaluateGradient(p.data(), p.size(), res.data());
	}

	virtual IObjective *Clone() = 0;
};

void NumericalGradient(IObjective *obj, const double *p, int nParams, double *res);

class IOptimizationMethod {
protected:
	unsigned int nEvals, nGradEvals;
	LConstraintEnforcer *con;
	IObjective *objective;

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

#endif
