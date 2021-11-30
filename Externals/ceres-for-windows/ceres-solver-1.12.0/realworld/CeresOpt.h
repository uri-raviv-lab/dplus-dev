// Copyright Tal Ben-Nun, 2014. All rights reserved.

#ifndef __CERESOPT_H
#define __CERESOPT_H

#include "Optimization.h"
#include "ceres/ceres.h"

typedef ceres::CostFunction * (*GetCeresCostFunction_t)(double x, double y, const std::vector<int>& params, int numResiduals);

typedef ceres::CostFunction * (*GetCeresVecCostFunction_t)(const double* x, const double* y, const std::vector<int>& params, int numResiduals);

class CeresOptimizer : public IOptimizationMethod {
protected:
	ceres::Problem problem;
	ceres::Solver::Options options;
	int maxIters;

	int m_numParamBlocks;
	int m_numResiduals;
	std::vector<int> m_paramsPerBlock;
	std::vector<VectorXd> curParams;
	double curEval;

	bool bClonedObjective;
	bool bConverged;
	bool m_bDogleg, m_bInnerIters, m_bLineSearch;
	VectorXd m_x, m_y, m_lowerBound, m_upperBound;
	GetCeresVecCostFunction_t m_createCostFunction;

	void InitProblem(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals,
					 const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, 
					 bool bDogleg, bool bInnerIters, bool bLineSearch);

	CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals, 
				   const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, 
				   bool bDogleg, bool bInnerIters, bool bLineSearch, bool bCloned);
public:
	CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals, 
			       const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, 
				   bool bDogleg, bool bInnerIters, bool bLineSearch);
	virtual ~CeresOptimizer();

	virtual double Iterate(const VectorXd& p, VectorXd& pnew);

	virtual bool Convergence() const;
	virtual double GetEval() const;
	virtual void GetParams(VectorXd& params) const;	

	virtual IOptimizationMethod *Clone();
};

#endif
