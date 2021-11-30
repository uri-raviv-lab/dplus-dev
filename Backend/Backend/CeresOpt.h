#ifndef __CERESOPT_H
#define __CERESOPT_H

#include "Optimizer.h"
#include "ceres/ceres.h"
#include "Common.h"

// Forward declaration
class ICeresModel;

typedef ceres::CostFunction * (*GetCeresCostFunction_t)(double x, double y, const std::vector<int>& params, int numResiduals);

typedef ceres::CostFunction * (*GetCeresVecCostFunction_t)(const double* x, const double* y, int params, int numResiduals, 
														   ICeresModel* dm, double stepSize, double derEps,
														   VectorXd *bestParams, double *bestEval);

class CeresOptimizer : public IOptimizationMethod {
protected:
	CeresProperties cp_;
	ceres::Problem problem;
	ceres::Solver::Options options;
	int maxIters;

	int m_numParamBlocks;
	int m_numResiduals;
	std::vector<int> m_paramsPerBlock;
	std::vector<VectorXd> curParams;
	std::vector<VectorXd> bestParams;
	double curEval;
	double bestEval;
	double m_convergence;
	double m_stepSize;
	double m_derEps;
	int convergenceCounter;

	bool bClonedObjective;
	bool bConverged;
	VectorXd m_x, m_y, m_lowerBound, m_upperBound;
	GetCeresVecCostFunction_t m_createCostFunction;

	ICeresModel* dm_;

	void InitProblem(GetCeresVecCostFunction_t createCostFunction,
		const std::vector<int>& paramsPerBlock, int numResiduals,
		const VectorXd& x, const VectorXd& y, int maxIterations,
		const VectorXd& lowerBound, const VectorXd& upperBound,
		double stepSize, double convergence, double derEps);

	CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const CeresProperties &cp, 
		int numParams, const VectorXd& x, const VectorXd& y, int maxIterations,
		const VectorXd& lowerBound, const VectorXd& upperBound, ICeresModel* dm,
		double stepSize, double convergence, double derEps, bool bCloned);

public:
	CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const CeresProperties &cp, 
		int numParams, const VectorXd& x, const VectorXd& y, int maxIterations,
		const VectorXd& lowerBound, const VectorXd& upperBound, ICeresModel* dm,
		double stepSize, double convergence, double derEps);
	virtual ~CeresOptimizer();

	virtual double Iterate(const VectorXd& p, VectorXd& pnew);

	virtual bool Convergence() const;
	virtual double GetEval() const;
	virtual void GetParams(VectorXd& params) const;

	virtual void AddCallback(ceres::IterationCallback *callback);

	virtual IOptimizationMethod *Clone();
};


class ProgressFromCeres : public ceres::IterationCallback
{
public:
	virtual ~ProgressFromCeres();
	virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);

	ProgressFromCeres(progressFunc prog, void* progress_args, int maxIterations, int* pStop = nullptr);
private:
	progressFunc _prg;
	void* _args;
	int _maxIterations;
	int* _p_stop;
};

#endif // __CERESOPT_H