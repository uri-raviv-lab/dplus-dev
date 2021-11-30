// Copyright Tal Ben-Nun, 2014. All rights reserved.

#include "CeresOpt.h"


IOptimizationMethod * CeresOptimizer::Clone()
{
	return new CeresOptimizer(m_createCostFunction, m_paramsPerBlock, m_numResiduals, 
		m_x, m_y, maxIters, m_lowerBound, m_upperBound, 
		m_bDogleg, m_bInnerIters, m_bLineSearch, true);
}

void CeresOptimizer::GetParams(VectorXd& params) const
{
	const int num_parameters = std::accumulate(m_paramsPerBlock.begin(),
		m_paramsPerBlock.end(), 0);

	params = VectorXd::Zero(num_parameters);

	// Convert parameter matrix back to vector
	int start = 0;
	for(int N = 0; N < m_numParamBlocks; N++)
	{
		for(int i = start; i < (start + m_paramsPerBlock[N]); i++)
		{
			params[i] = curParams[N][i - start];
		}
		start += m_paramsPerBlock[N];
	}
}

double CeresOptimizer::GetEval() const
{
	return curEval;
}

bool CeresOptimizer::Convergence() const
{
	return bConverged;
}

double CeresOptimizer::Iterate(const VectorXd& p, VectorXd& pnew)
{
	// Convert parameter vector to parameter matrix
	int start = 0;
	for(int N = 0; N < m_numParamBlocks; N++)
	{
		for(int i = start; i < (start + curParams[N].size()); i++)
		{
			curParams[N][i - start] = p[i];
		}
		start += curParams[N].size();
	}

	// Apply/enforce local constraints
	/*if(!EnforceConstraints(pnew)) {
	nEvals++;
	return (curEval = objective->Evaluate(pnew)); // If cannot enforce constraints
	}*/

	// Perform the actual optimization
	ceres::Solver::Summary summary;
	//std::cout << "Problem? " << (void *)&problem << std::endl;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	// Get parameter vector from blocks
	GetParams(pnew);

	curEval = summary.final_cost;
	bConverged = (summary.termination_type == ceres::CONVERGENCE);

	return summary.final_cost;
}

CeresOptimizer::~CeresOptimizer()
{

}

CeresOptimizer::CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals, const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, bool bDogleg, bool bInnerIters, bool bLineSearch)
{
	maxIters = maxIterations; 
	bClonedObjective = false;

	InitProblem(createCostFunction, paramsPerBlock, numResiduals, x, y, maxIterations, lowerBound, upperBound, bDogleg, bInnerIters, bLineSearch);
}

CeresOptimizer::CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals, const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, bool bDogleg, bool bInnerIters, bool bLineSearch, bool bCloned)
{
	maxIters = maxIterations;
	bClonedObjective = bCloned;

	InitProblem(createCostFunction, paramsPerBlock, numResiduals, x, y, maxIterations, lowerBound, upperBound, bDogleg, bInnerIters, bLineSearch);
}

void CeresOptimizer::InitProblem(GetCeresVecCostFunction_t createCostFunction, const std::vector<int>& paramsPerBlock, int numResiduals, const VectorXd& x, const VectorXd& y, int maxIterations, const VectorXd& lowerBound, const VectorXd& upperBound, bool bDogleg, bool bInnerIters, bool bLineSearch)
{
	m_x = x; m_y = y; m_lowerBound = lowerBound; m_upperBound = upperBound; m_paramsPerBlock = paramsPerBlock;
	m_bDogleg = bDogleg; m_bInnerIters = bInnerIters; m_bLineSearch = bLineSearch;
	m_numResiduals = numResiduals; m_createCostFunction = createCostFunction;
	bConverged = false;

	m_numParamBlocks = paramsPerBlock.size();
	std::vector<double *> paramdata (m_numParamBlocks);
	curParams.resize(m_numParamBlocks);

	for(int i = 0; i < m_numParamBlocks; i++)
	{
		curParams[i] = VectorXd::Zero(paramsPerBlock[i]);
		paramdata[i] = curParams[i].data();
	}


	ceres::CostFunction* cost_function = createCostFunction(x.data(), y.data(), paramsPerBlock, numResiduals);
	problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), paramdata);
/*
	for (int i = 0; i < x.size(); ++i) {
		ceres::CostFunction* cost_function = createCostFunction(x[i], y[i], paramsPerBlock, numResiduals);

		problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), paramdata);
	}
*/

	// Set bounds
	if(!bLineSearch)
	{
		int start = 0;

		for(int N = 0; N < m_numParamBlocks; N++)
		{
			for(int i = start; i < (start + paramsPerBlock[N]); i++)
			{
				problem.SetParameterLowerBound(paramdata[N], i - start, lowerBound[i]);
				problem.SetParameterUpperBound(paramdata[N], i - start, upperBound[i]);
			}
			start += paramsPerBlock[N];
		}
	}

#if defined(DEBUG) || defined(_DEBUG)
	options.minimizer_progress_to_stdout = true;
#endif

	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = maxIterations;

	if(bLineSearch)
	{
		options.minimizer_type = ceres::LINE_SEARCH;
		options.line_search_direction_type = ceres::LBFGS;
	}
	else
	{
		if(bDogleg)
			options.trust_region_strategy_type = ceres::DOGLEG;
		if(bInnerIters)
			options.use_inner_iterations = true;
		//options.use_nonmonotonic_steps = true;  
	}
}
