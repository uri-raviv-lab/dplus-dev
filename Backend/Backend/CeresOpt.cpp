
#include "Amplitude.h"
#include "CeresOpt.h"


#define _cOnVeRgEnCe_CrItErIoN_ 2

IOptimizationMethod * CeresOptimizer::Clone()
{
	return new CeresOptimizer(m_createCostFunction, cp_, m_paramsPerBlock[0], 
		m_x, m_y, maxIters, m_lowerBound, m_upperBound, dm_, m_stepSize,
		m_convergence, m_derEps, true);
}

void CeresOptimizer::GetParams(VectorXd& params) const
{
	dm_->GetMutatedParamVecForCeres(&params);
	return;

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
	//std::cout << summary.BriefReport() << "\n";
	std::cout << summary.FullReport() << "\n";
	
	// Check for reason why we failed, (no convergence, e.g., is not a bad reason)
	if (summary.termination_type == ceres::TerminationType::FAILURE)
	{
		// Ceres initialization failed.
		if (summary.num_successful_steps < 0 && summary.num_unsuccessful_steps < 0)
		{
			throw backend_exception(ERROR_CERES_INITIALIZATION_FAILURE, g_errorStrings[ERROR_CERES_INITIALIZATION_FAILURE]);
		}
	}

	// Get parameter vector from blocks
	GetParams(pnew);

	curEval = summary.final_cost;
	bConverged = (summary.termination_type == ceres::CONVERGENCE);
	// TODO: If a new convergence test based on ceres' result is to be added,
	//       here is the place.
	if((p - pnew).norm() < m_convergence)
		convergenceCounter++;
	else
		convergenceCounter = 0;

	if(convergenceCounter > _cOnVeRgEnCe_CrItErIoN_)
	{
		convergenceCounter = 0;
		bConverged = true;
	}

	// OPTIONAL
	// Store best params in the original positions
	if (
		(bestParams[0].array() <= m_upperBound.array()).all() &&
		(bestParams[0].array() >= m_lowerBound.array()).all()
		)
	{
		curEval = bestEval;
		pnew = bestParams[0];
	}

	dm_->SetMutatedParamVecForCeres(pnew);
	return curEval;
}

CeresOptimizer::~CeresOptimizer()
{

}
CeresOptimizer::CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const CeresProperties &cp,
							   int numParams, const VectorXd& x, const VectorXd& y, int maxIterations,
							   const VectorXd& lowerBound, const VectorXd& upperBound, ICeresModel* dm,
							   double stepSize, double convergence, double derEps, bool bCloned )
{
	bClonedObjective = bCloned;
	dm_ = dm;
	m_createCostFunction = createCostFunction;
	cp_ = cp; m_x = x; m_y = y; m_lowerBound = lowerBound; m_upperBound = upperBound;

	InitProblem(createCostFunction, std::vector<int>(1,numParams), x.size(), x, y, maxIterations,
		lowerBound, upperBound, stepSize, convergence, derEps);
}

CeresOptimizer::CeresOptimizer(GetCeresVecCostFunction_t createCostFunction, const CeresProperties &cp,
							   int numParams, const VectorXd& x, const VectorXd& y, int maxIterations,
							   const VectorXd& lowerBound, const VectorXd& upperBound, ICeresModel* dm,
							   double stepSize, double convergence, double derEps )
{
	bClonedObjective = false;
	dm_ = dm;
	cp_ = cp; m_x = x; m_y = y; m_lowerBound = lowerBound; m_upperBound = upperBound;

	InitProblem(createCostFunction, std::vector<int>(1,numParams), x.size(), x, y, maxIterations,
		lowerBound, upperBound, stepSize, convergence, derEps);
}

void CeresOptimizer::InitProblem(GetCeresVecCostFunction_t createCostFunction,
								 const std::vector<int>& paramsPerBlock,
								 int numResiduals, const VectorXd& x, 
								 const VectorXd& y, int maxIterations, 
								 const VectorXd& lowerBound, const VectorXd& upperBound,
								 double stepSize, double convergence, double derEps)
{
	m_x = x; m_y = y; m_lowerBound = lowerBound;
	m_upperBound = upperBound; m_paramsPerBlock = paramsPerBlock;
	m_numResiduals = numResiduals; m_createCostFunction = createCostFunction;
	m_convergence = convergence;
	m_stepSize = stepSize;
	m_derEps = derEps;
	bConverged = false;
	convergenceCounter = 0;

	m_numParamBlocks = paramsPerBlock.size();
	std::vector<double *> paramdata (m_numParamBlocks);
	curParams.resize(m_numParamBlocks);
	bestParams.resize(m_numParamBlocks);
	bestEval = std::numeric_limits<double>::infinity();

	for(int i = 0; i < m_numParamBlocks; i++)
	{
		curParams[i] = VectorXd::Zero(paramsPerBlock[i]);
		paramdata[i] = curParams[i].data();
	}


	ceres::CostFunction* cost_function = createCostFunction(m_x.data(), m_y.data(),
		paramsPerBlock[0], numResiduals, dm_, stepSize, derEps, &bestParams[0], &bestEval);

	switch (cp_.lossFuncType)
	{
	case TRIVIAL_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::TrivialLoss(), paramdata);
		break;
	case HUBER_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::HuberLoss(cp_.lossFunctionParameters[0]), paramdata);
		break;
	case SOFTLONE_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::SoftLOneLoss(cp_.lossFunctionParameters[0]), paramdata);
		break;
	case CAUCHY_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(cp_.lossFunctionParameters[0]), paramdata);
		break;
	case ARCTAN_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::ArctanLoss(cp_.lossFunctionParameters[0]), paramdata);
		break;
	case TOLERANT_LOSS:
		problem.AddResidualBlock(cost_function,
			new ceres::TolerantLoss(cp_.lossFunctionParameters[0], cp_.lossFunctionParameters[1]), paramdata);
		break;
	}

	// Set bounds
	if(cp_.minimizerType == TRUST_REGION)
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
	options.minimizer_progress_to_stdout = true;

	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = maxIterations;
	options.use_nonmonotonic_steps = true;  

	if(cp_.minimizerType == LINE_SEARCH)
	{
		options.minimizer_type = ceres::MinimizerType(cp_.minimizerType);
		options.line_search_direction_type = ceres::LineSearchDirectionType(cp_.lineSearchDirectionType);
		options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType(cp_.nonlinearConjugateGradientType);
		options.line_search_type = ceres::LineSearchType(cp_.lineSearchType);
	}
	else
	{
		options.trust_region_strategy_type = ceres::TrustRegionStrategyType(cp_.trustRegionStrategyType);
//		if(bInnerIters)
			options.use_inner_iterations = false;
	}

	options.function_tolerance = m_convergence;
	options.update_state_every_iteration = true;
	options.gradient_tolerance = 1e-4 * options.function_tolerance;
}

void CeresOptimizer::AddCallback(ceres::IterationCallback *callback)
{
	options.callbacks.push_back(callback);
}

ceres::CallbackReturnType ProgressFromCeres::operator()(const ceres::IterationSummary& summary)
{
	if (_prg)
	{
		_prg(_args, double(summary.iteration) / double(_maxIterations));
// 		std::cout << "Parameters at the end of iteration #" << summary.iteration << ":\n";
// 		for (int i = 0; i < summary. summary.parameters)
	}
	fflush(stdout);

	if (_p_stop && *_p_stop)
		return ceres::CallbackReturnType::SOLVER_ABORT;

	return ceres::CallbackReturnType::SOLVER_CONTINUE;
}

ProgressFromCeres::~ProgressFromCeres()
{
}

ProgressFromCeres::ProgressFromCeres(progressFunc prog, void* progress_args, int maxIterations, int* pStop)
	: _prg(prog), _args(progress_args), _maxIterations(maxIterations), _p_stop(pStop)
{
}

;