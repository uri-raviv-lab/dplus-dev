#include <cmath>
#include <cstdlib>

#include "Amplitude.h"
#include "Geometry.h"

#include "modelfitting.h"
#include "JobManager.h"
#include "CommProtocol.h"

#include "fittingfactory.h"
#include "mathfuncs.h"

#include "LocalBackend.h"

#undef ERROR
#include "CeresOpt.h"
#include "dynamic_adaptive_numeric_diff_cost_function.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// Helper functions
#pragma region Helper Functions

void ParameterToVectorIndex(const Parameter& param, int index, VectorXd& a,
							VectorXi& ia, cons& a_min, cons& a_max) {
	a[index] = param.value;
	ia[index] = param.isMutable;

	if(param.isConstrained) {
		// Absolute constraints
		a_min.num[index] = param.consMin;
		a_max.num[index] = param.consMax;

		// Relative constraints
		a_min.index[index] = param.consMinIndex;
		if(a_min.index[index] >= index)
			a_min.index[index]++;

		a_max.index[index] = param.consMaxIndex;
		if(a_max.index[index] >= index)
			a_max.index[index]++;

		// Linking constraints
		a_min.link[index] = a_max.link[index] = 
			param.linkIndex;
	
	} else {
		a_min.num[index]	= -std::numeric_limits<double>::infinity();
		a_max.num[index]	= std::numeric_limits<double>::infinity();
		a_min.index[index]	= -1;
		a_max.index[index]	= -1;
		a_min.link[index]	= a_max.link[index] = -1;
	}
}

bool RecursiveModelSetter(Job *j, const ParameterTree *pt, int *pStop, bool bAmplitude) {
	if(!pt)
		return false;

	if(!bAmplitude) { // IModel
		if(j->uidToModel.find(pt->GetNodeModel()) == j->uidToModel.end())
			return false;

		IModel *mod = j->uidToModel[pt->GetNodeModel()];

		// Set the stop signal
		mod->SetStop(pStop);

		if(dynamic_cast<ISymmetry *>(mod)) { // Domain model
			ISymmetry *symm = dynamic_cast<ISymmetry *>(mod);

			symm->ClearSubAmplitudes();
			int num = pt->GetNumSubModels();

			// Recursively set the amplitude's children
			for(int i = 0; i < num; i++) {
				if(!RecursiveModelSetter(j, pt->GetSubModel(i), pStop, true))
					return false;

				symm->AddSubAmplitude(j->uidToAmp[pt->GetSubModel(i)->GetNodeModel()]);
			}

		} else if(dynamic_cast<CompositeModel *>(mod)) {
			CompositeModel *cm = dynamic_cast<CompositeModel *>(mod);

			std::vector<IModel *> submodels;

			int num = pt->GetNumSubModels();
			cm->ClearMultipliers();
			cm->SetSubModels(submodels);

			// HACK: This allows multipliers only for the case of FF * SF + BG
			std::vector<unsigned int> ffIndices;

			// Recursively set the model's children
			for(int i = 0; i < num; i++) {
				if(!RecursiveModelSetter(j, pt->GetSubModel(i), pStop, false))
					return false;

				IModel *submod = (IModel *)j->uidToModel[pt->GetSubModel(i)->GetNodeModel()];
				if(dynamic_cast<FFModel *>(submod))
					ffIndices.push_back(i);
	
				submodels.push_back(submod);
			}

			cm->SetSubModels(submodels);

		} // Other geometries cannot contain children

	} else { // Amplitude
		if(j->uidToAmp.find(pt->GetNodeModel()) == j->uidToAmp.end())
			return false;

		Amplitude *amp = j->uidToAmp[pt->GetNodeModel()];

		// TODO: Implement SetStop and stop signals in amplitudes
		//amp->SetStop(pStop);

		if(dynamic_cast<ISymmetry *>(amp)) {
			ISymmetry *symm = dynamic_cast<ISymmetry *>(amp);

			symm->ClearSubAmplitudes();
			int num = pt->GetNumSubModels();

			// Recursively set the amplitude's children
			for(int i = 0; i < num; i++) {
				if(!RecursiveModelSetter(j, pt->GetSubModel(i), pStop, true))
					return false;

				symm->AddSubAmplitude(j->uidToAmp[pt->GetSubModel(i)->GetNodeModel()]);
			}
		}
	}


	return true;
 }

 IModel *CreateModelFromParamTree(Job *job, const ParameterTree& pt, int *pStop, VectorXd& p, VectorXi& pMut, cons& pMin, cons& pMax)  {
	 // PRECONDITION: Assumes that all arguments exist and are valid
	 IModel *res = NULL, *originalModel = NULL;

	 if(job->uidToModel.find(pt.GetNodeModel()) == job->uidToModel.end())
		 return NULL;

	 res = originalModel = job->uidToModel[pt.GetNodeModel()];

	 //int dummy;
	// res->SetStop(&dummy);

	 // Set the model and its children and set their stop signal
	 if(!RecursiveModelSetter(job, &pt, pStop, false))
		 return NULL;

	 int numParams = pt.ToParamVector();
	 p = VectorXd::Zero(numParams);
	 pMut = VectorXi::Zero(numParams);
	 pMin = cons(numParams);
	 pMax = cons(numParams);

	 // Initialize the arrays
	 double *darr = new double[numParams];
	 int *iarr = new int[numParams];

	 // Set parameter vector
	 if(pt.ToParamVector(darr) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 p[i] = darr[i];

	 // Set mutability vector
	 if(pt.ToMutabilityVector(iarr) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 pMut[i] = iarr[i];

	 // Set minimum constraints
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_MINVAL) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 pMin.num[i] = darr[i];
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_MININD) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 pMin.index[i] = dbltoint(darr[i]);

	 // Set maximum constraints
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_MAXVAL) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 pMax.num[i] = darr[i];
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_MAXIND) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++)
		 pMax.index[i] = dbltoint(darr[i]);

	 // Set link constraints
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_MAXIND) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++) {
		 pMin.link[i] = dbltoint(darr[i]);
		 pMax.link[i] = dbltoint(darr[i]);
	 }

	 // Model Modifiers
	 //////////////////////////////////////////////////////////////////////////

	 // TODO::EDP Custom Electron Density Profile

	 // Set polydispersity if sigma is non-zero
	 if(pt.ToConstraintVector(darr, ParameterTree::CT_SIGMA) != numParams) {
		 delete[] darr;
		 delete[] iarr;
		 return NULL;
	 }
	 for(int i = 0; i < numParams; i++) {
		 if(darr[i] > 0.0) {
			 // TODO::PD Custom Polydispersity resolution/shape (in params?)
			 res = new PolydisperseModel(res, i, darr[i], p, DEFAULT_PDRES, 
				                         SHAPE_GAUSSIAN, (res != originalModel));
			 res->SetStop(pStop);
		 }
	 }

	 delete[] darr;
	 delete[] iarr;

	 return res;
 }

 struct XRayResiduals
 {
	 XRayResiduals(const double *x, const double* y, int numParams,
		 int numResiduals, ICeresModel* dm, 
		 VectorXd *pBestParams, double *pBestEval)
		 : x_(x), y_(y), numParams_(numParams), numResiduals_(numResiduals), 
		   dm_(dm), pBestParams_(pBestParams),
		   pBestEval_(pBestEval),
		   bestEval_(std::numeric_limits<double>::infinity())
	 {
	 }

	 static ceres::CostFunction *GetCeresCostFunction(const double *x, const double *y,
		 int numParams, int numResiduals, ICeresModel* dm, double stepSize = 1e-2,
		 double eps = 1e-6, VectorXd *pBestParams = NULL, double *pBestEval = NULL)
	 {
		auto *res =
			new ceres::DynamicAdaptiveNumericDiffCostFunction<XRayResiduals/*, ceres::CENTRAL, 10, false*/>(
				 new XRayResiduals(x, y, numParams, numResiduals, dm, pBestParams, pBestEval),
				 ceres::TAKE_OWNERSHIP, stepSize, eps);

		res->AddParameterBlock(numParams);
		res->SetNumResiduals(numResiduals);

		return res;
	 }

	 //template <typename T> 
	 bool operator()(double const* const* p, double* residual) const {
		 double res = 0.0;
		 dm_->CalculateVectorForCeres(x_, p, residual, numResiduals_);
		 for(int i = 0; i < numResiduals_; i++) {
/*
			double rat = residual[i] / y_[i];
			rat = 1. - ( (rat < 1.) ? rat : (1./rat) );
			residual[i] = rat;
*/
			residual[i] = fabs(y_[i] - residual[i]);
			res += residual[i] * residual[i];
		 }

		 res /= 2.0;

		 if(res < bestEval_)
		 {
			 bestEval_ = res;

			 if(pBestEval_)
				*pBestEval_ = res;
			 if(pBestParams_)
				 *pBestParams_ = Eigen::Map<const VectorXd>(p[0], numParams_);
		 }

		 return true;
	 }

	 const double* operator()() const {
		 return x_;
	 }

 private:
	 ICeresModel *dm_;
	 const double *x_;
	 const double *y_;
	 int numResiduals_;
	 int numParams_;

	 VectorXd *pBestParams_;
	 double *pBestEval_;
	 mutable double bestEval_;
 };

 struct XRayRatioResiduals
 {
	 XRayRatioResiduals(const double *x, const double* y, int numParams,
		 int numResiduals, ICeresModel* dm,
		 VectorXd *pBestParams, double *pBestEval)
		 : x_(x), y_(y), numParams_(numParams), numResiduals_(numResiduals), dm_(dm),
		   pBestParams_(pBestParams), pBestEval_(pBestEval),
		   bestEval_(std::numeric_limits<double>::infinity())
	 {
	 }

	 static ceres::CostFunction *GetCeresCostFunction(const double *x, const double *y,
		 int numParams, int numResiduals, ICeresModel* dm, double stepSize = 1e-2,
		 double eps = 1e-6, VectorXd *pBestParams = NULL, double *pBestEval = NULL)
	 {
		 ceres::DynamicAdaptiveNumericDiffCostFunction<XRayRatioResiduals> *res = 
			 new ceres::DynamicAdaptiveNumericDiffCostFunction<XRayRatioResiduals>(
					new XRayRatioResiduals(x, y, numParams, numResiduals, dm, pBestParams, pBestEval),
					ceres::TAKE_OWNERSHIP, stepSize, eps);

		 res->AddParameterBlock(numParams);
		 res->SetNumResiduals(numResiduals);

		 return res;
	 }

	 //template <typename T> 
	 bool operator()(double const* const* p, double* residual) const {
		 double res = 0.0;
		 dm_->CalculateVectorForCeres(x_, p, residual, numResiduals_);
		 for(int i = 0; i < numResiduals_; i++) {
			 double rat = residual[i] / y_[i];
			 //rat = 1. - ( (rat < 1.) ? rat : (1./rat) );
			 rat = 1. - rat;
			 residual[i] = rat;
			 res += rat * rat;
		 }

		 res /= 2.0;

		 printf("p = ");
		 for(int i = 0; i < numParams_; ++i)
			 printf("%f ", p[0][i]);
		 printf("\t cost: %f\n", res);		 

		 if(res < bestEval_)
		 {
			 bestEval_ = res;

			 if(pBestEval_)
				 *pBestEval_ = res;
			 if(pBestParams_)
				 *pBestParams_ = Eigen::Map<const VectorXd>(p[0], numParams_);
		 }

		 return true;
	 }

	 const double* operator()() const {
		 return x_;
	 }

 private:
	 ICeresModel *dm_;
	 const double *x_;
	 const double *y_;
	 int numResiduals_;
	 int numParams_;

	 VectorXd *pBestParams_;
	 double *pBestEval_;
	 mutable double bestEval_;
 };

 struct XRayLogResiduals
 {
	 XRayLogResiduals(const double *x, const double* y, int numParams, int numResiduals, ICeresModel* dm, 
		 VectorXd *pBestParams, double *pBestEval)
		 : x_(x), y_(y), numParams_(numParams), numResiduals_(numResiduals), dm_(dm),
		 pBestParams_(pBestParams), pBestEval_(pBestEval),
		 bestEval_(std::numeric_limits<double>::infinity())
	 {
	 }

	 static ceres::CostFunction *GetCeresCostFunction(const double *x, const double *y,
		 int numParams, int numResiduals, ICeresModel* dm, double stepSize = 1e-2,
		 double eps = 1e-6, VectorXd *pBestParams = NULL, double *pBestEval = NULL)
	 {
		auto *res =
			new ceres::DynamicAdaptiveNumericDiffCostFunction<XRayLogResiduals/*, ceres::CENTRAL, 10, false*/>(
				 new XRayLogResiduals(x, y, numParams, numResiduals, dm, pBestParams, pBestEval),
				 ceres::TAKE_OWNERSHIP, stepSize, eps);

		 res->AddParameterBlock(numParams);
		 res->SetNumResiduals(numResiduals);

		 return res;
	 }

	 //template <typename T> 
	 bool operator()(double const* const* p, double* residual) const {
		 double res = 0.0;
		 dm_->CalculateVectorForCeres(x_, p, residual, numResiduals_);
		 for(int i = 0; i < numResiduals_; i++) {
			 residual[i] = /*fabs*/(log10(residual[i]) - log10(y_[i]));
			 res += residual[i] * residual[i];
		 }

		 if(res < bestEval_)
		 {
			 bestEval_ = res;

			 if(pBestEval_)
				 *pBestEval_ = res;
			 if(pBestParams_)
				 *pBestParams_ = Eigen::Map<const VectorXd>(p[0], numParams_);
		 }

		 return true;
	 }

	 const double* operator()() const {
		 return x_;
	 }

 private:
	 ICeresModel *dm_;
	 const double *x_;
	 const double *y_;
	 int numResiduals_;
	 int numParams_;

	 VectorXd *pBestParams_;
	 double *pBestEval_;
	 mutable double bestEval_;
 };


 static void STDCALL NotifyGenerateProgress(void *aargs, double progress) {
	 fitJobArgs *args = (fitJobArgs *)aargs;
	 if (!args)
		 return;

	 Job job = JobManager::GetInstance().GetJobInformation(args->jobID);

	 job.progress = progress;
	 JobManager::GetInstance().UpdateJob(job);

	 if (args->fp.bProgressReport && args->backend)
		 args->backend->NotifyProgress(args->jobID, progress);
 }
#pragma endregion Helper Functions

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// Actual functions

/**
 *  Fits a form factor with the possibility to update a graph.
 *  ffx:            Input data X vector
 *  ffy:            Input data Y vector
 *  my:             Output model (may also be pre-filled with
 *                  existing structure factor to fit with)
 *  bgy:            Input additive Y background vector
 *  p:              Input/Output parameters data structure
 *  GraphModify:    A callback used to update the graph while fitting
 *  pStop:          A pointer to a stop signal - 0 means run, 1 means stop
 *  gauss:			A flag to indicate use of a gaussian or discrete model
 *  ProgressReport: A callback used to send back the progress
 **/
ErrorCode PerformModelFitting(fitJobArgs *args) {
	// Scheme:
	// 0. Argument validation (e.g., requested GPU but backend doesn't have it, return ERROR_UNSUPPORTED)
	if(!args)
		return ERROR_INVALIDARGS;

	Job job = JobManager::GetInstance().GetJobInformation(args->jobID);
	
	bool success = true;

	// 0.5 Masking
	// Check to see if there are masked elements and crop them from the relevant vectors
	// TODO::mask Mask existing vectors (i.e. ffY, sfY, bgY)
	std::vector<double> datay = args->y, datax = args->x;
	if(args->mask.size() == datay.size()) {
		for(int k = (int)args->mask.size() - 1; k >= 0; k--) {
			if(args->mask.at(k)) {
				datax.erase(datax.begin() + k);
				datay.erase(datay.begin() + k);
			} // if
		} // for
	} // if
	int size = (int)datay.size();
	
	ParameterTree pt = *job.tree;
	if(job.uidToModel.find(pt.GetNodeModel()) == job.uidToModel.end())
		return ERROR_MODELNOTFOUND;

	IModel *topModel = job.uidToModel[pt.GetNodeModel()];

	VectorXd p;
	VectorXi pMut;
	cons pMin, pMax;

	// 1. Create vector p, pmut from the model/parameter tree.
	// Create polydisperse models and model modifiers (electron density profile wrappers) 
	// as necessary. Return new (or old) fit model.
	IModel *finalModel = CreateModelFromParamTree(&job, pt, job.pStop, p, pMut, pMin, pMax);
	if(!finalModel)
		return ERROR_INVALIDARGS;

#define CERES_FITTING

#ifndef CERES_FITTING
	if(args->fp.logScaleFitting)
		for(int i = 0; i < size; i++)
			datay[i] = log10(datay[i]);

	// Initializing weight function: [w(x) = sqrt(x) + 1]
	// Avi: Is this the reason logScaleFitting doesn't work well?
	std::vector<double> weights(size, 0.0);
	for(int i = 0; i < size; i++)
		weights[i] = (sqrt(fabs(datay[i])) + 1.0);
#endif

	// 3. FITTING
	bool bStopped = false;

#ifdef CERES_FITTING

	ICeresModel *dm = dynamic_cast<ICeresModel*>(finalModel);
	finalModel->SetStop(job.pStop);

	Eigen::Map<VectorXd> eigX(args->x.data(), args->x.size());
	Eigen::Map<VectorXd> eigY(args->y.data(), args->y.size());
	std::vector<int> mutPIndices;
	std::vector<double> mutPMin, mutPMax;

	for(int i = 0; i < pMut.size(); i++) {
		if(pMut[i] == 1) {
			mutPIndices.push_back(i);	// Collect the indices of the mutable parameters
			mutPMax.push_back(pMax.num[i]);
			mutPMin.push_back(pMin.num[i]);
		}
	}

	// If there are no mutable variables, then what exactly are we trying to do?
	if (mutPIndices.size() == 0)
	{
		throw backend_exception(ERROR_NO_MUTABLES_FOR_FIT, g_errorStrings[ERROR_NO_MUTABLES_FOR_FIT]);
	}

	// Check all the constraints to make sure that the mutables are within them
	{
		bool within_constraints = true;
		std::string bad_parameters = "Index\t\tValue\t\tMin\t\tMax\n";
		for (int i = 0; i < mutPIndices.size(); i++)
		{
			double param_value = p[mutPIndices[i]];
			if (param_value < pMin.num[i] || param_value > pMax.num[i] || pMin.num[i] >= pMax.num[i])
			{
				std::string bad_line =
					std::to_string(i) + "\t\t" +
					std::to_string(param_value) + "\t\t" +
					std::to_string(pMin.num[i]) + "\t\t" +
					std::to_string(pMax.num[i]) + "\n";
				bad_parameters.append(bad_line);
				within_constraints = false;
			}
		}

		std::string exception_text = std::string(g_errorStrings[ERROR_PARAMETER_NOT_WITHIN_CONSTRAINTS]) + "\n" + bad_parameters;

		// Throw backend_exception
		if (!within_constraints)
			throw backend_exception(ERROR_PARAMETER_NOT_WITHIN_CONSTRAINTS, exception_text.c_str());
	}

	// Creates a map that points to all the mutable parameters (as a member within dm/finalModel)
	dm->SetInitialParamVecForCeres(&p, mutPIndices);

	Eigen::Map<VectorXd> eigMin(mutPMin.data(), mutPMin.size());
	Eigen::Map<VectorXd> eigMax(mutPMax.data(), mutPMax.size());

	GetCeresVecCostFunction_t costFunc;

	switch (args->fp.ceresProps.residualType)
	{
	case XRayResidualsType_Enum::NORMAL_RESIDUALS:
		costFunc = XRayResiduals::GetCeresCostFunction;
		break;

	case XRayResidualsType_Enum::RATIO_RESIDUALS:
		costFunc = XRayRatioResiduals::GetCeresCostFunction;
		break;

	case XRayResidualsType_Enum::LOG_RESIDUALS:
		costFunc = XRayLogResiduals::GetCeresCostFunction;
		break;
	default:
		break;
	}

	IOptimizationMethod * opt;
	opt = new CeresOptimizer(costFunc, args->fp.ceresProps,
		mutPIndices.size(), eigX,
		eigY, args->fp.fitIterations,
		eigMin, eigMax,
		dm, args->fp.ceresProps.derivativeStepSize,
		args->fp.ceresProps.fittingConvergence * eigY.mean() * 1e-3,
		args->fp.ceresProps.derivativeEps);

	VectorXd mutP(mutPIndices.size());
	for(int i = 0; i < mutPIndices.size(); i++) {
		mutP[i] = p[mutPIndices[i]];
	}

	progressFunc prog = (args->fp.bProgressReport ? &NotifyGenerateProgress : NULL);

	if (prog)
		prog(args, 0.01);

	double gof = std::numeric_limits<double>::infinity();
	VectorXd pnew;
	ProgressFromCeres cprog(NULL, NULL, args->fp.fitIterations);
	if (prog)
	{
		cprog = ProgressFromCeres(prog, args, args->fp.fitIterations);
		CeresOptimizer* as_ceres_opt = (dynamic_cast<CeresOptimizer*>(opt));
		if(as_ceres_opt)
			as_ceres_opt->AddCallback(&cprog);
	}
	gof = opt->Iterate(mutP, pnew);
	mutP = pnew;
	printf("Iteration GoF = %f\n", gof);

#else
	std::vector<double> ones(datay.size(), 1.0), zeros(datay.size(), 0.0);	
	ModelFitter *fitter = CreateFitter(args->fp.method, finalModel, args->fp, datax, datay, ones, zeros,
						weights, p, pMut, pMin, pMax, 3);

	if(fitter->GetError()) {
		delete fitter;
		if(finalModel != topModel)
			delete finalModel;

		return ERROR_NOMUTABLES;		
	}

	
	// The main fitting loop (each iteration yields a different parameter structure)
	for(int i = 0; i < args->fp.fitIterations; i++) {
		fitter->FitIteration();
		p = fitter->GetResult();

		if(args->fp.liveFitting) {
			VectorXd y = fitter->GetInterimRes();
			job.resultGraph.resize(y.size());
			memcpy(&job.resultGraph[0], y.data(), y.size() * sizeof(double));			
		}
	
		// Update parameter tree
		job.tree->FromParamVector(p.data(), p.size());

		job.progress = (double)i / (double)(args->fp.fitIterations);
		JobManager::GetInstance().UpdateJob(job);

		// Graph update during fitting
		if(args->fp.bProgressReport && args->backend)
			args->backend->NotifyProgress(args->jobID, job.progress, args->fp.msUpdateInterval);
		

		if(job.pStop && *job.pStop) {
			bStopped = true;
			break;
		}

		if(fitter->GetError()) {
			success = false; //set flag for ERROR_FITERROR?
			break;
		}
			
	}
#endif // CERES_FITTING

	///////////////////////////////////////////////////////////////////////////////
	// Finalization
	
	// a. Calculate errors
	// b. clear stopped signal
	
	// c. return new parameters to the paramStructs
	job.tree->FromParamVector(p.data(), p.size());

	// d. generate final graph and update job
#ifndef CERES_FITTING
	VectorXd y = fitter->GetInterimRes();
#else
	VectorXd y = finalModel->CalculateVector(args->x, 0, p);
#endif // !CERES_FITTING
	job.resultGraph.resize(y.size());
	memcpy(&job.resultGraph[0], y.data(), y.size() * sizeof(double));		
	JobManager::GetInstance().UpdateJob(job);


	// Destroy any remnants of model modifiers
	if(finalModel != topModel)
		delete finalModel;

#ifdef CERES_FITTING
	delete opt;
#else
	delete fitter;
#endif // !CERES_FITTING

	// 4. If stopped, return ERROR_STOPPED. Else, done!
	if(bStopped)
		return ERROR_STOPPED;
	if(!success && ! bStopped)
		return ERROR_UNSUCCESSFULLFITTING;

	return OK;
}





ErrorCode PerformModelGeneration(fitJobArgs *args) {
	// Scheme:
	// 0. Argument validation (e.g., requested GPU but backend doesn't have it, return ERROR_UNSUPPORTED)
	if(!args)
		return ERROR_INVALIDARGS;

	Job job = JobManager::GetInstance().GetJobInformation(args->jobID);

	ParameterTree pt = *job.tree;
	if(job.uidToModel.find(pt.GetNodeModel()) == job.uidToModel.end())
		return ERROR_MODELNOTFOUND;

	IModel *topModel = job.uidToModel[pt.GetNodeModel()];

	VectorXd p;
	VectorXi pMut;
	cons pMin, pMax;

	// 1. Create vector p, pmut from the model/parameter tree.
	// Create polydisperse models and model modifiers (electron density profile wrappers) 
	// as necessary. Return new (or old) fit model.
	
	IModel *finalModel = CreateModelFromParamTree(&job, pt, job.pStop, p, pMut, pMin, pMax);

	if(!finalModel)
		return ERROR_INVALIDARGS;
	
	// 2. TODO::GPU-- (later) If GPU is requested and (the model and GPU itself are) available, compute in GPU
	/* TO BE REPLACED BY 2
	// GPU generation overrides loop
	if(isGPUBackend() && args->gp.bGPU && finalModel->HasGPU) {
		VectorXd& y = finalModel->GPUCalculate(x, guessLayers, guess);

		#pragma omp parallel for
		for(int i = 0; i < y.size(); i++)
			genY[i] = y[i];

		// Destroy any remains of model modifiers
		if(finalModel != p->model)
			delete finalModel;

		return OK;
	}*/



	// 3. GENERATION
	//   Call CalculateVector
	bool bStopped = false;


	VectorXd y = finalModel->CalculateVector(args->x, 0, p,
		(args->fp.bProgressReport ? &NotifyGenerateProgress : NULL), args);
	if(job.pStop && *job.pStop)
		bStopped = true;

	job.resultGraph.resize(y.size());

	Eigen::Map<VectorXd>(job.resultGraph.data(), y.size()) = y;

	job.progress = 1.0;

	JobManager::GetInstance().UpdateJob(job);
	

	// Destroy any remnants of model modifiers
	if(finalModel != topModel)
		delete finalModel;
	
	// 4. If stopped, return ERROR_STOPPED. Else, done!
	if(bStopped)
		return ERROR_STOPPED;

	return OK;
}
