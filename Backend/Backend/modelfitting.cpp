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

/**
 * @file modelfitting.cpp
 * @brief Implements model fitting and generation logic, including parameter vector construction,
 *        constraint handling, recursive model setup, and job-based model generation in the backend.
 *
 * The model fitting module is responsible for:
 *  - Translating parameter trees into parameter vectors and constraint structures for fitting.
 *  - Recursively setting up models and amplitudes, including composite and symmetry models.
 *  - Managing parameter mutability, constraints, and linking for fitting algorithms.
 *  - Creating polydisperse model wrappers as needed based on parameter tree configuration.
 *  - Handling job-based model generation and result storage (1D and 2D).
 *  - Integrating with the job management system for progress reporting and interruption.
 *
 * Key Concepts:
 *  - ParameterTree: Represents the hierarchical structure of model parameters, constraints, and links.
 *  - Job: Encapsulates the state and resources for a single fitting or generation task.
 *  - IModel: Abstract interface for all models, supporting calculation and parameterization.
 *  - cons: Structure holding parameter constraints (min, max, index, link).
 *  - VectorXd, VectorXi: Eigen vectors for parameter values and mutability flags.
 *  - PolydisperseModel: Decorator for models supporting polydispersity in parameter space.
 *
 * Fields:
 *  - Parameter (struct, from ParameterTree):
 *      - double value: Parameter value.
 *      - bool isMutable: Indicates if the parameter is mutable during fitting.
 *      - bool isConstrained: Indicates if the parameter has constraints.
 *      - double consMin, consMax: Absolute minimum and maximum constraints.
 *      - int consMinIndex, consMaxIndex: Indices for relative constraints.
 *      - int linkIndex: Index for parameter linking.
 *  - cons (struct):
 *      - std::vector<double> num: Numeric constraint values (min/max).
 *      - std::vector<int> index: Indices for relative constraints.
 *      - std::vector<int> link: Indices for parameter linking.
 *      - cons(int n): Constructor initializing vectors to size n.
 *  - Job:
 *      - std::map<unsigned int, IModel*> uidToModel: Maps model handles to model instances.
 *      - std::map<unsigned int, Amplitude*> uidToAmp: Maps amplitude handles to amplitude instances.
 *      - ParameterTree* tree: Pointer to the parameter tree for the job.
 *      - int* pStop: Pointer to an integer flag for job interruption.
 *      - std::vector<double> resultGraph: Stores the 1D result graph.
 *      - Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> resultGraph2D: Stores the 2D result graph.
 *      - double progress: Progress of the job (0.0–1.0).
 *      - unsigned int jobID: Unique identifier for the job.
 *  - fitJobArgs:
 *      - int jobID: Job identifier.
 *      - std::vector<double> x: Input data points for calculation.
 *      - FittingProperties fp: Fitting options and progress reporting flags.
 *      - LocalBackend* backend: Pointer to backend for progress notification.
 *  - VectorXd, VectorXi (Eigen):
 *      - VectorXd: Dynamic-size vector of doubles (parameter values).
 *      - VectorXi: Dynamic-size vector of integers (mutability flags).
 *
 * Main Methods:
 *  - ParameterToVectorIndex: Converts a Parameter to its vector representation and constraint structures.
 *  - RecursiveModelSetter: Recursively sets up models and amplitudes from a parameter tree.
 *  - CreateModelFromParamTree: Constructs a model and its parameter vectors/constraints from a parameter tree.
 *  - PerformModelGeneration: Generates a 1D result graph for a job using the constructed model.
 *  - PerformModelGeneration2D: Generates a 2D result graph for a job using the constructed model.
 *
 * Threading and Safety:
 *  - Stop signals are propagated to models for interruptible calculations.
 *  - Job state and progress are managed via the JobManager singleton.
 *
 * Error Handling:
 *  - Returns error codes for invalid arguments, missing models, or interrupted jobs.
 *  - Cleans up dynamically allocated model wrappers and resources.
 *
 * Dependencies:
 *  - Eigen for vector and matrix operations.
 *  - JobManager for job state management.
 *  - Geometry, Amplitude, and fittingfactory for model construction.
 *  - mathfuncs for utility functions.
 *
 * See modelfitting.h for class and method declarations.
 */




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


ErrorCode PerformModelGeneration2D(fitJobArgs* args) {

	// Scheme:
	// 0. Argument validation (e.g., requested GPU but backend doesn't have it, return ERROR_UNSUPPORTED)
	if (!args)
		return ERROR_INVALIDARGS;

	Job job = JobManager::GetInstance().GetJobInformation(args->jobID);

	ParameterTree pt = *job.tree;
	if (job.uidToModel.find(pt.GetNodeModel()) == job.uidToModel.end())
		return ERROR_MODELNOTFOUND;

	IModel* topModel = job.uidToModel[pt.GetNodeModel()];

	VectorXd p;
	VectorXi pMut;
	cons pMin, pMax;

	// 1. Create vector p, pmut from the model/parameter tree.
	// Create polydisperse models and model modifiers (electron density profile wrappers) 
	// as necessary. Return new (or old) fit model.

	IModel* finalModel = CreateModelFromParamTree(&job, pt, job.pStop, p, pMut, pMin, pMax);

	if (!finalModel)
		return ERROR_INVALIDARGS;

	



	// 3. GENERATION
	//   Call CalculateMatrix
	bool bStopped = false;

	MatrixXd y = finalModel->CalculateMatrix(args->x, 0, p,
		(args->fp.bProgressReport ? &NotifyGenerateProgress : NULL), args);
	if (job.pStop && *job.pStop)
		bStopped = true;
	job.resultGraph2D = y; 

	job.progress = 1.0;

	JobManager::GetInstance().UpdateJob(job);

	// Destroy any remnants of model modifiers
	if (finalModel != topModel)
		delete finalModel;

	// 4. If stopped, return ERROR_STOPPED. Else, done!
	if (bStopped)
		return ERROR_STOPPED;

	return OK;
}
