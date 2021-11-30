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
