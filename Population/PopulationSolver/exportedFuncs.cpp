#include "exportedFuncs.h"

int m_FitIterations = 20;
bool m_logFitting = false;

int GetFitIterations() { return m_FitIterations; }
bool isLogFitting() { return m_logFitting; }
void SetFitIterations(int value) { m_FitIterations = value; }
void SetLogFitting(bool value) { m_logFitting = true; }



bool FitCoeffs(const ArrayXd& datay, const ArrayXd& datax, const std::string expression,
						const ArrayXXd& curvesX, const ArrayXXd& curvesY, ArrayXd& p, const ArrayXi& pmut,
						cons *pMin, cons *pMax, ArrayXd& paramErrors, ArrayXd& modelErrors) {
	bool success = true;
	int layers = p.size();
	ArrayXd dataY = datay;
	ArrayXXd YCurves = curvesY;
	ArrayXXd XCurves = curvesX;

	//// Check to see if there are masked elements and crop them from relevant vectors
	//if(mask.size() == inffx.size()) {
	//	for(int k = mask.size() - 1; k >= 0; k--) {
	//		if(mask.at(k)) {
	//			bgy.erase(bgy.begin() + k);
	//			ffx.erase(ffx.begin() + k);
	//			ffy.erase(ffy.begin() + k);
	//			my.erase(my.begin() + k);
	//		}	//if
	//	}	//for
	//}	//if


	//////////////////////////////////////////////////////////////////////////
	// Initialization

	// Initializing vectors
//	int extraParams = p->extraParams.size();
	int ma = layers;//(layers * p->model->GetNumLayerParams()) + extraParams, ndata;
	
	ArrayXd a  = ArrayXd::Zero(ma); // Parameter vector
	ArrayXi ia = ArrayXi::Zero(ma); // Mutability vector
	cons a_min(ma); // Fit range/constraint vector
	cons a_max(ma); // Fit range/constraint vector
	//Model *finalModel = p->model;

	//// Initializing additional GUI parameters
	//SetSignal(pStop);
	//p->model->SetStop(pStop);
	//finalModel->SetStop(pStop);


	//// Initializing parameter vector from our input vectors	
	//for(int i = 0; i < p->model->GetNumLayerParams(); i++)
	//	for(int j = 0; j < layers; j++) {
	//		ParameterToVectorIndex(p->params[i][j], i * layers + j, a, ia, 
	//							   a_min, a_max);
	//	}

	//// Initializing extra parameters
	//for(int i = 0; i < extraParams; i++) {
	//	ParameterToVectorIndex(p->extraParams[i], ma - extraParams + i, a, ia, 
	//						   a_min, a_max);
	//}

	// Initializing graph vectors
	//ArrayXd y = dataY;
	//ArrayXXd x = YCurves;
//	ndata = x.rows();
	
	// TODO
	//if(isLogFitting())
	//	y = y.log10();
		//for(int i = 0; i < (int)y.size(); i++)
		//	y[i] = log10(y[i]);

	// Initializing weight function: [w(x) = sqrt(x) + 1]
	//vector<double> weights (ndata, 0.0);
	//for(int i = 0; i < ndata; i++)
	//	weights[i] = (sqrt(fabs(y[i])) + 1.0);
//	weights = dataY.abs().sqrt() + 1.0;

	// Adjust linked parameter indices and relative constraints
	//int nlp = p->model->GetNumLayerParams();
	//for(int i = 0; i < nlp; i++) {
	//	for(int j = 0; j < layers; j++) {
	//		if(a_max.link[i * layers + j] >= 0)
	//			a_max.link[i * layers + j] += i * layers;
	//		if(a_max.index[i * layers + j] >= 0)
	//			a_max.index[i * layers + j] += i * layers;
	//		if(a_min.index[i * layers + j] >= 0)
	//			a_min.index[i * layers + j] += i * layers;
	//	}
	//}

	//if(!p->bConstrain) {
	//	a_min.num	= VectorXd::Constant(a_min.num.size(), -std::numeric_limits<double>::infinity());
	//	a_max.num	= VectorXd::Constant(a_max.num.size(), std::numeric_limits<double>::infinity());
	//	a_max.index	= VectorXi::Constant(a_max.num.size(), -1);
	//	a_min.index	= VectorXi::Constant(a_min.num.size(), -1);

	//}

	// Initializing Levenberg-Marquardt fitter
	LMPopFitter *fitter = new LMPopFitter(dataY, datax, expression, XCurves, YCurves, p, pmut, pMin, pMax, paramErrors, modelErrors, p.size());

	// No mutables
	if(fitter->GetError()) {
		delete fitter;

//		my.clear();

		//// Destroy any remains of model modifiers
		//if(finalModel != p->model)
		//	delete finalModel;

		return false; //GenerateModel(inffx, my, p, pStop);
	}

	// Initializing visual objects
	//ArrayXd intermY (ndata, 0.0); // Intermediate model for realtime graph plotting

	//////////////////////////////////////////////////////////////////////////
	// Fitting

	// The main fitting loop (each iteration yields a different parameter structure)
	for(int i = 0; i < m_FitIterations; i++) {
		
		// Fitting
		modelErrors(0) = fitter->FitIteration();
		a = fitter->GetResult();

		//if(pStop && *pStop)
		//	break;
		if(fitter->GetError()) {
			success = false;
			break;
		}

		// Graph update during fitting
		//if(GraphModify) {
		//	VectorXd guess = a;
		//	int guessLayers = layers;

		//	// The new layer model is created based on the Electron Density Profile
		//	EDPFunction *edp = finalModel->GetEDProfile().func;
		//	if(finalModel->IsLayerBased() && edp && ffx.size() > 1)
		//		guess = edp->ComputeParamVector(finalModel, a, ffx, layers, guessLayers);
		//	// END of profile reshaping

		//	VectorXd tmp = fitter->GetInterimRes();
		//	for(int r = 0; r < int(my.size()); r++)
		//		intermY[r] = tmp(r);

		//	// Modifying the generated graph
		//	GraphModify(ffx, intermY);
		//}

		// Progress Report
		//if(ProgressReport)
		//	ProgressReport(int(double(i) / double(GetFitIterations()) * 100.0));
	}

	//////////////////////////////////////////////////////////////////////////
	// Finalization

	//fitter->calcErrors();

	//for(int bbq = 0; bbq < ia.size(); bbq++) {
	//	if(ia[bbq] == 0)
	//		paramErrors.insert(paramErrors.begin() + bbq, -1.0);
	//}

	//if(!pStop || (pStop && !*pStop))
	//	success = !fitter->GetError();

	delete fitter;

	// Clearing the stop signal so it won't interrupt us while we generate the final model
	//ClearSignal();
	p = a;
	// Saving back the parameters and extra parameters to the paramStruct
	//for(int i = 0; i < p->model->GetNumLayerParams(); i++)
	//	for(int j = 0; j < layers; j++)
	//		p->params[i][j].value = a[i * layers + j];

	//for(int i = 0; i < extraParams; i++)
	//	p->extraParams[i].value = a[ma - extraParams + i];

	// After fitting the model, generate the final graph to show to the user
	//my.clear();
	//if(pStop && *pStop != 2) {	// == 2 is set only when the FF window is closed
	//	*pStop = 0;		// So that the generate will work for the slower models

	//	GenerateModel(inffx, my, p, pStop);
	//}

	// Destroy any remains of model modifiers
	//if(finalModel != p->model)
	//	delete finalModel;

	return success;
}
