//#define NOMINMAX
//#include "Windows.h"	//For messagebox debugging
#include <iostream>

#include "Geometry.h"
#include "Quadrature.h" // For Quadrature

#include "Eigen/LU" // For matrix inverse
#include "mathfuncs.h" // For gaussianSig


/**
 * @file Geometry.cpp
 * @brief Implements the Geometry class hierarchy and related logic for geometric model parameterization,
 *        electron density profile management, and intensity calculation in the backend.
 *
 * The geometry module is responsible for:
 *  - Defining the Geometry base class and its derivatives (e.g., PolydisperseModel, LuaModel, FFModel).
 *  - Managing geometric model parameters, including layer-based and extra parameters.
 *  - Supporting electron density (ED) profile configuration and dynamic profile functions.
 *  - Providing methods for vectorized and matrix-based intensity calculations, with optional OpenMP parallelization.
 *  - Organizing and validating parameter vectors for model calculations.
 *  - Supporting numerical differentiation for parameter sensitivity analysis.
 *  - Integrating with Eigen for linear algebra and OpenMP for parallel computation.
 *
 * Key Concepts:
 *  - Geometry: Abstract base class for all geometric models, handling parameter organization, ED profiles, and calculation routines.
 *  - PolydisperseModel: Decorator for Geometry models, supporting polydispersity via convolution with distribution functions.
 *  - LuaModel: Geometry model defined by user-provided Lua scripts, supporting dynamic model logic.
 *  - FFModel: Specialized geometry model with additional handling for form factor calculations and extra parameters.
 *  - Parameter Organization: Parameters are organized into layer-based and extra parameters, with support for dynamic ED profiles.
 *  - Electron Density Profile: Supports static and dynamic ED profiles, with runtime configuration and memory management.
 *  - Parallelization: Uses OpenMP for parallel evaluation of intensity vectors and matrices.
 *  - Numerical Derivatives: Provides high-accuracy numerical differentiation for model parameters.
 *
 * Fields:
 *  - std::string modelName:
 *      Name of the geometric model.
 *  - int nLayerParams:
 *      Number of parameters per layer in the model.
 *  - int nExtraParams:
 *      Number of extra (non-layer) parameters.
 *  - int minLayers, maxLayers:
 *      Minimum and maximum number of layers supported by the model.
 *  - int displayParams:
 *      Number of parameters shown in the UI or for display.
 *  - EDProfile profile:
 *      Electron density profile configuration for the model.
 *  - EDPFunction* profileFunc:
 *      Pointer to the function object for the electron density profile.
 *  - MatrixXd* parameters:
 *      Matrix of organized layer parameters (rows: layers, cols: parameters).
 *  - VectorXd* extraParams:
 *      Vector of extra parameters not associated with layers.
 *  - bool bParallelizeVector:
 *      Flag indicating if vector calculations should be parallelized (OpenMP).
 *  - int* pStop:
 *      Pointer to an external stop flag for interrupting calculations.
 *  - void* GPUKernel:
 *      Pointer to GPU kernel or context (if GPU acceleration is used).
 *  - PolydisperseModel* model (in PolydisperseModel):
 *      Pointer to the inner model being decorated for polydispersity.
 *  - int polyInd (in PolydisperseModel):
 *      Index of the parameter subject to polydispersity.
 *  - double polySigma (in PolydisperseModel):
 *      Standard deviation for the polydispersity distribution.
 *  - int pdResolution (in PolydisperseModel):
 *      Number of points for polydispersity integration.
 *  - int pdFunction (in PolydisperseModel):
 *      Shape of the polydispersity distribution (e.g., Gaussian, Lorentzian).
 *  - std::string modelCode (in LuaModel):
 *      Lua script code defining the model.
 *  - void* luactx (in LuaModel):
 *      Pointer to the Lua context/environment.
 *  - bool bContextCreated (in LuaModel):
 *      Indicates if the Lua context was created internally.
 *  - VectorXd parVec (in LuaModel):
 *      Stores the parameter vector for the Lua model.
 *
 * Main Methods:
 *  - Geometry (constructor/destructor): Initializes geometry state, parameters, and ED profile function.
 *  - Get/Set Methods: Accessors for model name, parameter counts, layer names, and ED profile configuration.
 *  - OrganizeParameters: Arranges parameter vectors into layer and extra parameter matrices for calculation.
 *  - PreCalculate: Prepares model state before calculation (can be overridden by subclasses).
 *  - CalculateVector/Matrix: Computes intensity vectors or matrices for a set of q-values, with progress reporting and parallelization.
 *  - GPUCalculate: Placeholder for GPU-accelerated calculation (not implemented in base class).
 *  - Derivative/NumericalDerivative: Computes numerical derivatives of intensity with respect to parameters.
 *  - SetEDProfile: Configures the electron density profile and updates parameter organization.
 *  - GetAllParameters: Flattens all organized parameters into a single vector for serialization or further processing.
 *  - OrientationAverage: Computes orientation-averaged intensity for anisotropic models.
 *
 * Threading and Safety:
 *  - OpenMP is used for parallelization of vector and matrix calculations.
 *  - Geometry objects are not inherently thread-safe; synchronization is managed externally.
 *  - Progress reporting and stop flags are supported for long-running calculations.
 *
 * Error Handling:
 *  - Parameter bounds and applicability are checked before use.
 *  - NaN and invalid results are detected and flagged during calculation.
 *  - Memory management for dynamic ED profiles and parameter matrices is handled in destructors.
 *
 * Dependencies:
 *  - Eigen for linear algebra and matrix operations.
 *  - OpenMP for parallelization of calculations.
 *  - mathfuncs.h for distribution functions (e.g., gaussianSig, lorentzian).
 *  - Quadrature.h for numerical integration support.
 *  - Lua (optional) for scriptable model definitions.
 *
 * See Geometry.h for class and method declarations.
 */


Geometry::Geometry(std::string name, int extras, int nlp, 
			 int minlayers, int maxlayers, EDProfile edp, int disp) : 
			GPUKernel(NULL), nLayerParams(nlp),
			minLayers(minlayers), maxLayers(maxlayers),
			modelName(name), nExtraParams(extras), profile(edp), profileFunc(NULL),
			displayParams(disp), parameters(NULL), extraParams(NULL), IModel() {
				bParallelizeVector = true;
}


Geometry::~Geometry() {
	// Delete a stray arbitrary ED profile, if exists
	if(profileFunc) {
		delete profileFunc;
		profileFunc = NULL;
	}
	if(parameters) {
		delete parameters;
		parameters = NULL;
	}
	if(extraParams) {
		delete extraParams;
		extraParams = NULL;
	}

}


///// Get/Set Methods

std::string Geometry::GetName() {
	return modelName;
}

int Geometry::GetNumLayerParams() {
	return nLayerParams;
}

int Geometry::GetNumExtraParams() {
	return nExtraParams;
}

bool Geometry::IsLayerBased() {
	return true;
}

int Geometry::GetMinLayers() { 
	return minLayers; 
}

int Geometry::GetMaxLayers() { 
	return maxLayers; 
}

int Geometry::GetNumDisplayParams() {
	return displayParams;
}

std::string Geometry::GetLayerParamName(int index, EDPFunction *edpfunc) {
	switch(index) {
		default:
			// TODO::EDProfile: Removed due to the change to static (cannot use 
			// nLayerParams in a static function)
			/*if(edpfunc) {
				int edpparams = (nLayerParams - edpfunc->GetNumEDParams());

				if(index >= edpparams && index < nLayerParams)
					return edpfunc->GetEDParamName(index - edpparams);
			}*/

			return "N/A";
		case 0:
			return "Radius";
		case 1:
			return "E.D.";
	}
}

ExtraParam Geometry::GetExtraParameter(int index) {
	if(index < 0 || index >= 2)
		return ExtraParam("N/A");

	switch(index) {
		case 0:
			return ExtraParam("Scale", 1.0);

		case 1:
			return ExtraParam("Background", 0.0);

		default:
			return ExtraParam("Unimplemented");
	}
}

bool Geometry::IsParamApplicable(int layer, int lpindex) {
	if(layer < 0 || lpindex < 0)
		return false;
	// first layer = solvent , lpindex == 0 is radius
	if (layer == 0 && lpindex == 0)
		return false;
	return true;
}

std::string Geometry::GetLayerName(int layer) {
	if(layer < 0)
		return "N/A";

	if(layer == 0)
		return "Solvent";

	return "Layer %d"; // The Frontend will fill the actual layer number
}

void Geometry::SetStop(int *stop) { 
	pStop = stop; 
}

EDProfile Geometry::GetEDProfile() {
	return profile;
}

std::string Geometry::GetDisplayParamName(int index) {
	// Override this function in subclasses
	return "";
}

bool Geometry::ParallelizeVector() {
	return bParallelizeVector;
}

double Geometry::GetDisplayParamValue(int index, const paramStruct *p) {
	// Override this function in subclasses
	return -1.0;
}

double Geometry::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
	switch(paramIndex) {
		default:
			// TODO::EDProfile: Removed due to the change to static (cannot use 
			// nLayerParams in a static function)
			/*if(edpfunc) {
				int edpparams = (nLayerParams - edpfunc->GetNumEDParams());

				if(paramIndex >= edpparams && paramIndex < nLayerParams)
					return edpfunc->GetEDParamDefaultValue(paramIndex - edpparams, layer);
			}*/
			// FALLBACK

		case 0:
			// Radius
			if(layer == 0)
				return 0.0;
			
			return 1.0;

		case 1:
			// Electron Density
			if(layer == 0)
				return 333.0;

			return 400.0;
	}
}

///// Calculation Methods

void Geometry::PreCalculate(VectorXd& p, int nLayers) {
}



VectorXd Geometry::GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p) {

	// If there is no GPU backend, return nothing
	//if(!isGPUBackend())
		return VectorXd();

	/*
	PreCalculate(p, nLayers);

	VectorXf eigenX, eigenParams, eigenY;
	
	eigenX = VectorXf::Zero(q.size());
	eigenParams = p.cast<float>();
	eigenY = VectorXf::Zero(q.size());
	
	for(int i = 0; i < (int)q.size(); i++)
		eigenX[i] = (float)q[i];

	if(!GenerateGPUModel(GPUKernel, eigenX, eigenParams, eigenY, p.size(), nExtraParams))
		return VectorXd();

	return eigenY.cast<double>();
	*/
}

VectorXd Geometry::CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p, 
								progressFunc progressReport, void *progressArgs) {
	VectorXd res (q.size());
	PreCalculate(p, nLayers);
    
    int size = (int)q.size();
	bool error = false;
	int progress = 0;

	// When CailleModel is extended, use this code piece before calling Geometry::CalculateVector
	/*if(GetPeakType() == SHAPE_CAILLE) 
		SetX(x);*/
    
    // 1st tier of parallelization
#pragma omp parallel for shared(progress) if(bParallelizeVector)//if(GetPeakType() != SHAPE_CAILLE)
    for (int i = 0; i < size; i++) {
		progress++;

		// Report progress
		if(progressReport)
			progressReport(progressArgs, (double)progress / (double)size);		

        double cury;
		if(error)
			continue;
        
        if(pStop && *pStop) {
			error = true;
			continue;
		}
		Eigen::VectorXd dummy;
		// we shouldn't have to do this; however, using "cury = Calculate(q[i],  nLayers);"
		// causes an assertion failure (in Eigen because of the VectorXd() and there is 
		// a line in Eigen stating "ei_assert(dim > 0);" we therefore shouldn't have a 0
		// dimension VectorXd. Any other solutions?)
		cury = Calculate(q[i],  nLayers, dummy);
		if(cury != cury) {
	        error = true;
			continue;
		}

		res[i] = cury;		
    }
	
	return res;
}

MatrixXd Geometry::CalculateMatrix(const std::vector<double>& q, int nLayers, VectorXd& p,
	progressFunc progressReport, void* progressArgs) 
{
	VectorXd res(q.size());
	PreCalculate(p, nLayers);

	int size = (int)q.size();
	bool error = false;
	int progress = 0;

	// When CailleModel is extended, use this code piece before calling Geometry::CalculateVector
	/*if(GetPeakType() == SHAPE_CAILLE)
		SetX(x);*/

		// 1st tier of parallelization
#pragma omp parallel for shared(progress) if(bParallelizeVector)//if(GetPeakType() != SHAPE_CAILLE)
	for (int i = 0; i < size; i++) {
		progress++;

		// Report progress
		if (progressReport)
			progressReport(progressArgs, (double)progress / (double)size);

		double cury;
		if (error)
			continue;

		if (pStop && *pStop) {
			error = true;
			continue;
		}
		Eigen::VectorXd dummy;
		// we shouldn't have to do this; however, using "cury = Calculate(q[i],  nLayers);"
		// causes an assertion failure (in Eigen because of the VectorXd() and there is 
		// a line in Eigen stating "ei_assert(dim > 0);" we therefore shouldn't have a 0
		// dimension VectorXd. Any other solutions?)
		cury = Calculate(q[i], nLayers, dummy);
		if (cury != cury) {
			error = true;
			continue;
		}

		res[i] = cury;
	}

	return res;
}

// Numerical derivation helper function
static inline VectorXd derF(IModel *mod, const std::vector<double>& x, VectorXd& p, 
							int nLayers, int ai, double h, double m) {  
	VectorXd pDummy, res = VectorXd::Zero(x.size());
	int size = (int)x.size();

	p[ai] += h;

	// Create copies for the parameter vector and the number of layers for
	// this iteration
	VectorXd guess = p;
	int guessLayers = nLayers;

	mod->PreCalculate(guess, guessLayers);
	
	VectorXd tmp = mod->CalculateVector(x, guessLayers, guess);
	for(int i = 0; i < size; i++)
		res[i] = m * tmp(i);

	p[ai] -= h;

	return res;
}

VectorXd NumericalDerivative(IModel *mod, const std::vector<double>& x, VectorXd param,
							 int nLayers, int ai, double epsilon) {
	double h = epsilon;

	// f'(x) ~ [f(x-2h) - f(x+2h)  + 8f(x+h) - 8f(x-h)] / 12h
	VectorXd av, bv, cv, dv;

	av = derF(mod, x, param, nLayers, ai, -2.0 * h, 1.0 / (12.0 * h));
	bv = derF(mod, x, param, nLayers, ai, h, 8.0 / (12.0 * h));
	cv = derF(mod, x, param, nLayers, ai, -h, -8.0 / (12.0 * h));
	dv = derF(mod, x, param, nLayers, ai, 2.0 * h, -1.0 / (12.0 * h));
	

	return (av + bv + cv + dv);
}

VectorXd PolydisperseModel::Derivative(const std::vector<double>& x, VectorXd param,
									   int nLayers, int ai) {
	return NumericalDerivative(this, x, param, nLayers, ai, 1.0e-9);
}

void Geometry::OrganizeParameters(const VectorXd& p, int nLayers) {
	//std::string str = debugMatrixPrintM(p);
	//MessageBoxA(NULL, str.c_str(), "Parameters Vector", NULL);

	if(parameters)
		delete parameters;
	if(extraParams)
		delete extraParams;

	parameters = new MatrixXd(MatrixXd::Zero(nLayers, nLayerParams));
	extraParams = new VectorXd(VectorXd::Zero(nExtraParams));
	int c = 0;
	for (int j = 0; j < parameters->cols(); j++)
		for (int i = 0; i < parameters->rows(); i++)
			(*parameters)(i,j) = p[c++];
	for (int i = 0; i < extraParams->size(); i++)
		(*extraParams)[i] = p[c++];
}

void Geometry::SetEDProfile(EDProfile edp) {
	if(profileFunc) {
		nLayerParams -= profileFunc->GetNumEDParams();
		delete profileFunc;
	}

	profile.type = edp.type;
	profile.shape = edp.shape;
	profileFunc = ProfileFromShape(profile.shape, MatrixXd::Zero(1, 1));
	if(!profileFunc)
		return;

	nLayerParams += profileFunc->GetNumEDParams();
}

std::vector<double> Geometry::GetAllParameters() {
	std::vector<double> res;
	if (parameters) // Hacky fix. Should cause other problems. Think of a better solution.
		for (int j = 0; j < parameters->cols(); j++)
			for (int i = 0; i < parameters->rows(); i++)
				res.push_back((*parameters)(i, j));
	if (extraParams)
		for (int i = 0; i < extraParams->size(); i++)
			res.push_back((*extraParams)[i]);
	return res;
}

VectorXd Geometry::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	return NumericalDerivative(this, x, param, nLayers, ai, 1.0e-9);
}

bool Geometry::GetHasAnomalousScattering()
{
	return false;
}

///// Miscellaneous Methods

// Orientation Average
double OrientationAverage(double q, FFModel *model, int nLayers, VectorXd& p) {
	/*_w = a[1]; _d = a[2]; _h = a[3];
	_q = q; _ed = a[nd];
	//Pablo
	int innerres = 2;
	int osc  = int((max(max(_w,_d),_h))*q*innerres);*/

	static VectorXd phix, thetax, phiw, thetaw;
	double result = 0.0;

	SetupIntegral(phix, phiw, 0.0 + EPS, 2.0 * PI + EPS, defaultQuadRes);
	SetupIntegral(thetax, thetaw, 0.0 + EPS, PI + EPS, defaultQuadRes);

	if(defaultQuadRes <= 1)
		return result;
	
	#pragma omp parallel for default(shared) schedule(static) reduction(+ : result)
	for(int i = 0; i < defaultQuadRes; i++) {
		double inner = 0.0;
		
		for(int j = 0; j < defaultQuadRes; j++) {
			double precision = (j > 0) ? (thetax[j] - thetax[j -1]) / 2.0 : (thetax[j]) / 2.0   ;
			Vector3d qvector (q * sin(thetax[j]) * cos(phix[i]), 
					  q * sin(thetax[j]) * sin(phix[i]),
					  q * cos(thetax[j]));
			inner += std::norm(model->CalculateFF(qvector, 
							   nLayers,thetaw[j],precision)) * sin(thetax[j]) * thetaw[j];
		}
		result += inner * phiw[i];
	}

	return result;
}

// Polydisperse model
double PolydisperseModel::Calculate(double q, int nLayers, VectorXd& a) {
	// This should never be called
	return -5.3;
}


IModel *PolydisperseModel::GetInnerModel() {
	return model;
}

VectorXd PolydisperseModel::GPUCalculate( const std::vector<double>& q,int nLayers, VectorXd& p /*= VectorXd()*/ ) {
	// TODO::GPU: Later
	return VectorXd();
}

bool PolydisperseModel::GetHasAnomalousScattering()
{
	return model->GetHasAnomalousScattering();
}

VectorXd PolydisperseModel::CalculateVector(const std::vector<double> &q, int nLayers, Eigen::VectorXd &a,
											progressFunc progress, void *progressArgs) {
	int points = pdResolution;
	VectorXd b = a, a1 = a, x = VectorXd::Zero(points), intensity = VectorXd::Zero(q.size());

	if(a1.size() == 0) {	// Outermost PD layer
		b = a1 = p;
	}

	int param = polyInd;
	double sig = polySigma;
	double Z = 0.0;
	if(param < 0 || sig < 1.0e-7)
		return model->CalculateVector(q, nLayers, a, progress);

	for(int i = 0; i < points; i++) {
		if(pStop && *pStop)
			return VectorXd::Zero(q.size());

		x[i] = a1[param] - 2.0 * sig + double(i) / double(points - 1) * 4.0 * sig; // taking 2 sigma on each side
		if(x[i] < 0.0)	// don't use...
			continue;
		double ga = 0.0;

		// Later on, this will use a generic PDProfile class, so that each PD
		// can have its own arbitrary pattern
		switch(pdFunction) {
			default:
				break;
			case SHAPE_GAUSSIAN:
				ga = gaussianSig(sig, a1[param], 1.0, 0.0, x[i]);
				break;
			case SHAPE_LORENTZIAN:
				ga = lorentzian(sig, a1[param], 1.0, 0.0, x[i]);
				break;
			case SHAPE_LORENTZIAN_SQUARED: // Actually this is Uniform
				ga = 1.0 / (double)points;
				break;
		}

		Z += ga;
		b[param] = x[i];
		intensity += ga * model->CalculateVector(q, nLayers, b);
	}
	return intensity / Z;
}


MatrixXd PolydisperseModel::CalculateMatrix(const std::vector<double>& q, int nLayers, Eigen::VectorXd& a,
	progressFunc progress, void* progressArgs) 
{

	int points = pdResolution;
	VectorXd b = a, a1 = a, x = VectorXd::Zero(points), intensity = VectorXd::Zero(q.size());

	if (a1.size() == 0) {	// Outermost PD layer
		b = a1 = p;
	}

	int param = polyInd;
	double sig = polySigma;
	double Z = 0.0;
	if (param < 0 || sig < 1.0e-7)
		return model->CalculateVector(q, nLayers, a, progress);

	for (int i = 0; i < points; i++) {
		if (pStop && *pStop)
			return VectorXd::Zero(q.size());

		x[i] = a1[param] - 2.0 * sig + double(i) / double(points - 1) * 4.0 * sig; // taking 2 sigma on each side
		if (x[i] < 0.0)	// don't use...
			continue;
		double ga = 0.0;

		// Later on, this will use a generic PDProfile class, so that each PD
		// can have its own arbitrary pattern
		switch (pdFunction) {
		default:
			break;
		case SHAPE_GAUSSIAN:
			ga = gaussianSig(sig, a1[param], 1.0, 0.0, x[i]);
			break;
		case SHAPE_LORENTZIAN:
			ga = lorentzian(sig, a1[param], 1.0, 0.0, x[i]);
			break;
		case SHAPE_LORENTZIAN_SQUARED: // Actually this is Uniform
			ga = 1.0 / (double)points;
			break;
		}

		Z += ga;
		b[param] = x[i];
		intensity += ga * model->CalculateVector(q, nLayers, b);
	}
	return intensity / Z;
}


VectorXd FFModel::Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai) {
	double h = 1.0e-9;

	// Special cases
	// Partial Scale Derivative
	if(ai == (param.size() - nExtraParams)) { 

		// Create copies for the parameter vector and the number of layers for
		// this iteration
		VectorXd guess = param;
		int guessLayers = nLayers;

		// Tal: I don't like the use of OrganizeParameters. Why not use PreCalculate instead?
		OrganizeParameters(guess, guessLayers);

		VectorXd der = CalculateVector(x, guessLayers, guess);
		der -= VectorXd::Constant(x.size(), (*extraParams)[1]);

		der /= (*extraParams)[0];
		return der;
	}
	//Partial Background Derivative
	else if(ai == (param.size() - nExtraParams + 1)) {
		return VectorXd::Ones(x.size());
	}

	return Geometry::Derivative(x, param, nLayers, ai);
}

std::string FFModel::GetLayerNameStatic(int layer)
{
	if (layer < 0)
		return "N/A";

	if (layer == 0)
		return "Solvent";

	return "Layer %d";

}

ExtraParam FFModel::GetExtraParameterStatic(int index)
{
	switch (index)
	{
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0);
	case 1:
		return ExtraParam("Background", 0.0);
	}
}

LuaModel::LuaModel(std::string script, void *luaContext) : 
Geometry("Scripted Model"), modelCode(script), luactx(luaContext) {

	// No globals/context
	if(!luaContext) {		
		// TODO: Create Lua context specifically for the script

		bContextCreated = true;
	} else
		bContextCreated = false;

	// TODO: Check validity of code (catch compiler/interpreter exceptions)

	// TODO: Check code compatibility (must include a calculate method)

	// TODO: Get model information
	// _NAME
	// _EXTRAPARAMS
	// _NLP
	// _DISPLAYPARAMS
	// _MINLAYERS
	// _MAXLAYERS		
	// _EDPROFILETYPE (opt)
	/*std::string name, int extras, int nlp, 
		int minlayers, int maxlayers, EDProfile edp, int disp*/
}

LuaModel::~LuaModel() {
	// TODO: Call inner Dispose function, if exists

	if(bContextCreated) {
		// TODO: Free Lua context
	}
}

void LuaModel::PreCalculate( VectorXd &p, int nLayers ) {
	parVec = p;

	// TODO: Call inner pre-calculate, if exists
}

double LuaModel::Calculate( double q, int nLayers, VectorXd& p ) {
	// TODO: Call inner calculate

	return -1.0;
}


