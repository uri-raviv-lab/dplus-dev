//#define NOMINMAX
//#include "Windows.h"	//For messagebox debugging
#include <iostream>

#include "Geometry.h"
#include "Quadrature.h" // For Quadrature

#include "Eigen/LU" // For matrix inverse
#include "mathfuncs.h" // For gaussianSig



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


