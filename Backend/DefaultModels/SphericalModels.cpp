#include "SphericalModels.h"

#include <algorithm>

#include "Quadrature.h" // For SetupIntegral

#include "mathfuncs.h" // For ln2 and square

#include "GPUHeader.h"

//#include "Windows.h"	//For MessageBox


#ifdef _WIN32
#define NOMINMAX  // Don't shadow std::min and std::max
#include <windows.h> // For LoadLibrary
#pragma comment(lib, "user32.lib") // TEMP TODO REMOVE
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym 
#endif

#include <vector_functions.h>
#include "BackendInterface.h"

typedef bool (*GPUDirectSphereAmplitude_t)(Workspace& work, float2 *params, int numLayers);
GPUDirectSphereAmplitude_t gpuSphereAmplitude = NULL;

typedef bool (*GPUHybridSetUSphereAmplitude_t)(GridWorkspace& work, float2 *params, int numLayers, float* extras, int nExtras);
GPUHybridSetUSphereAmplitude_t gpuHybridSetUSphereAmplitude = NULL;

typedef bool (*GPUHybridUSphereAmplitude_t)(GridWorkspace& work);
GPUHybridUSphereAmplitude_t gpuHybridUSphereAmplitude;

#pragma region Abstract Sphere

	SphericalModel::SphericalModel(std::string st, int nlp, ProfileShape edp, int exParams) : 
		FFModel(st, exParams, nlp, 2, -1, EDProfile(SYMMETRIC, edp)) {}
	

	bool SphericalModel::IsParamApplicable(int layer, int lpindex) {
		return Geometry::IsParamApplicable(layer, lpindex);
	}

	std::string SphericalModel::GetLayerParamName(int index, EDPFunction *edpfunc)
	{
		return GetLayerParamNameStatic(index, edpfunc);
	}
	std::string SphericalModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {
		switch(index) {
			default:
				return Geometry::GetLayerParamName(index, edpfunc);
			case 0:
				return "Radius";
			case 1:
				return "E.D.";
		}
	}

	void SphericalModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
		Geometry::OrganizeParameters(p, nLayers);
	
		r			= (*parameters).col(0);
		ED			= (*parameters).col(1);
		edSolvent	= (*parameters)(0,1);
	}

	ExtraParam SphericalModel::GetExtraParameterStatic(int index)
	{
		return FFModel::GetExtraParameterStatic(index);
	}

	ExtraParam SphericalModel::GetExtraParameter(int index)
	{
		return GetExtraParameterStatic(index);
	}

	std::string SphericalModel::GetLayerName(int layer)
	{
		return FFModel::GetLayerNameStatic(layer);
	}

#pragma endregion

#pragma region Uniform Sphere

	UniformSphereModel::UniformSphereModel(std::string st) : SphericalModel(st) {
	}

	void UniformSphereModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
		SphericalModel::OrganizeParameters(p, nLayers);

		for(int i = 1; i < nLayers; i++)
			r[i] += r[i - 1];
	}

	void UniformSphereModel::PreCalculate(VectorXd& p, int nLayers) {
		OrganizeParameters(p, nLayers);
	}

	double UniformSphereModel::Calculate(double q, int nLayers, Eigen::VectorXd &p) {
		double intensity = 0.0;

		if(p.size() > 0)
			OrganizeParameters(p, nLayers);

		for(int i = 0; i < nLayers - 1; i++)
			intensity += (ED[i] - ED[i + 1]) * 
			( (sin(q * r[i])) - (q * r[i] * cos(q * r[i])));

		intensity += (ED[nLayers - 1] - edSolvent) * 
			( (sin(q * r[nLayers - 1]) )- (q * r[nLayers - 1] * cos(q * r[nLayers - 1])));

		intensity = sq(4.0 * PI * intensity / (q * sq(q)));

		intensity *= (*extraParams)(0);	// Multiply by scale
		intensity += (*extraParams)(1);	// Add background

		return intensity;
	}

	std::complex<double> UniformSphereModel::CalculateFF(Vector3d qvec, int nLayers,
											double w, double precision, VectorXd* p) {
		double q = sqrt(sq(qvec(0)) + sq(qvec(1)) + sq(qvec(2)));

		if(closeToZero(q)) {
			double electrons = 0.0;
			for(int i = 1; i < nLayers; i++) {
				electrons += (ED(i) - ED(0)) * (4.0 / 3.0) * PI * 
					(r(i) * r(i) * r(i) - r(i-1) * r(i-1) * r(i-1));
			}
			return (std::complex<double>(electrons, 0.0) * (*extraParams)(0)) + (*extraParams)(1);
		}											   

		double res = 0.0;
#pragma omp parallel for reduction( - : res)
		for(int i = 0; i < nLayers - 1; i++)
			res -= (ED(i) - ED(i + 1)) * (cos(q * r(i)) * q * r(i) - sin(q * r(i)));
		res -= (ED(nLayers - 1) - ED(0)) * (cos(q * r(nLayers - 1)) * q * r(nLayers - 1) - sin(q * r(nLayers - 1)));

		res *= 4.0 * PI / (sq(q) * q);
		
		res *= (*extraParams)(0);	// Multiply by scale
		res += (*extraParams)(1);	// Add background

		return std::complex<double>(res, 0.0);
	}	

	void UniformSphereModel::PreCalculateFF(Eigen::VectorXd &p, int nLayers) {
		OrganizeParameters(p, nLayers);
	}

	bool UniformSphereModel::SetModel( Workspace& workspace )
	{
		return true;
	}

	void UniformSphereModel::CorrectLocationRotation(double& x, double& y, double& z, 
													 double& alpha, double& beta, double& gamma)
	{
		// Cancel orientation values
		alpha = Radian(0.0); 
		beta  = Radian(0.0); 
		gamma = Radian(0.0);
	}

	bool UniformSphereModel::SetParameters(Workspace& workspace, const double *params, unsigned int numParams)
	{
		/*
		if(!g_gpuModule) {
			load_gpu_backend(g_gpuModule);

			if(!g_gpuModule)
				return false;
		}

		if(!gpuSphereAmplitude)
			gpuSphereAmplitude = (GPUDirectSphereAmplitude_t)GPUDirect_SetSphereParamsDLL;
		if(!gpuSphereAmplitude)
			return false;/**/

		VectorXd vParams (numParams);
		for(int i = 0; i < numParams; i++) vParams[i] = params[i];
		unsigned int numLayers = (numParams - 2) / 2;

		// Convert to float2 representation
		PreCalculateFF(vParams, numLayers);
		float2 *fparams = new float2[numLayers];
		for(int i = 0; i < numLayers; i++)
			fparams[i] = make_float2(r[i], ED[i]);

		return GPUDirect_SetSphereParamsDLL(workspace, fparams, numLayers);
	}

	bool UniformSphereModel::ComputeOrientation( Workspace& workspace, float3 rotation )
	{
		// A sphere has no meaning for orientation
		return true;
	}

	bool UniformSphereModel::CalculateGridGPU( GridWorkspace& workspace )
	{
		/*
		if(!g_gpuModule) {
			load_gpu_backend(g_gpuModule);

			if(!g_gpuModule)
				return false;
		}

		if(!gpuHybridUSphereAmplitude)
			gpuHybridUSphereAmplitude = (GPUHybridUSphereAmplitude_t)GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_USphereAmplitudeDLL");
		if(!gpuHybridUSphereAmplitude)
			return false;/**/

		return GPUHybrid_USphereAmplitudeDLL(workspace);
	}

	bool UniformSphereModel::SetModel( GridWorkspace& workspace )
	{
		/*
		if(!g_gpuModule) {

			load_gpu_backend(g_gpuModule);
			if(!g_gpuModule)
				return false;
		}

		if(!gpuHybridSetUSphereAmplitude)
			gpuHybridSetUSphereAmplitude = (GPUHybridSetUSphereAmplitude_t)GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_SetUSphereDLL");
		if(!gpuHybridSetUSphereAmplitude)
			return false;
		/**/
		std::vector<float2> vParams(r.size());
		for(int i = 0; i < r.size(); i++)
			vParams[i] = make_float2(r[i], ED[i] - edSolvent);
		float2 extras = make_float2((*extraParams)(0), (*extraParams)(1));
		workspace.scale = 1.;	// Already appears in the extras
	
		return GPUHybrid_SetUSphereDLL(workspace, vParams.data(), vParams.size(), (float*)&extras, 2);
	}

	bool UniformSphereModel::ImplementedHybridGPU() {
		return true;
	}

	ExtraParam UniformSphereModel::GetExtraParameterStatic(int index)
	{
		return SphericalModel::GetExtraParameterStatic(index);
	}

	ExtraParam UniformSphereModel::GetExtraParameter(int index)
	{
		return GetExtraParameterStatic(index);
	}

#pragma endregion

#pragma region Gaussian Sphere

	GaussianSphereModel::GaussianSphereModel(std::string st, ProfileShape edp) : SphericalModel(st, 3, edp) {
		steps = 500;
		SetupIntegral(xx, ww, 0.0f, 1.0f, steps);
	}
	
	bool GaussianSphereModel::IsParamApplicable(int layer, int lpindex) {
		if(layer == 0 && (lpindex == 2 || lpindex == 0))
			return false;
		return Geometry::IsParamApplicable(layer, lpindex);
	}

	std::string GaussianSphereModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
		return GetLayerParamNameStatic(index, edpfunc);
	}
	std::string GaussianSphereModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {

		if(index == 2)
			return "R 0";
		return SphericalModel::GetLayerParamNameStatic(index, edpfunc);
	}
	double GaussianSphereModel::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
		if(paramIndex == 2) // Z0
			return (double)(layer - 1);
		return Geometry::GetDefaultParamValue(paramIndex, layer, edpfunc);
	}

	void GaussianSphereModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
		SphericalModel::OrganizeParameters(p, nLayers);
		z0 = (*parameters).col(2);
	}

	void GaussianSphereModel::PreCalculate(VectorXd& p, int nLayers) {
		OrganizeParameters(p, nLayers);
	}

	double GaussianSphereModel::Calculate(double q, int nLayers, Eigen::VectorXd &p) {
		double intensity = 0.0;

		if(p.size() > 0)
			OrganizeParameters(p, nLayers);

		VectorXd edtexp = VectorXd::Zero(nLayers);

		for(int i = 1; i < nLayers; i++) {
			edtexp[i] = (ED[i] - edSolvent) * r[i] * exp(-sq(r[i] * q) / (16.0 * ln2));
			intensity += edtexp[i]
				* ( ((-2.0 * z0[i] * sin(q * z0[i])) / sqrt(ln2)) - ((sq(r[i]) * q * cos(q * z0[i])) / (4.0 * pow(ln2, 1.5))) );
		}

		// += integral

		#pragma omp parallel for reduction(+ : intensity)
		for(int i = 0; i < steps; i++) {
			double inner = 0.0;
			for(int j = 1; j < nLayers; j++) {
				double rqy2 = z0[j] * q * (sq(xx[i])- 1);

				inner += edtexp[j] * exp(sq(xx[i]) * ((sq(r[j] * q) / (16.0 * ln2)) - ((4.0 * ln2 * sq(z0[j] / r[j]) ))))
					* (( (sq(2.0 * z0[j]) / r[j]) - (sq(r[j] * q / (4.0 * ln2) ) * r[j])) * sin(rqy2) - (( z0[j] * r[j] * q / ln2 ) * cos(rqy2)));
			}

			intensity += inner * ww[i];
		}

		intensity *= pow(PI, 1.5) / (2.0 * q);

		intensity *= intensity;	// Square, no need for an orientation average

		intensity *= (*extraParams)(0);	// Multiply by scale
		intensity += (*extraParams)(1);	// Add background

		return intensity;

	}

	std::complex<double> GaussianSphereModel::CalculateFF(Vector3d qvec, 
									 int nLayers, double w, double precision, VectorXd* p) {
		return std::complex<double>(0.0, -1.0); //TODO::ComplexModels
	}

	std::string GaussianSphereModel::GetLayerName(int layer)
	{
		return SphericalModel::GetLayerName(layer);
	}

#pragma endregion

