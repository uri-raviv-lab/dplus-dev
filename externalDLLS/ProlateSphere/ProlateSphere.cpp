// ProlateSphere.cpp : Defines the exported functions for the DLL application.
//
#include "ProlateSphere.h"
#include "Quadrature.h"

// OpenGL includes
#ifdef _WIN32
#include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>
// END of OpenGL includes

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// Container information

int GetNumCategories() {
	return 1;
}

ModelCategory GetCategoryInformation(int catInd) {
	switch(catInd) {
		default:
			{
				ModelCategory mc = { "N/A", MT_FORMFACTOR, {-1} };
				return mc;
			}

		case 0:
			{
				ModelCategory mc = { "Prolate Spheroids", MT_FORMFACTOR, {0, -1} };
				return mc;
			}
	}
}

// Returns the number of models in this container
int GetNumModels() {
	return 1;	
}

// Returns the model's display name from the index. Supposed to return "N/A"
// for indices that are out of bounds
ModelInformation GetModelInformation(int index) {
	switch(index) {
		default:
			return ModelInformation("N/A");
		case 0:
			return ModelInformation("Prolate Sphere", 0, index, true, 1, 2, 2, 4);
	}
}

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
Geometry *GetModel(int index) {
	switch(index) {
		default:
			return NULL;
		case 0:
			return new ProlateSphereModel();
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// ProlateSphereModel functions
ProlateSphereModel::ProlateSphereModel(std::string st, ProfileType edp, int intSteps) : FFModel(st, 4, 1, 2, 2, EDProfile(NONE)) {
	steps = intSteps;
	SetupIntegral(xx, ww, 0.0f, 1.0f, intSteps);
}

double ProlateSphereModel::GetDefaultParamValue(int paramIndex, int layer) {
	if(paramIndex != 0)
		return -1.0;

	switch(layer) {
		case 0:
			return 333.0;
		case 1:
			return 400.0;
		default:
			return -1.0;
	}
}

std::string ProlateSphereModel::GetLayerParamName(int index) {
	switch(index) {
		case 0:
			return "E.D.";
		default:
			return "N/A";
	}
}


void ProlateSphereModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void ProlateSphereModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);
	a = (*extraParams)(2);
	b = (*extraParams)(3);
	ed = (*parameters).col(0);
}

double bessel_j1(double x);
double sphericalBessel_j1(double x);
double ProlateSphereModel::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0;
//#pragma omp parallel for reduction(+ : intensity)
	for(int i = 0; i < steps; i++) {
		double u = q * sqrt((a * a * xx(i) * xx(i) + b * b * (1.0 - xx(i) * xx(i))));
		double amp = ((ed(1) - ed(0)) * 3.0 * sphericalBessel_j1(u) / u);
		//double amp = ((ed(1) - ed(0)) * 3.0 * bessel_j1(u) / u);
		//double amp = ((ed(1) - ed(0)) * 3.0 * (cos(u) * u - sin(u)) / (u * u * u));
		intensity += amp * amp * ww(i);
	}

	//intensity *= intensity;

	intensity *= (*extraParams)(0);	// Scale
	intensity += (*extraParams)(1);	// Background

	return intensity;
}

std::complex<double> ProlateSphereModel::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam ProlateSphereModel::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 3.0);
	case 2:
		return ExtraParam("a", 2.0, false, true);
	case 3:
		return ExtraParam("b", 1.0, false, true);
	}
}

double bessel_j1(double x)
{
	double ax,z;
	double xx,y,ans,ans1,ans2;

	if ((ax=fabs(x)) < 8.0)
	{ y=x*x;
	ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
		+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
	ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
		+y*(99447.43394+y*(376.9991397+y*1.0))));
	ans=ans1/ans2;
	}

	else {
		z=8.0/ax;
		y=z*z;
		xx=ax-2.356194491;
		ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
			+y*(0.2457520174e-5+y*(-0.240337019e-6))));
		ans2=0.04687499995+y*(-0.2002690873e-3
			+y*(0.8449199096e-5+y*(-0.88228987e-6
			+y*0.105787412e-6)));
		ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
		if (x < 0.0)
			ans = -ans;
	}
	return ans;
}

double sphericalBessel_j1(double x) {
	return sin(x)/(x*x) - cos(x)/x;
}