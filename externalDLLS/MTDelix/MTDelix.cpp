// MTDelix.cpp : Defines the exported functions for the DLL application.
//
#include "MTDelix.h"
#include "Quadrature.h"

// OpenGL includes
#ifdef _WIN32
#include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>
// END of OpenGL includes

template <typename T> inline T sq(T x) { return x * x; }

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
				ModelCategory mc = { "MT Discrete Helices", MT_FORMFACTOR, {0, -1} };
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
			return ModelInformation("Infinite Uniform Delix", 0, index, true, 0, 0, 0, 8);
	}
}

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
Geometry *GetModel(int index) {
	switch(index) {
		default:
			return NULL;
		case 0:
			return new MTDelixModel();
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// MTDelixModel functions
MTDelixModel::MTDelixModel(std::string st, ProfileType edp, int tSteps, int pSteps) : FFModel(st, 8, 0, 0, 0, EDProfile(NONE)) {
	thetaSteps = tSteps;
	phiSteps = pSteps;
#pragma omp parallel sections
	{
#pragma omp section
		{
			SetupIntegral(xTheta, wTheta, 0.0, PI, thetaSteps);
		}
#pragma omp section
		{
			SetupIntegral(xPhi, wPhi, 0.0, 2.0 * PI, phiSteps);
		}
	}
}

double MTDelixModel::GetDefaultParamValue(int paramIndex, int layer) {
	return -1.0;
}

std::string MTDelixModel::GetLayerParamName(int index) {
	return "N/A";
}


void MTDelixModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void MTDelixModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);
	P			= (*extraParams)(2);
	nHelices	= int((*extraParams)(3));
	helixR		= (*extraParams)(4);
	sphereR		= (*extraParams)(5);
	dw			= (*extraParams)(6);
	ed			= (*extraParams)(7);
}

double MTDelixModel::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0;
	double missedWT = 0.0, missedWP = 0.0;

#pragma omp parallel for reduction(+ : intensity)
	for(int i = 0; i < thetaSteps; i++) {
		double tmp = 0.0;
		if(pStop && *pStop)
			continue;
		for(int j = 0; j < phiSteps; j++) {
			try {
				tmp = sq(ed * sin(P * q * cos(xTheta[i]) / 2.0) / 
							(sin(P * q * cos(xTheta[i]) / (2.0 * (double)nHelices))* 
							sin((2.0 * sphereR + dw) * (P * q * cos(xTheta[i]) + 2.0 * PI * xPhi[j])) / (4.0 * PI * helixR))
						)
					* sin(xTheta[i]) * wTheta[i] * wPhi[j];
			}
			catch(...) {
				missedWP += wPhi[j];
				missedWT += wTheta[j];
				continue;
			}
			if(tmp != tmp) {	// Exception not generated
				missedWP += wPhi[j];
				missedWT += wTheta[j];
				continue;
			}
			intensity += tmp;
		}
	}

	intensity /= (1.0 - missedWP) * (1.0 - missedWT) * (32.0 * PI * PI * PI);;

	intensity *= (*extraParams)(0);	// Scale
	intensity += (*extraParams)(1);	// Background

	return intensity;
}

std::complex<double> MTDelixModel::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam MTDelixModel::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 5.0);
	case 2:
		return ExtraParam("Pitch", 11.0, false, true);
	case 3:
		return ExtraParam("# of Helices", 3, false, true, false, 0.0, 0.0, true);
	case 4:
		return ExtraParam("Helix Radius", 10.0, false, true);
	case 5:
		return ExtraParam("Sphere Radius", 1.0, false, true);
	case 6:
		return ExtraParam("Water Spacing", 1.0, false, true);
	case 7:
		return ExtraParam("E.D. Contrast", 67.0, false, true);
	}
}

bool MTDelixModel::IsLayerBased() {
	return false;
}