// Icosahedron.cpp : Defines the exported functions for the DLL application.
//
#include "Icosahedron.h"
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
				ModelCategory mc = { "Polyhedra", MT_FORMFACTOR, {0, -1} };
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
			return ModelInformation("Icosahedron with Inner Sphere", 0, index, true, 2, 1, -1, 5);
	}
}

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
Geometry *GetModel(int index) {
	switch(index) {
		default:
			return NULL;
		case 0:
			return new Icosahedron();
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


// Icosahedron functions
Icosahedron::Icosahedron(std::string st, ProfileType edp) : FFModel(st, 5, 2, 1, -1, EDProfile(NONE)) {
	displayParams = 1;
	tau = (1.0 + sqrt(5.0)) / 2.0;
	rVec = MatrixXd::Zero(12, 3);

	rVec(0,1) = 1.0;	rVec(0,2) = tau;
	rVec(1,1) = -1.0;	rVec(1,2) = -tau;
	rVec(2,1) = -1.0;	rVec(2,2) = tau;
	rVec(3,1) = 1.0;	rVec(3,2) = -tau;
	rVec(4,0) = 1.0;	rVec(4,1) = tau;
	rVec(5,0) = -1.0;	rVec(5,1) = -tau;
	rVec(6,0) = -1.0;	rVec(6,1) = tau;
	rVec(7,0) = 1.0;	rVec(7,1) = -tau;
	rVec(8,0) = tau;	rVec(8,2) = 1.0;
	rVec(9,0) = -tau;	rVec(9,2) = -1.0;
	rVec(10,0) = -tau;	rVec(10,2) = 1.0;
	rVec(11,0) = tau;	rVec(11,2) = -1.0;

	rVec = (rVec.array() / 2.0).matrix();

	norms = Eigen::ArrayXXd::Zero(12, 12);

	for(int i = 1; i < 12; i++)
		for(int j = i + 1; j < 12; j++) {
			VectorXd ni = rVec.row(i);
			VectorXd nj = rVec.row(j);
			VectorXd sub = ni - nj;
			double Roi = sub.dot(sub);
			Roi = sqrt(Roi);

			norms(i,j) = Roi;
		}
	
}

double Icosahedron::GetDefaultParamValue(int paramIndex, int layer) {
	switch(paramIndex) {
	default:
		return -1.0;
	case 0:
		if(layer == 0)
			return 0.0;
		return 1.0;
	case 1:
		if(layer == 0)
			return 333.0;
		return 400.0;		
	}
}

std::string Icosahedron::GetLayerParamName(int index) {
	switch(index) {
		default:
			return "N/A";
		case 0:
			return "Width";
		case 1:
			return "E.D.";
	}
}


void Icosahedron::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void Icosahedron::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);
	width		= (*parameters).col(0);
	EDs			= (*parameters).col(1);

	EDcapsid	= (*extraParams)(2);
	capsid_subR	= (*extraParams)(3);
	Rico		= (*extraParams)(4);
	
	r = width;

	for (int i = 1; i < nLayers; i++)
		r[i] += r[i - 1];
}

double Icosahedron::Calculate(double q, int nLayers, VectorXd& p) {
	if(p.size() > 0)
		OrganizeParameters(p, nLayers);

	double intensity = 0.0;
	double Icore = 0.0 , Icapsid = 0.0 , Imix = 0.0;
	double missedWT = 0.0, missedWP = 0.0;

//#pragma omp parallel for reduction(+ : intensity)

// Intensity from the core
	for (int i = 0; i < nLayers - 1 ; i++)
		Icore += (EDs[i] - EDs[i + 1]) * (sin(q*r[i])/q - r[i]*cos(q*r[i]));
	
	Icore += (EDs[nLayers - 1] - EDs[0]) * (sin(q*r[nLayers - 1])/q
											-r[nLayers - 1]*cos(q*r[nLayers - 1]));


// Intensity from the capsid
	Icapsid = 12.0;
	for(int i = 1; i < 12; i++)
		for(int j = i + 1; j < 12; j++)
			Icapsid += 2.0 * sin(q * Rico * norms(i,j)) / (q * Rico * norms(i,j));
	Icapsid *= sq(sin(q * capsid_subR) / q - capsid_subR * cos(q * capsid_subR));
	Icapsid *= sq((4.0 * PI / sq(q)) * EDcapsid);

// Intensity from the interaction
	Imix = Icore * ((4.0 * PI / sq(q)) * EDcapsid);
	Imix *= sin(q * capsid_subR) / q - capsid_subR * cos(q * capsid_subR);
	Imix *= 12.0 / (q * Rico) * sin(q * Rico) * (4.0 * PI) / sq(q);

	Icore *= Icore * sq(4.0 * PI / sq(q));

	intensity = Icore + Icapsid + Imix;

	intensity *= (*extraParams)(0);	// Scale
	intensity += (*extraParams)(1);	// Background

	return intensity;
}

std::complex<double> Icosahedron::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam Icosahedron::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 5.0);
	case 2:
		return ExtraParam("VP1 electron density", 370.0);
	case 3:
		return ExtraParam("VP1 radius", 4);
	case 4:
		return ExtraParam("Icosahedron radius", 8.0);
	}
}

bool Icosahedron::IsLayerBased() {
	return false;
}