//#define NOMINMAX
//#include "Windows.h"	//For min/max and messagebox debugging


#include "CylindricalModels.h"
#include "Quadrature.h" // For SetupIntegral
#include "kronrod.hpp"
#include "mathfuncs.h" // For bessel functions and square

using Eigen::VectorXf;

#pragma region Abstract Cylindrical Geometry

CylindricalModel::CylindricalModel(int integralSteps, std::string str, ProfileShape eds, int nlp, int minlayers, int maxlayers, int extras)
							: FFModel(str, extras, nlp, minlayers, maxlayers, EDProfile(SYMMETRIC, eds)) {
	steps = integralSteps;
	SetupIntegral(xx, ww, 0.0f, 1.0f, steps);
}

ExtraParam CylindricalModel::GetExtraParameter(int index) {
	return GetExtraParameterStatic(index);
}
ExtraParam CylindricalModel::GetExtraParameterStatic(int index) {
	if (index == 2)
		return ExtraParam("Height", 10, 
						  true, true);
	
	// Default extra parameters
	return Geometry::GetExtraParameter(index);	
}

void CylindricalModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	Geometry::OrganizeParameters(p, nLayers);
	t	= (*parameters).col(0);
	ed	= (*parameters).col(1);
	H = (*extraParams)[2] / 2.0;
	edSolvent = ed[0]; 
}

std::string CylindricalModel::GetLayerNameStatic(int layer)
{
	return FFModel::GetLayerNameStatic(layer);
}

#pragma endregion

#pragma region Uniform Cylindrical Model

UniformHCModel::UniformHCModel(int integralSteps, std::string str, ProfileShape eds, int nlp, int minlayers, int maxlayers, int extras) :
						CylindricalModel(integralSteps, str, eds, nlp, minlayers, maxlayers, extras) {
}

std::string UniformHCModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
	return GetLayerParamNameStatic(index, edpfunc);
}
std::string UniformHCModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {
	return CylindricalModel::GetLayerParamName(index, edpfunc);
}

void UniformHCModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void UniformHCModel::OrganizeParameters(const VectorXd& p, int nLayers) {
	CylindricalModel::OrganizeParameters(p, nLayers);

	for(int i = 1; i < t.size(); i++)
		t[i] = t[i - 1] + t[i];
}

double UniformHCModel::Calculate(double q, int nLayers, VectorXd& a) {
	double H = -1.0; 
    double scale, background;
    double intensity = 0.0;

	if(a.size() > 0)
		OrganizeParameters(a, nLayers);
	
	scale = (*extraParams)[0];
    background = (*extraParams)[1];
	H = std::isinf((*extraParams)[2]) ? -1.0 :
							(*extraParams)[2]/2.0;
    
	/*Pablo & Avi's model -----  */
	int notZero = 0;
	for(notZero = 0; (notZero < nLayers) && (t[notZero] <= 0.0); notZero++);
	if(notZero == nLayers)
		return 0.0;	//No layer thickness

	for(int i = nLayers - 1; i >= 0; i--)
		ed[i] -= ed[0];
	/*Finite*/
	if(H >= 0.0) {
		
#pragma omp parallel for reduction(+ : intensity)
		for(int i = 0; i < steps; i++) {
			if (xx[i] > 0.0 && xx[i] < 1.0) {			
				double temp = 0.0;

				if(pStop && *pStop)		// Place these line strategically in
					continue;			// slow models.

				for(int j = notZero; j < nLayers - 1; j++) {
					temp += (ed[j] - ed[j + 1])* t[j] * bessel_j1(q * sqrt(1.0 - xx[i] * xx[i]) * t[j]); 								
				}
				temp += ed[nLayers - 1] * t[nLayers- 1] * bessel_j1(q * sqrt(1.0 - xx[i] * xx[i]) * t[nLayers - 1]);
				temp *= 4.0 * PI * sin(q * xx[i] * H) / (q * q * xx[i] * sqrt(1.0 - xx[i] * xx[i]));
				intensity += temp * temp * ww[i];
			}
		}
	}
	
	/*Pablo & Avi's model -----  Infinite*/
	else {
		// "pre"-calculate the bessel coefficients [R * J1(qR)]
		VectorXd besselCoeff = VectorXd::Zero(nLayers);
		for(int i = 0; i < nLayers; i++)
			besselCoeff[i] = t[i] * bessel_j1(q * t[i]);

		double temp = 0.0;
		for(int j = notZero; j < nLayers - 1; j++) {
			temp += (ed[j] - ed[j + 1]) * besselCoeff[j]; 								
		}
		temp *= 2.0 * ed[nLayers - 1] * besselCoeff[nLayers - 1];

#pragma omp parallel for reduction(+ : temp)
		for(int i = 0; i < nLayers - 1; i++) {
			for(int j = 0; j < nLayers - 1; j++)
				temp += (ed[i] - ed[i + 1]) * besselCoeff[i] * (ed[j] - ed[j + 1]) * besselCoeff[j];
		}
		intensity = temp + sq(ed[nLayers - 1] * besselCoeff[nLayers - 1]);
		intensity *= 16.0 * sq(sq(PI)) / (q * q * q);
	}
	   
    intensity *= scale;// * 1.0e-9; // the scale is way too low to use
    intensity += background;
 
	return intensity;   
}

std::complex<double> UniformHCModel::CalculateFF(Vector3d qvec, 
										   int nLayers, double w, double precision, VectorXd* p) {

    // TODO::ComplexModels Implement - Make sure the coefficients are correct!!
	std::complex<double> res(0.0, 0.0);
	double qx = qvec(0), qy = qvec(1), qz = qvec(2), qperp = sqrt(qx*qx + qy*qy);
	double q = sqrt(qx*qx + qy*qy + qz*qz);

	if(closeToZero(qperp)) {
		res = (ed[nLayers - 1] - edSolvent) * t[nLayers - 1] * t[nLayers - 1];
		for(int i = 0; i < nLayers - 1; i++) {
			res += (ed[i] - ed[i + 1]) * t[i] * t[i];
		}
	} else {
		res = (ed[nLayers - 1] - edSolvent) * t[nLayers - 1] * bessel_j1(t[nLayers - 1] * qperp);
		for(int i = 0; i < nLayers - 1; i++) {
			res += (ed[i] - ed[i + 1]) * t[i] * bessel_j1(t[i] * qperp);
		}
	}

	if(closeToZero(qperp)) {
		res *= 4.0 * PI * sinc(qz * H) * H / 2.0;
	} else {
		res *= 4.0 * PI * sinc(qz * H) * H / (qperp);
	}

	return res * (*extraParams)(0) + (*extraParams)(1); // Multiply by scale and add background
}

void UniformHCModel::PreCalculateFF(VectorXd& p, int nLayers) {
	OrganizeParameters(p, nLayers);
}

std::string UniformHCModel::GetLayerNameStatic(int layer)
{
	return CylindricalModel::GetLayerNameStatic(layer);
}

std::string UniformHCModel::GetLayerName(int layer)
{
	return GetLayerNameStatic(layer);
}

#pragma endregion

#pragma region Gaussian Cylindrical Model

GaussianHCModel::GaussianHCModel(int heightSteps, int radiiSteps, std::string name, int extras, int nlp, int minlayers, int maxlayers, EDProfile edp)
										: CylindricalModel(heightSteps, name, edp.shape, nlp, minlayers, maxlayers, extras), 
																	steps1(radiiSteps) {
}

bool GaussianHCModel::IsSlow() {
	return (!std::isinf(H) == 0);	// Doesn't work. The ! should not be there, but when it's not, it shows the infinite as 
}


void GaussianHCModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	CylindricalModel::OrganizeParameters(p, nLayers);
	r = (*parameters).col(2);
}

void GaussianHCModel::PreCalculate(VectorXd& p, int nLayers) {
	OrganizeParameters(p, nLayers);
	
	for(nonzero = 0; (nonzero < nLayers) && (t[nonzero] <= 0.0); nonzero++);

	if(nonzero >= nLayers)
		return;

	xxR = MatrixXd::Zero(nLayers - nonzero, steps1);
	wwR = MatrixXd::Zero(nLayers - nonzero, steps1);

	#pragma omp parallel for
	for(int i = nonzero; i < nLayers; i++) {
		VectorXd x, w;
		double s = std::min(0.0, r[i] - 3.0 * t[i]);
		SetupIntegral(x, w, s, r[i] + 3.0 * t[i], steps1);
		xxR.row(i - nonzero) = x;
		wwR.row(i - nonzero) = w;
	}
}

double GaussianHCModel::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0;
	
	if(!std::isinf(H)) {	// Finite Model
		double resouter = 0.0;
#pragma omp parallel for reduction(+ : resouter)
		for (int outer = 0; outer < steps; outer++) {
			double ressum = 0.0;
			for (int sum = nonzero; sum < nLayers; sum++) {
				if(pStop && *pStop)		// Place these line strategically in
					continue;			// slow models.

				int es = sum - nonzero;
				double resinner = 0.0;
				for (int inner = 0; inner < steps1; inner++) {
					resinner += exp(-4.0 * ln2 * sq(xxR(es,inner)-r[sum])/sq(t[sum])) *
						xxR(es,inner)* bessel_j0(q * sqrt(1-sq(xx[outer]))* xxR(es,inner)) * wwR(es,inner); 
				}
				ressum += resinner * (ed[sum] - edSolvent);
			}
			resouter += sq(ressum * sin(q * xx[outer] * H)/ xx[outer] ) * ww[outer];
		}
		intensity = 2.5 * resouter * 32.0 * sq(PI) * PI / sq(q) ; 
		//the 2.5 is a factor to normalize the ed area between this and the discrete model.
	} else {		// Infinite Model
		double ressum = 0.0;
#pragma omp parallel for reduction(+ : ressum)
		for (int sum = nonzero; sum < nLayers; sum++) {
			int es = sum - nonzero;
			double resinner = 0.0;

			if(pStop && *pStop)		// Place these line strategically in
				continue;			// slow models.

#pragma omp parallel for if(nLayers - nonzero < 2) reduction(+ : resinner)
			for (int inner = 0; inner < steps1; inner++ ) {
				resinner += exp(-4.0 * ln2 * sq(xxR(es,inner)-r[sum])/sq(t[sum])) *
					xxR(es,inner) * bessel_j0(q * xxR(es,inner)) * wwR(es,inner); 
			}
			ressum += resinner * (ed[sum] - edSolvent);
		}

		intensity = 2.0 * sq(ressum * sq(PI)) * 64.0 / q; // single integral
	//the 2.0 is a factor to normalize the ed area between this and the discrete model.
	}

	intensity *= (*extraParams)(0);	// Multiply by scale
	intensity += (*extraParams)(1);	// Add background

	return intensity;
}

std::string GaussianHCModel::GetLayerParamName(int index, EDPFunction *edpfunc) {
	return GetLayerParamNameStatic(index, edpfunc);
}
std::string GaussianHCModel::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {
	switch (index) {
		case 0:
			return "Thickness";
		case 2:
			return "Z_0";
		default:
			return CylindricalModel::GetLayerParamName(index, edpfunc);
	}
}

bool GaussianHCModel::IsParamApplicable(int layer, int lpindex) {
	if(layer == 0 && lpindex != 1)
		return false;
	return true;
}

struct GaussHCFunc
{
	GaussHCFunc(double _Qperp, double _R_i, double _Tau_i) :Qperp(_Qperp), R_i(_R_i), Tau_i(_Tau_i){}

	double operator()(double Rperp)
	{
		if (closeToZero(Qperp))
		{
			return exp(-4.0 * ln2 *sq(Rperp - R_i) / sq(Tau_i))
				* Rperp; //bessel j0(0) = 1

		}
		return exp(-4.0 * ln2 *sq(Rperp - R_i) / sq(Tau_i)) 
					* Rperp 
					*  bessel_j0(Qperp * Rperp);
	}

private:
	double Qperp, R_i, Tau_i;
};



std::complex<double> GaussianHCModel::CalculateFF(Vector3d qvec, int nLayers, double w, double precision, VectorXd* p) 
{
	double qx = qvec(0), qy = qvec(1), qz = qvec(2), qperp = sqrt(qx*qx + qy*qy);

	std::complex<double> res(0.0, 1.0);

	double mult = 4.0 * PI * sinc(qz * H) * H;
	
	double sum = 0;

	for (int i = 1; i < nLayers; i++)
		{
			double delPiG = ed[i] - ed[0]; 
			GaussHCFunc g(qperp, r[i], t[i]);
			res += delPiG * GaussKronrod15(g, 0., 5.2, 1e-6, 12, 3);
		}

	res *= mult;
	return res * (*extraParams)(0) + (*extraParams)(1); // Multiply by scale and add background


}

std::string GaussianHCModel::GetLayerNameStatic(int layer)
{
	return CylindricalModel::GetLayerNameStatic(layer);
}

std::string GaussianHCModel::GetLayerName(int layer)
{
	return GetLayerNameStatic(layer);
}

#pragma endregion
