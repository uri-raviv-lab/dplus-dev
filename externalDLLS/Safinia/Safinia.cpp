// StackRings.cpp : Defines the exported functions for the DLL application.
//
#include "Safinia.h"
#include "Quadrature.h"
#include <limits>
//#include "..\..\X+\Calculation\mathfuncs.h"

// OpenGL includes
#ifdef _WIN32
#include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>
// END of OpenGL includes

template <typename T> inline T sq(T x) { return x * x; }

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef EPS
#define EPS 3.0e-11
#endif

#ifndef miN
#define miN(a,b)	(a < b) ? a : b
#endif

double bessel_j0(double x);
double bessel_j1(double x);
double bessel_jn(int n, double x);

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
				ModelCategory mc = { "Safinya's Models", MT_FORMFACTOR, {0, 1, 2, 3, -1} };
				return mc;
			}
	}
}

// Returns the number of models in this container
int GetNumModels() {
	return 4;	
}

// Returns the model's display name from the index. Supposed to return "N/A"
// for indices that are out of bounds
ModelInformation GetModelInformation(int index) {
	switch(index) {
		default:
			return ModelInformation("N/A");
		case 0:
			return ModelInformation("Stack of Rings", 0, index, true, 1, 6, 6, 2);
		case 1:
			return ModelInformation("Helix w Lorentzian Hexagonal SF", 0, index, true, 2, 2 * NINDICES, 2 * NINDICES, 10);
		case 2:
			return ModelInformation("Uniform Hollow Cylinder with Hexagonal Structure Factor (Lorentzian)", 0, index, true, 2, 9, 9, 8+3);
		case 3:
			return ModelInformation("Finite Helix", 0, index, true, 0, 0, 0, 7);
	}
}

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
Geometry *GetModel(int index) {
	switch(index) {
		default:
			return NULL;
		case 0:
			return new StackRings();
		case 1:
			return new HelixWSF();
		case 2:
			return new HCwLorSFModel();
		case 3:
			return new FiniteHelixModel();
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// StackRings functions
StackRings::StackRings(std::string st, ProfileType edp, int steps) : FFModel(st, 2, 1, 6, 6, EDProfile(NONE)) {
	SetupIntegral(xx, ww, 0.0 - EPS, 1.0 + EPS, steps);
	rtx = xx.array();
	rtx = (1.0 - rtx * rtx).sqrt();

}

double StackRings::GetDefaultParamValue(int paramIndex, int layer) {
	switch(layer) {
		default:
			return 1.0;
		case 0:
			return 100.0;	
		case 1:
			return 6.0;	
		case 2:
			return 4.0;	
		case 3:
			return 10.0;	
		case 4:
			return 8.7;	
		case 5:
			return 4.9;	
	}
}

std::string StackRings::GetLayerParamName(int index) {
	return "Value";
}

std::string StackRings::GetLayerName(int layer) {
	std::stringstream ss;
	switch(layer) {
		case 0:
			ss << "Lz"; break;
		case 1:
			ss << "d"; break;
		case 2:
			ss << "T"; break;
		case 3:
			ss << "ED"; break;
		case 4:
			ss << "Ri"; break;
		case 5:
			ss << "Wall Thickness"; break;
	}

	return ss.str();
}


void StackRings::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
	}

void StackRings::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);
	pars = (*parameters).col(0);

	lz	= pars(0);
	d	= pars(1);
	T	= pars(2);
	ED	= pars(3);
	Ri	= pars(4);
	Ro	= pars(5);

	scale	= (*extraParams)(0);
	bg		= (*extraParams)(1);

}


	

double StackRings::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0;

#pragma omp parallel for reduction(+ : intensity)
	for(int i = 0; i < xx.size(); i++) {
		double sum = 0.0;
		for(int h = 0; h < 2; h++)
			sum += exp(-sq(lz * (q * xx[i] - 2.0 * PI * double(h) / d) )/(4.0 * PI));

		sum *= sq(4.0 * PI * sin(T * q * xx[i]) / (sq(q)  * xx[i])) / (1.0 - sq(xx[i]));
		sum *= sq(ED * ( (Ro + Ri) * bessel_j1((Ro + Ri) * q * rtx[i]) - Ri * bessel_j1(Ri * q * rtx[i]) ) ) * ww[i];
		
		intensity += sum;
	}

	return intensity * scale + bg;
}

std::complex<double> StackRings::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam StackRings::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 0.01);
	}
}

bool StackRings::IsLayerBased() {
	return false;
}

// HelixWSF functions
HelixWSF::HelixWSF(std::string st, ProfileType edp, int  xSteps, int phiSteps) : 
	FFModel(st, 10, 2, 2 * NINDICES, 2 * NINDICES, EDProfile(NONE)) {
#pragma omp parallel sections
		{
#pragma omp section 
			{
				SetupIntegral(xx, ww, 0.0, 1.0, xSteps);
				rtx = xx.array();
				rtx = (1.0 - rtx * rtx).sqrt();
			}
#pragma omp section 
			{
				SetupIntegral(xPhi, wPhi, 0.0, 2.0 * PI, phiSteps);
				cs = sn = xPhi.array();
				sn = sn.sin();
				cs = cs.cos();
			}
		}

		rr2 = sqrt(sqrt(2.0) - 1.0);
}

double HelixWSF::GetDefaultParamValue(int paramIndex, int layer) {
	switch(paramIndex) {
		default:
		case 0:
			return 1.0;
		case 1:
			return 0.075;
	}
}

std::string HelixWSF::GetLayerParamName(int index) {
	switch(index) {
		default:
		case 0:
			return "Amplitude";
		case 1:
			return "Lambda";
	}
}

std::string HelixWSF::GetLayerName(int layer) {
	int hh = 1, kk, cnt = 0, c = 2, d = 0;
	while(cnt + c  - 1 < layer) {
		cnt += c++;
	}

	hh = c - 1;
	kk = layer - cnt;
	
	std::stringstream ss;

	ss << "(" << hh << ","<< kk << ")";
	if (hh != kk)
		ss << ", (" << kk << ","<< hh << ")";
	return ss.str();
}

void HelixWSF::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void HelixWSF::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);

	Eigen::ArrayXd aa	= (*parameters).col(0);
	Eigen::ArrayXd La	= (*parameters).col(1);

	// kMax = hMax = int(qmax * a / (2.0 * PI) + 0.5);
	kMax = hMax = -1 + int(sqrt(double(1 + 2 * NINDICES)) + 0.3);	// The 0.3 is to ensure that the right integer is chosen from the double
	kMax = hMax = 8;
		
	A.resize(hMax + 1, hMax + 1);
	lambda.resize(hMax + 1, hMax + 1);

	for(int h = 1, cnt = 0; h <= hMax; h++) {
		for(int k = 0; k <= h; k++) {
			int rows = A.rows(), cols = A.cols(), sz = A.size();
			A(h, k) = aa(cnt);
			lambda(h, k) = La(cnt++);
		}
	}

	FFScale	= (*extraParams)(0);
	FFBg	= (*extraParams)(1);
	SFScale	= (*extraParams)(2);
	SFBg	= (*extraParams)(3);
	R		= (*extraParams)(4);
	P		= (*extraParams)(5);
	Rg		= (*extraParams)(6);
	N		= (*extraParams)(7);
	ED		= (*extraParams)(8);
	a		= (*extraParams)(9);

}



double HelixWSF::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0, preFac = sq(0.5 * ED/* * P * N *// PI) * exp(-0.4 * sq(q * Rg));

#pragma omp parallel for reduction(+ : intensity)
	for(int ix = 0; ix < xx.size(); ix++) {
		if(pStop && *pStop)		// Place these line strategically in
			continue;			// slow models.

		double outer = 0.0, inner = 0.0, innerS = 0.0;

		for(int n = 0; n < 2; n++)
			outer += sq(bessel_jn(n, q * R * rtx[ix])) * exp(-sq(N * P * (q * xx[ix] - 2.0 * PI * n / P)) / (4.0 * PI));

		// Integral over phi
		for(int ip = 0; ip < xPhi.size(); ip++) {
			for(int h = 1; h <= hMax; h++) {
				for(int k = 0; k <= h; k++) {	// remove the "min" for high q
					double gx = 2.0 * PI * double(h) / a;
					double gy = 2.0 * PI * ((double(k) + 0.5 * double(h)) / a) / (0.8660254038/*sin(2pi/3)*/);

					double tmp = A(h, k) * lambda(h,k) * lambda(h,k) * lambda(h,k) * (2.0 - double(int(h == k)));
					tmp /= 4.0 * PI * sq(rr2) * rr2 * sq( sq(q) - 2.0 * q * rtx[ix] * (gx * cs[ip] + gy * sn[ip]) + sq(gHex(h, k)) + sq(lambda(h,k) / (2.0 * rr2)));
					innerS += tmp;
				}	//for k
			}	// for h
			inner += innerS * wPhi[ip];
		}	// for ip
		//outer *= inner;
		intensity += (outer * ww[ix] * FFScale * preFac + (FFBg * ww[ix] / 2.0)) * (inner * ww[ix] * SFScale * preFac /** exp(-sq(q * debyeWaller) / 2.0) */+ SFBg);
	}	// for ix

	return intensity;
}

std::complex<double> HelixWSF::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam HelixWSF::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 0.01);
	case 2:
		return ExtraParam("SF Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 3:
		return ExtraParam("SF Background", 1.0);
	case 4:
		return ExtraParam("Helix Radius", 20.2);
	case 5:
		return ExtraParam("Pitch", 5.3);
	case 6:
		return ExtraParam("Rg", 2.4);
	case 7:
		return ExtraParam("N", 300.0);
	case 8:
		return ExtraParam("ED", 10.0);
	case 9:
		return ExtraParam("Lattice Parameter", 41.0);
	}
}

bool HelixWSF::IsLayerBased() {
	return false;
}

double HelixWSF::gHex(int h, int k) {
	return 2.0 * PI * sqrt( sq(double(h)/a) + (1/sq(0.8660254038/*sin(2pi/3)*/)) * sq( (double(k) + 0.5 * double(h)) / a));
}


#undef min
#undef max


double bessel_j0(double x)
{
    double ax,z;
    double xx,y,ans,ans1,ans2;
    if ((ax=fabs(x)) < 8.0) {
        y=x*x;
        ans1=57568490574.0+y*
            (-13362590354.0+y*(651619640.7
                               +y*(-11214424.18+y*
                                   (77392.33017+y*(-184.9052456)))));
        ans2=57568490411.0+y*
            (1029532985.0+y*(9494680.718
                             +y*(59272.64853+y*(267.8532712+y*1.0))));
        ans=ans1/ans2;
    }
    else {
        z=8.0/ax;
        y=z*z;
        xx=ax-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*
                    (0.2734510407e-4
                     +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*
            (0.1430488765e-3
             +y*(-0.6911147651e-5+y*(0.7621095161e-6
                                     -y*0.934935152e-7)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    }
    
    return ans;
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

double bessel_jn(int n, double x) {
	static const double ACC = 160.0;
	static const int IEXP = std::numeric_limits<double>::max_exponent/2;
	bool jsum;
	int j,k,m;
	double ax,bj,bjm,bjp,dum,sum,tox,ans;
	if (n==0) return bessel_j0(x);
	if (n==1) return bessel_j1(x);
	ax=fabs(x);
	if (ax*ax <= 8.0 * std::numeric_limits<double>::min())
		return 0.0;
	else if (ax > double(n)) {
		tox=2.0/ax;
		bjm=bessel_j0(ax);
		bj=bessel_j1(ax);
		for (j=1;j<n;j++) {
			bjp=j*tox*bj-bjm;
			bjm=bj;
			bj=bjp;
		}
		ans=bj;
	} else {
		tox=2.0/ax;
		m=2*((n+int(sqrt(ACC*n)))/2);
		jsum=false;
		bjp=ans=sum=0.0;
		bj=1.0;
		for (j=m;j>0;j--) {
			bjm=j*tox*bj-bjp;
			bjp=bj;
			bj=bjm;
			dum=frexp(bj,&k);
			if (k > IEXP) {
				bj=ldexp(bj,-IEXP);
				bjp=ldexp(bjp,-IEXP);
				ans=ldexp(ans,-IEXP);
				sum=ldexp(sum,-IEXP);
			}
			if (jsum) sum += bj;
			jsum=!jsum;
			if (j == n) ans=bjp;
		}
		sum=2.0*sum-bj;
		ans /= sum;
	}
	return x < 0.0 && (n & 1) ? -ans : ans;
}

#pragma region Uniform Hollow Cylinder with a Hexagonal Structure Factor (Lorentzian)

// GHCwLorSFModel functions
HCwLorSFModel::HCwLorSFModel(std::string st, ProfileType edp, int steps) : FFModel(st, 8+3, 2, 9, 9){
	
	SetupIntegral(phi, wphi, 0.0, 2.0 * PI, steps);
	SetupIntegral(xx, ww, -1.0 + EPS, 1.0 - EPS, 500);
}


double HCwLorSFModel::GetDefaultParamValue(int paramIndex, int layer) {
	switch(paramIndex) {
		default:
		case 0:
			return 1.0;
		case 1:
			return 0.075;
	}
}

std::string HCwLorSFModel::GetLayerParamName(int index) {
	switch(index) {
		default:
		case 0:
			return "Amplitude";
		case 1:
			return "Lambda";
	}
}

std::string HCwLorSFModel::GetLayerName(int layer) {
	int hh = 1, kk, cnt = 0, c = 2, d = 0;
	while(cnt + c  - 1 < layer) {
		cnt += c++;
	}

	hh = c - 1;
	kk = layer - cnt;
	
	std::stringstream ss;

	ss << "(" << hh << ","<< kk << ")";
	if (hh != kk)
		ss << ", (" << kk << ","<< hh << ")";
	return ss.str();
}


void HCwLorSFModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void HCwLorSFModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	Geometry::OrganizeParameters(p, nLayers);
	Eigen::ArrayXd aa	= (*parameters).col(0);
	Eigen::ArrayXd La	= (*parameters).col(1);

	// kMax = hMax = int(qmax * a / (2.0 * PI) + 0.5);
	kMax = hMax = -1 + int(sqrt(double(1 + 2 * NINDICES)) + 0.3);	// The 0.3 is to ensure that the right integer is chosen from the double
	
	// For Safinia's model (MT through q=1.0nm^{-1}) only the first 4 indices are needed.
	kMax = hMax = 5;
	
	A.resize(hMax + 1, hMax + 1);
	lambda.resize(hMax + 1, hMax + 1);

	for(int h = 1, cnt = 0; h <= hMax; h++) {
		for(int k = 0; k <= h; k++) {
			A(h, k) = aa(cnt);
			lambda(h, k) = La(cnt++);
		}
	}

	FFScale  = (*extraParams)(0);
	FFBg	 = (*extraParams)(1);
	SFScale  = (*extraParams)(3);
	SFBg	 = (*extraParams)(4);
	H  = (*extraParams)(2);
	lf = (*extraParams)(5);
	a  = (*extraParams)(6);
	debyeWaller = (*extraParams)(7);
	ED = (*extraParams)(8);
	ri = (*extraParams)(9);
	ro = (*extraParams)(10);
}

VectorXd HCwLorSFModel::CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p) {
	qmax = q[q.size() - 1];
	return FFModel::CalculateVector(q, nLayers, p);
}

double HCwLorSFModel::Calculate(double q, int nLayers, VectorXd& p) {
	double sf = hexSF2D(q, nLayers, p);

	return sf;
}

double HCwLorSFModel::hexSF2D(double q, int nLayers, VectorXd& p) {
	if(p.size() > 0)
		OrganizeParameters(p, nLayers);

	double intensity = 0.0;
	double rtx = 0.0, rr2 = sqrt(sqrt(2.0) - 1.0);
	bool conv = false;

	// Integral over x
#pragma omp parallel for reduction(+ : intensity)
	for(int ix = 0; ix < xx.size(); ix++) {

		if(pStop && *pStop)		// Place these line strategically in
			continue;			// slow models.

		double outer, inner = 0.0, innerS = 0.0;
		rtx = sqrt(1.0 - sq(xx[ix]));
		outer = sq(4.0 * PI * sin(q * H * xx[ix]) / (sq(q) * xx[ix])) / (1.0 - sq(xx[ix]));
		outer *= sq( ED * ( (ro + ri) * bessel_j1((ro + ri) * q * rtx) - ri * bessel_j1(ri * q * rtx) ) ) * ww[ix];

		// Integral over phi
		for(int ip = 0; ip < phi.size(); ip++) {
			for(int h = 1; h < hMax; h++) {
				for(int k = 0; k <= (miN(h, (int)2)); k++) {
					double gx = 2.0 * PI * double(h) / a;
					double gy = 2.0 * PI * ((double(k) + 0.5 * double(h)) / a) / (0.8660254038/*sin(2pi/3)*/);

					double tmp = A(h, k) * lambda(h,k) * (2.0 - double(int(h == k)));
					tmp /= 4.0 * PI * sq(rr2) * rr2 * sq( sq(q) + 2.0 * q * rtx * (gx * cos(phi[ip]) + gy * sin(phi[ip])) + sq(gHex(h, k)) + sq(lambda(h,k) / 2.0 * rr2));
					innerS += tmp;
				}	//for k
			}	// for h
			inner += innerS * wphi[ip];
		}	// for ip
		//outer *= inner;
		intensity += (outer * FFScale + (FFBg * ww[ix] / 2.0)) * (inner * ww[ix] * SFScale * exp(-sq(q * debyeWaller) / 2.0) + SFBg);
	}	// for ix

	//intensity *= sq(sq(lf / a));	// N^4
	return intensity;

}

std::complex<double> HCwLorSFModel::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam HCwLorSFModel::GetExtraParameter(int index) {
	switch(index) {
	case 2:
		return ExtraParam("Length", 100.0, false, true);
	case 3:
		return ExtraParam("SF Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 4:
		return ExtraParam("SF Background", 10.0e-2, false, false);
	case 5:
		return ExtraParam("Lf", 200.0, false, true);
	case 6:
		return ExtraParam("Lattice Spacing", 25.0, false, true);
	case 7:
		return ExtraParam("Debye-Waller", 0.0, false, true);
	case 8:
		return ExtraParam("ED Contrast", 67.0, false, true);
	case 9:
		return ExtraParam("Inner Radius", 8.0, false, true);
	case 10:
		return ExtraParam("Outer Radius", 4.0, false, true);
	default:
		return FFModel::GetExtraParameter(index);
	}
}

bool HCwLorSFModel::IsLayerBased() {
	return false;
}

double HCwLorSFModel::gHex(int h, int k) {
	return 2.0 * PI * sqrt( sq(double(h)/a) + (1/sq(0.8660254038/*sin(2pi/3)*/)) * sq( (double(k) + 0.5 * double(h)) / a));
}

#pragma endregion

#pragma region Finite Helix

// HelixWSF functions
FiniteHelixModel::FiniteHelixModel(std::string st, ProfileType edp, int  xSteps) : 
	FFModel(st, 7, 0, 0, 0, EDProfile(NONE)) {

		SetupIntegral(xx, ww, 0.0, 1.0, xSteps);
		rtx = xx.array();
		rtx = (1.0 - rtx * rtx).sqrt();

		rr2 = sqrt(sqrt(2.0) - 1.0);
}

double FiniteHelixModel::GetDefaultParamValue(int paramIndex, int layer) {
	return -1.0;
}

std::string FiniteHelixModel::GetLayerParamName(int index) {
	return "N/A";
}

std::string FiniteHelixModel::GetLayerName(int layer) {
	return "N/A";
}

void FiniteHelixModel::PreCalculate(VectorXd& p, int nLayers){
	OrganizeParameters(p, nLayers);
}

void FiniteHelixModel::OrganizeParameters(const Eigen::VectorXd &p, int nLayers) {
	FFModel::OrganizeParameters(p, nLayers);

	FFScale	= (*extraParams)(0);
	FFBg	= (*extraParams)(1);
	R		= (*extraParams)(2);
	P		= (*extraParams)(3);
	Rg		= (*extraParams)(4);
	N		= (*extraParams)(5);
	ED		= (*extraParams)(6);
}

double FiniteHelixModel::Calculate(double q, int nLayers, VectorXd& p) {
	double intensity = 0.0, preFac = sq(0.5 * ED/* * P * N *// PI) * exp(-0.4 * sq(q * Rg));

#pragma omp parallel for reduction(+ : intensity)
	for(int ix = 0; ix < xx.size(); ix++) {
		if(pStop && *pStop)		// Place these line strategically in
			continue;			// slow models.

		double outer = 0.0, inner = 0.0, innerS = 0.0;

		for(int n = 0; n < 4; n++)
			outer += sq(bessel_jn(n, q * R * rtx[ix])) * exp(-sq(N * P * (q * xx[ix] - 2.0 * PI * n / P)) / (4.0 * PI));
		intensity += (outer * ww[ix] * FFScale * preFac + (FFBg * ww[ix] / 2.0));
	}	// for ix

	return intensity;
}

std::complex<double> FiniteHelixModel::CalculateFF(Vector3d qvec, int nLayers, double w, 
											  double precision, VectorXd& p) {
	return std::complex<double> (0.0, 1.0);
}

ExtraParam FiniteHelixModel::GetExtraParameter(int index) {
	switch(index) {
	default:
		return ExtraParam("N/A");
	case 0:
		return ExtraParam("Scale", 1.0, false, true, false, 0.0, 0.0, false, 12);
	case 1:
		return ExtraParam("Background", 0.01);
	case 2:
		return ExtraParam("Helix Radius", 20.2);
	case 3:
		return ExtraParam("Pitch", 5.3);
	case 4:
		return ExtraParam("Rg", 2.4);
	case 5:
		return ExtraParam("N", 300.0);
	case 6:
		return ExtraParam("ED", 10.0);
	}
}

bool FiniteHelixModel::IsLayerBased() {
	return false;
}

#pragma endregion
