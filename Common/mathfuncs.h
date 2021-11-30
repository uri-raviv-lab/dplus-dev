#ifndef __MATHFUNCS_H
#define __MATHFUNCS_H

#include <limits>
#include <vector>
#include "Common.h"

// Common numerical factors
#ifndef ln2
#define ln2 0.69314718055995
#endif

#ifndef RTRT2m1
#define RTRT2m1 0.643594252905582624735443 // (sqrt(sqrt(2) - 1)
#endif

#ifndef RT2LN2
#define RT2LN2 2.354820045030949 // (2.0*sqrt (2.0*log (2.0)))
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef PIx2
#define PIx2 6.2831853071795864769
#endif

#ifndef EPS
#define EPS 3.0e-11
#endif

// Peak Shapes
double EXPORTED_BE_FUNCTION gaussianSig(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION gaussianFW(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION DenormGaussianFW(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION lorentzian(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION lorentzian_squared(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION Caille_peak(double fwhm, double xc, double A, double B, double x);
double EXPORTED_BE_FUNCTION CailleDummy(double fwhm, double xc, double A, double B, double x, double &N_diff);

// Background Shapes
double EXPORTED_BE_FUNCTION exponentDecay(double x, double base, double decay, double xcenter);
double EXPORTED_BE_FUNCTION linearFunction(double x, double base, double decay, double xcenter);
double EXPORTED_BE_FUNCTION powerFunction(double x, double base, double decay, double xcenter);

// Bessel functions (of first, second and nth order)
double EXPORTED_BE_FUNCTION bessel_j0(double x);
double EXPORTED_BE_FUNCTION bessel_j1(double x);
double EXPORTED_BE_FUNCTION bessel_jn(int n, double x);

template<typename T>
T sinc(T x) {
	if(fabs(x) < 100.0 * std::numeric_limits<T>::epsilon()) {
		return 1.0;
	}
	return sin(x) / x;
}

inline bool closeToZero(double x)
{
	return (fabs(x) < 100.0 * std::numeric_limits<double>::epsilon());
}

// Integer powers
inline double ipow(double x, int pow) {
	if(pow == 0)
		return 1.0;
	if(pow < 0)
		return 1.0 / ipow(x, -pow);
	if(pow == 1)
		return x;

	double tmp = ipow(x, pow / 2);	// == ipow(x, pow / 2);
	if((pow % 2) == 0)	// ==if(pow % 2 == 0)
		return tmp * tmp;
	else return x * tmp * tmp;
}

// Double comparison (the Avi way)
inline bool isequal(double a, double b, int significant) {
	if(fabs(1.0 - (a / b)) > ipow(10.0, -significant))
		return false;

	return true;
}

// Modified Bessel function (zeroth order)
// double accuracy problem: bessel_i0(701) == #inf
double bessel_i0(double x);

template <typename T> inline T sq(T x) { return x * x; }
	
void SetX(std::vector <double> xIn);

std::vector <double> MachineResolution(const std::vector <double> &q ,const std::vector <double> &orig, double width);




static const double g7x[] = {
	-0.9491079123427585245261897, -0.7415311855993944398638648, -0.4058451513773971669066064, 0.0,
	0.4058451513773971669066064, 0.7415311855993944398638648, 0.9491079123427585245261897
};
static const double g7w[] = {
	0.1294849661688696932706114, 0.2797053914892766679014678, 0.3818300505051189449503698, 0.4179591836734693877551020,
	0.3818300505051189449503698, 0.2797053914892766679014678, 0.1294849661688696932706114
};

static const double k15x[] = {
	-0.9914553711208126392068547, -0.9491079123427585245261897, -0.8648644233597690727897128, -0.7415311855993944398638648,
	-0.5860872354676911302941448, -0.4058451513773971669066064, -0.2077849550078984676006894, 0.0,
	0.2077849550078984676006894, 0.4058451513773971669066064, 0.5860872354676911302941448, 0.7415311855993944398638648,
	0.8648644233597690727897128, 0.9491079123427585245261897, 0.9914553711208126392068547
};
static const double k15w[] = {
	0.0229353220105292249637320, 0.0630920926299785532907007, 0.1047900103222501838398763, 0.1406532597155259187451896,
	0.1690047266392679028265834, 0.1903505780647854099132564, 0.2044329400752988924141620, 0.2094821410847278280129992,
	0.2044329400752988924141620, 0.1903505780647854099132564, 0.1690047266392679028265834, 0.1406532597155259187451896,
	0.1047900103222501838398763, 0.0630920926299785532907007, 0.0229353220105292249637320
};
template<typename F>
double GaussKronrod15(F func, double lowEnd, double highEnd, double epsilon, int maxDepth, int minDepth)
{
	return GaussKronrod15_Impl(func, lowEnd, highEnd, epsilon, maxDepth, minDepth) * (highEnd - lowEnd) / 2;
}
template<typename F>
double GaussKronrod15_Impl(F func, double lowEnd, double highEnd, double epsilon, int maxDepth, int minDepth)
{
	double kVal = 0.0, gVal = 0.0, delta = (highEnd - lowEnd), halfDelta = delta * 0.5;
	double mid = (highEnd + lowEnd) / 2.0;
	double kVals[15] = { 0.0 }, gVals[7] = { 0.0 };
	for (int i = 0; i < 15; i++) {

		kVals[i] = func(k15x[i] * halfDelta + mid);
		kVal += kVals[i] * k15w[i];
		if (i % 2 == 1) {
			gVals[i / 2] = kVals[i];
			gVal += gVals[i / 2] * g7w[i / 2];
		}
	}

	if (minDepth < 0 && (maxDepth < 0 || fabs(1.0 - (gVal / kVal)) <= epsilon || gVal == kVal)) {
		return kVal;
	}

	return
		(GaussKronrod15_Impl(func, lowEnd, mid, epsilon, maxDepth - 1, minDepth - 1) +
		GaussKronrod15_Impl(func, mid, highEnd, epsilon, maxDepth - 1, minDepth - 1)) / 2;

}

#endif
