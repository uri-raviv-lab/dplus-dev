#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <algorithm>
#include "mathfuncs.h"

#undef min
#undef max

std::vector <double> xGlob;
void SetX(std::vector <double> xIn) {
	xGlob = xIn;
}

double DenormGaussianFW(double fwhm, double xc, double A, double B, double x) {
	double result = 0.0, sigma;

	sigma = fwhm / RT2LN2;

	result = exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));

	return (A * result) + B;
}

double gaussianFW(double fwhm, double xc, double A, double B, double x) {
	
	double result = 0.0, sigma;
	
	sigma = fwhm / RT2LN2;

	result = exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));

	// Normalization
	result /= sqrt(2.0 * M_PI) * sigma;

	return (A * result) + B;
}

double gaussianSig(double fwhm, double xc, double A, double B, double x) {
	
	double result = 0.0, sigma;
	
	sigma = fwhm;

	result = exp(-sq(x - xc) / (2.0 * sigma * sigma));

	// Normalization
	result /= sqrt(2.0 * M_PI) * sigma;

	return (A * result) + B;
}

double lorentzian(double fwhm, double xc, double A, double B, double x) {
	double result = 0.0, hwhm = fwhm / 2.0;

	result = hwhm;

	result /= (sq(x - xc) + sq(hwhm));

	// Normalization
	result /= M_PI;

	return (A * result) + B;
}

double lorentzian_squared(double fwhm, double xc, double A, double B, double x) {
	double result = 1.0, fwqm = fwhm / RTRT2m1;

	result *= fwqm * fwqm * fwqm;

	result /= 4.0 * PI * sq(sq(x - xc) + sq(fwqm / 2.0));

	return (A * result) + B;
}

double Caille_peak(double fwhm, double xc, double A, double B, double x) {
		return A * (pow(fabs(x - xc),(-1 + fwhm)))+ B;
		/**
		**/
}

double exponentDecay(double x, double base, double decay, double xcenter) {
	return base*exp(-(x - xcenter) / decay); 
}

double linearFunction(double x, double base, double decay, double xcenter) {
	return (-base * (x) + decay);
}

double powerFunction(double x, double base, double decay, double xcenter) {
	if(fabs(x - xcenter) <= 1e-6) return 0.0;
	return (base * pow((x - xcenter), (-decay)));
}

inline double vmax(const std::vector<double>& v) {
	if(v.size() == 0)
		return 0.0;
	double val = v.at(0);
	for(unsigned int i = 0; i < v.size(); i++)
		if(v[i] > val)
			val = v[i];
	return val;
}

std::vector <double> MachineResolution(const std::vector <double> &q, const std::vector <double> &orig, double sig) {
	if(fabs(sig) < 1e-12) return orig;
	int size = (int)orig.size();
	std::vector <double> res(size);
	
	res[0] = orig[0];
	res[size - 1] = orig[size - 1];
	for(int i = 1; i < size - 1; i++) {
		if((fabs(q[i] - q[i+1]) > 3.0 * sig) ){
			res[i] = orig[i];
			continue;
		}

		double val = 0.0;
		double norm = 0.0;
		
		// Worse case scenario values
		int start = 0;
		int end = size - 1;
		double quan1 = std::max(q[0], q[i] - 3.0 * sig),		// Lower limit
			quan2 = std::min(q[i] + 3.0 * sig, q[size - 1]);	// Upper limit
		// Find the upper limit in the array
		for (int j = i; j < size; j++) {
			if (q[j] >= quan2) {
				end = j;
				break;
			}
		}
		// Find the lower limit in the array
		for (int j = i; j >= 0; j--) {
			if (q[j] <= quan1) {
				start = j;
				break;
			}
		}

		// Symmetrize the limits 
		if(fabs(fabs( q[i] - q[start]) - fabs( q[i] - q[end])) > 1e-4) {
			
			if(fabs(q[i] - q[start])<fabs( q[i] - q[end]))
				while (( q[i] - q[start]) + ( q[i] - q[end]) > -1e-4) end--;
			else
				while (( q[i] - q[start]) + ( q[i] - q[end]) < 1e-4) start++;
		}
	

		for (int j = start; j <= end; j++) {
			double g = gaussianSig(sig,  q[i],  1.0,  0.0, q[j]);
			val += orig[j] * g;
			norm += g;
		}

		res[i] = val / norm;
	}
	return res;

	/****Avi's version****/
	/****Would work fine if we didn't want to crop the Gaussian symmetrically****/
	//int j;
	//double wt = 0.0, wtot = 0.0;
	//res.resize(size, 0.0);

	//for(int i = 0; i < size; i++) {
	//	wtot = 0.0;

	//	// Find the first point within 3 sigma
	//	for(j = i; j >= 0; j--) {
	//		if(fabs(q[i] - q[j]) < 3.0 * sig) {
	//			j++;
	//			break;
	//		}
	//	}
	//	// Find the last point within 3 sigma
	//	for(end = i; end < size; end++) {
	//		if(fabs(q[i] - q[end]) < 3.0 * sig) {
	//			end--; //?
	//			break;
	//		}
	//	}
	//	end = (end < j) ? end : j;

	//	// Average weighted adjacent points
	//	for(j = i - end; j < i + end; j++) {
	//		if(fabs(q[i] - q[j]) < 3.0 * sig)
	//			break;
	//		wt = gaussianSig(sig, q[i], 1.0, 0.0, q[j]);
	//		wtot += wt;
	//		res[i] += wt * orig[j];
	//	}
	//	res[i] /= wtot;
	//}
	/****End of Avi's version****/
}

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
	if (ax*ax <= 8.0 * std::numeric_limits<double>::min()) return 0.0;
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

double bessel_i0(double x) {
	double ax, ans, y;
	if((ax=fabs(x)) < 3.75) {
		y = x / 3.75;
		y *= y;
		ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))) );
	} else {
		y = 3.75 / ax;
		ans = (exp(ax)/sqrt(ax)) * (0.39894228 + y * (0.132859e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 
			+ y * (-0.2057706e-1) + y * (0.2635537e-1) + y * (-0.1647633e-1) + y * (0.392377e-2))))));
	}

	return ans;
}


