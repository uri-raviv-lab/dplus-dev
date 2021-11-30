#include "Quadrature.h"

#include <cstdlib>
#include <ctime>

static inline double generatePoint(double a, double b) {
	return ((((double)rand() / (double)RAND_MAX) * (b - a) + a));
}

double MonteCarlo2D(ptF func, int resolution, double sx, double ex, 
				    double sy, double ey) {
	double finalresult = 0.0, res = double(resolution);
	static bool bInit = false;

	if(!bInit) {
		srand((unsigned)time(NULL));
		bInit = true;
	}

	if(!func)
		return -1.0;

	#pragma omp parallel for default(shared) schedule(static) reduction(+ : finalresult)
	for(int iter = 0; iter < 50; iter++) {
		double result = 0.0;
		for(int i = 0; i < resolution; i++)
			result += (func(generatePoint(sx, ex), generatePoint(sy, ey))  / res);

		//result /= double(resolution);

		finalresult += result * (ex - sx) * (ey - sy);
	}

	finalresult /= 50.0;
	
	return finalresult;
}

