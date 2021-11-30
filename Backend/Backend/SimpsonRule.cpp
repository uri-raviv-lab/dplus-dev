#include "Quadrature.h"


double SimpsonRule2D(ptF func, int resolution, double sx, double ex, 
					 double sy, double ey) {
	double finalresult = 0.0, h = (ex - sx) / (2.0 * double(resolution)),
		   k = (ey - sy) / (2.0 * double(resolution));
	int n = resolution;

	// a->b (x), c->d (y)
	// h = b - a / (2res)
	// k = d - c / (2res)
	// i = 0..2*res, p[i] = a + i * res, t[j] = c + j * res

	finalresult = func(sx, sy) + func(sx, ey) + func(ex, sy) + func(ex, ey);

	double intResult = 0.0;

	#pragma omp parallel for default(shared) schedule(static) reduction(+ : intResult)
	for(int i = 0; i < n; i++) {
		double result = 0.0;

		result += (4.0 * (func(sx, (sy + k*((2.0 * i) + 1.0))) +
			              func(ex, (sy + k*((2.0 * i) + 1.0))) ));

		result += (4.0 * (func((sx + h*((2.0 * i) + 1.0)), sy) +
			              func((sx + h*((2.0 * i) + 1.0)), ey) ));

		if(i < n - 1) {
			result += (2.0 * (func(sx, (sy + k*((2.0 * i) + 2.0))) +
				              func(ex, (sy + k*((2.0 * i) + 2.0))) ));

			result += (2.0 * (func((sx + h*((2.0 * i) + 2.0)), sy) +
							  func((sx + h*((2.0 * i) + 2.0)), ey) ));
		}

		for(int j = 0; j < n; j++) {
			result += (16.0 * func((sx + h*((2.0 * i) + 1.0)), 
			                       (sy + k*((2.0 * j) + 1.0))));

			if(j < n - 1) {
				result += (8.0 * func((sx + h*((2.0 * i) + 1.0)), 
					                  (sy + k*((2.0 * j) + 2.0))));

	
				result += (8.0 * func((sx + h*((2.0 * j) + 2.0)), 
					                  (sy + k*((2.0 * i) + 1.0))));

				result += (4.0 * func((sx + h*((2.0 * i) + 2.0)), 
					                  (sy + k*((2.0 * j) + 2.0))));
			}
		}

		intResult += result;
	}

	finalresult += intResult;
	finalresult *= (1.0 / 9.0) * h * k;

	return finalresult;
}