#include "Quadrature.h"
#include "Externals/alglib.h"
#include <omp.h>

#include "Eigen/Core"

using Eigen::VectorXf;
using Eigen::VectorXd;

void SetupIntegral(VectorXd& x, VectorXd& w, 
				   double s, double e, int steps) {
	if(steps == 0)
		return;
	if(x.size() != steps ||
		x.size() != w.size() ||
		fabs(x[steps - 1] - e) > x[steps - 1] / steps ||
		fabs(x[0] - s) > (x[steps - 1] - x[0]) / steps)
		buildgausslegendrequadrature(steps, s, e, x, w);
}

void SetupIntegral(VectorXf& x, VectorXf& w, 
				   float s, float e, int steps) {
	if(steps == 0)
		return;
	if(x.size() != steps ||
		x.size() != w.size() ||
		fabs(x[steps - 1] - e) > x[steps - 1] / steps ||
		fabs(x[0] - s) > (x[steps - 1] - x[0]) / steps)
		buildgausslegendrequadrature(steps, s, e, x, w);
}


double GaussLegendre2D(ptF func, int resolution, double sx, double ex,
					   double sy, double ey) {
	double result = 0.0;

	static int _res = 0;
	static double _sx = 0, _sy = 0, _ex = 0, _ey = 0;
	static VectorXd x, wx, y, wy; 

	bool newRes = _res != resolution;

	if(newRes || fabs(_sx - sx) > 1e-7 || fabs(_ex - ex) > 1e-7) {
		SetupIntegral(x, wx, sx, ex, resolution);
		_res = resolution;
		_sx = sx;
		_ex = ex;
	}

	if(newRes || fabs(_sy - sy) > 1e-7 || fabs(_ey - ey) > 1e-7) {
		SetupIntegral(y, wy, sy, ey, resolution);
		_sy = sy;
		_ey = ey;
	}

	if(resolution <= 1)
		return result;
	
	#pragma omp parallel for default(shared) schedule(static) reduction(+ : result)
	for(int i = 0; i < resolution; i++) {
		double inner = 0.0;
		for(int j = 0; j < resolution; j++)
			inner += func(x[i], y[j]) * wy[j];
		result += inner * wx[i];
	}

	return result;
}
