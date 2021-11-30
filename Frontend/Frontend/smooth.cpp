//#include "smooth.h"
#include "FrontendExported.h"
#include <cmath>
#include <vector>
#include "Eigen/Core"

using Eigen::ArrayXd;
using std::vector;

template <typename T> inline T sq(T x) { return x * x; }

void smoothVector(int strength, vector<double>& data) {
	vector<double> newy = data;

	for(int iter = 0; iter < strength; iter++) {
		newy[0] = data[0];
		newy[data.size() - 1] = data[data.size() - 1];
		for(int i = 1; i < (int)data.size() - 1; i++)
			newy[i] = 0.25 * (data[i - 1] + (2.0 * data[i]) + data[i + 1]);
		
		data = newy;
	}
}

// Smoothing
std::vector<double> bilateralFilter(std::vector<double> y, std::vector<double> x, double sigD, double sigR) {
	// Convert std::vector to Eigen?
	if((y.size() != x.size()) ||
		(sigD == 0.0) ||	// Think about changing
		(sigR == 0.0))
		return y;
	int size = (int)y.size();
	std::vector<double> resY;
	resY.resize(y.size(), 0.0);

#pragma omp parallel for
	for(int i = 0; i < size; i++) {
		double k = 0.0, tmp;
		for(int j = 0; j < size; j++) { // Integration
			tmp = exp(-0.5 * sq((x[j] - x[i]) / sigD)) * exp(-0.5 * sq((y[j] - y[i]) / sigR));
			k += tmp;
			resY[i] += tmp * y[j];
		} // for j
		resY[i] /= k;
	} // for i
	return resY;
} // bilateralFilter
