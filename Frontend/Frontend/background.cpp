#include "background.h"

#include <cmath>
#include <cstring>
#include <limits>

using std::vector;


int selOne = -1, selTwo = -1;



double InterpolatePoint(double x0, const std::vector<double>& x, const std::vector<double>& y) {
    for(int i = 0; i < (int)x.size(); i++) {
		if(x0 <= x[i] && i != 0)
			return y[i - 1] + (x0 - x[i - 1]) * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]));
	}
	return 0.0;
}


int vmin(const std::vector<double>& vec) {
	double y_min = vec[0];
	int res = 0;
	for (int i = 1; i < (int)vec.size(); i++) {
		if (y_min > vec[i]) {
			y_min = vec[i];
			res = i;
		}
	}

	return res;
}

inline double vminfrom(vector<double>& v, int from, int *m) {
	if(v.size() == 0)
		return 0.0;
	double val = v.at(0);
	for(unsigned int i = from; i < v.size(); i++) {
		if(v[i] < val) {
			val = v[i];
			if(m)
				*m = i;
		}
	}
	return val;
}

inline double vmax(vector<double>& v, int *m) {
	if(v.size() == 0)
		return 0.0;
	double val = v.at(0);
	for(unsigned int i = 0; i < v.size(); i++) {
		if(v[i] > val) {
			val = v[i];
			if(m)
				*m = i;
		}
	}
	return val;
}

/**
* Returns the slope and intercept (line equation) from given two points
* (x1,y1) and (x2,y2).
*/
inline void LineFunction(double x1, double y1, double x2, double y2, 
						 double *slope, double *intercept) {
	 double x11 = x1, x22 = x2, y11 = y1, y22 = y2;

	 *slope = ((y22 - y11) / (x22 - x11));
	 *intercept = y11 - ((*slope) * x11);
}

void interpolate(const std::vector<double>& x, std::vector<double>& vec, int x1, int x2, double y1, double y2) {
	for(int i = x1; i < x2; i++)
		vec[i] = y1 + (x[i] - x[x1]) * ((y2 - y1) / (x[x2] - x[x1]));
}

// Returns -1 when there is no intersection
int findIntersection(const std::vector<double>& data, const std::vector<double>& bg, int x1, int x2) {
	
	for(int i = x1; i < x2; i++)
		if(bg[i] >= data[i])
			return i;
	/*
	for(int i = x2 - 1; i >= x1; i--)
		if(bg[i] >= data[i])
			return i;
	*/

	return -1;
}

void AutoBaselineGen(const std::vector<double>& datax, const std::vector<double>& datay, std::vector<double>& bgy) {
	// PSEUDOCODE:
	/*
	1. xmax = [find global minimum]
	2. we would like to find the first intersection of the line with increasing slope from xmin to xmax
	3. from intersection to xmax, we interpolate a line
	4. xmax = intersection
	5. do this (steps 3-6) until intersection = 0
	*/

	int minpos = vmin(datay);
	int xmax = minpos, xmin = 0, intersection = xmax;
	const double EPSILON = 1e-4;

	std::vector<double> logy;

	// Use log-scale y
	for(int i = 0; i < (int)datay.size(); i++)
		logy.push_back(log10((datay[minpos] <= 0.0) ? (datay[i] - datay[minpos] + EPSILON) : datay[i]));

	bgy.clear();
	bgy.resize(datax.size(), logy[minpos]);

	double cury = logy[xmax], height = -EPSILON;

	do {
		interpolate(datax, bgy, xmin, xmax, cury + height, bgy[xmax]);

		// While intersection is -1 (no intersection), we should keep going up
		do {
			height += EPSILON;

			interpolate(datax, bgy, xmin, xmax, cury + height, bgy[xmax]);

			// We aim to increase the slope until we find the point of the intersection
			intersection = findIntersection(logy, bgy, xmin, xmax);

			if(xmin >= xmax)
				intersection = 0;

		} while (intersection < 0);

		// Here we undo the last subiteration, so that the baseline won't pass the data
		height -= EPSILON;
		interpolate(datax, bgy, xmin, xmax, cury + height, bgy[xmax]);
		
		xmax = intersection;

	} while (intersection > 0);

	// Un-logscale the background
	for(int i = 0; i < (int)bgy.size(); i++)
		bgy[i] = pow(10, bgy[i]);

	if(datay[minpos] <= 0.0)
		for(int i = 0; i < (int)bgy.size(); i++)
			bgy[i] += datay[minpos] - EPSILON;
}
