#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include "Eigen/Core"
#include "Common.h"

// Helper function to set up 1D quadrature
EXPORTED_BE void SetupIntegral(Eigen::VectorXd& x, Eigen::VectorXd& w, 
				   double s, double e, int steps);
void SetupIntegral(Eigen::VectorXf& x, Eigen::VectorXf& w, 
				   float s, float e, int steps);

typedef double (*ptF)(double phi, double theta);

typedef double (*QuadFunc)(ptF f, int res, double sx, double ex, double sy,
						   double ey);

#define DEF_QUAD_RES 200

extern QuadFunc Quadrature;
extern int defaultQuadRes;

EXPORTED_BE void ClassifyQuadratureMethod(QuadratureMethod method);

EXPORTED_BE void SetQuadResolution(int res);

double MonteCarlo2D(ptF func, int resolution, double sx, double ex,
				    double sy, double ey);

double SimpsonRule2D(ptF func, int resolution, double sx, double ex, 
					 double sy, double ey);

double GaussLegendre2D(ptF func, int resolution, double sx, double ex,
				       double sy, double ey);


#endif
