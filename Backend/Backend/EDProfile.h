#ifndef __EDPROFILE_H__
#define __EDPROFILE_H__

// #undef EXPORTED
// #ifdef _WIN32
// #ifdef CALCULATION
// #define EXPORTED __declspec(dllexport)
// #else
// #define EXPORTED __declspec(dllimport)
// #endif
// #else
// #define EXPORTED extern "C"
// #endif

// Avoid "need dll-interface" warnings (the easy way)
#pragma warning (disable: 4251)

#include "Common.h"
#include "Eigen/Core" // For VectorXd
#include <vector> // For std::vector
#include <map> // For std::map

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

// Forward declarations
class EDPFunction;
class Geometry;

#define DEFAULT_EDRES 157
#define DEFAULT_EDEPS 4

// Custom ED profile function
class EDPFunction {
protected:
	MatrixXd _params;

	int _res;


	// Uses the difference between the midpoint and trapezoidal methods to
	// determine the subdivision scheme of a function to discrete steps
	// Also returns the integral as a side-effect
	// Uses new[] for steps, if not null
	double Subdivide(double a, double b, std::map<double, double>& steps,
					 double eps, int maxDepth);


	// This function should receive all the necessary parameters (layer-wise)
	// as input.
	EDPFunction(const MatrixXd& p) : _params(p), _res(DEFAULT_EDRES) {}

public:
	virtual ~EDPFunction() {}

	// Evaluate the ED function in radius/thickness "r". On error, returns -1.0.
	virtual double Evaluate(double r) = 0;

	// Get the lower limit from which to start the step computation
	virtual double GetLowerLimit();
	
	// Get the upper limit until which the step computation is performed
	virtual double GetUpperLimit();

	virtual VectorXd ComputeParamVector(Geometry *model, const VectorXd& oldParams,
										const std::vector<double>& x,
										int nLayers, int& guessLayers);

	// Creates a step-function representation of this ED profile function.
	// Input: q_min, q_max and the average difference between two q points.
	// Returns the matrix of (radius, ED) with the corresponding steps.
	virtual MatrixXd MakeSteps(double qmin, double qmax, double avgqdiff,
							   double& area);

	// Returns the number of EXTRA ED profile-related parameters required to
	// evaluate this ED function
	virtual int GetNumEDParams() { return 0; }

	// Returns the extra ED function parameter name
	virtual std::string GetEDParamName(int index) { return "N/A"; }

	virtual double GetEDParamDefaultValue(int index, int layer) { return 1.0; }

	// Set the number of discrete steps per layer
	void SetResolution(int res) { _res = res; }	
};

// Abstract factory for ED profiles
EXPORTED_BE EDPFunction *ProfileFromShape(ProfileShape shape, const MatrixXd& params);

// Step Electron Density Profile. Mainly used for model implementation sanity
// checking (if this doesn't work, fix the model before trying others)
class StepED : public EDPFunction {
public:
	StepED(const MatrixXd& p) : EDPFunction(p) {}

	virtual ~StepED() {}

	// Evaluate the ED function in radius/thickness "r". On error, returns -1.0.
	virtual double Evaluate(double r);
};

// Gaussian Electron Density Profile
class GaussianED : public EDPFunction {
public:
	GaussianED(const MatrixXd& p) : EDPFunction(p) {}

	virtual ~GaussianED() {}

	virtual int GetNumEDParams() { return 1; }

	virtual std::string GetEDParamName(int index);

	// Get the upper limit until which the step computation is performed
	virtual double GetUpperLimit();

	// Evaluate the ED function in radius/thickness "r". On error, returns -1.0.
	virtual double Evaluate(double r);
};

// Hyperbolic Tangent Electron Density Profile
class TanhED : public EDPFunction {
public:
	TanhED(const MatrixXd& p) : EDPFunction(p) {}

	virtual ~TanhED() {}

	virtual int GetNumEDParams() { return 1; }

	virtual std::string GetEDParamName(int index);

	// Get the upper limit until which the step computation is performed
	virtual double GetUpperLimit();

	// Evaluate the ED function in radius/thickness "r". On error, returns -1.0.
	virtual double Evaluate(double r);
};

#endif
