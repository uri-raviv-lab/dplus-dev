#include "EDProfile.h"
#include "Geometry.h"
#include "mathfuncs.h" // For Gaussian functions

// Uses the difference between the midpoint and Simpson's rules to
// determine the subdivision scheme of a function to discrete steps
// Also returns the integral as a side-effect
// Uses new[] for steps, if not null
double EDPFunction::Subdivide(double a, double b, 
							  std::map<double, double>& steps,
							  double eps, int maxDepth) {
	double midpoint, simpson;
	double fa, fb, fc;
	double c = (a + b) / 2.0;

	if(steps.find(a) == steps.end())
		fa = Evaluate(a); 
	else
		fa = steps[a];

	if(steps.find(b) == steps.end())
		fb = Evaluate(b);
	else
		fb = steps[b];

	if(steps.find(c) == steps.end())
		fc = steps[c] = Evaluate(c);
	else
		fc = steps[c];

	// Compute both methods
	midpoint = (b - a) * fc;
	simpson = ((b - a) / 6.0) * (fa + (4.0 * fc) + fb);

	// If tree is too deep or we have reached our epsilon goal, return
	if(maxDepth <= 0 || fabs(midpoint - simpson) < eps)
		return simpson;

	// Recursion
	return Subdivide(a, c, steps, eps / 2.0, maxDepth - 1) +
		   Subdivide(c, b, steps, eps / 2.0, maxDepth - 1);
}

MatrixXd EDPFunction::MakeSteps(double qmin, double qmax, double avgqdiff,
								double& area) {
	int nlp = (int)_params.cols();
	VectorXd r = _params.col(0), ed = _params.col(1);

	int layers = (int)r.size();

	area = 0.0;

	if(layers < 2)
		return MatrixXd::Zero(0, 0);
		
	// Use smart subdivision to obtain the steps
	std::map<double, double> steps;
	

	// Use adaptive subdivision
	if(_res < 0) {
		// TODO::EDP Change _res to tolerance (epsilon)
		double epsilon = pow(10.0, _res);		
		area = Subdivide(GetLowerLimit(), GetUpperLimit(), steps, epsilon, 500);
	} else if(_res > 0) { // Use uniform discrete steps per layer

		double lower = GetLowerLimit(), upper = GetUpperLimit();
		int pts = _res * layers;
		double segmentSize = ((upper - lower) / (double)pts);		
		
		for(int i = 0; i < pts; i++) {
			double startlocation = lower + (i * segmentSize);
			double location = lower + ((i + 1) * segmentSize);
			steps[location] = Evaluate(location);

			area += (location - startlocation) * steps[location];
		}
	} else
		return MatrixXd::Zero(0, 0);

	MatrixXd result = MatrixXd::Zero(steps.size() + 1, nlp);

	// Manually add the solvent
	result(0, 0) = 0.0;
	result(0, 1) = ed[0];

	double last = GetLowerLimit();
	int points = (int)result.rows();
	int i = 1;
	for(std::map<double, double>::iterator iter = steps.begin(); 
		iter != steps.end(); ++iter) {
		result(i, 0) = iter->first - last;
		result(i, 1) = iter->second;
		last = iter->first;

		// TODO::EDP How are we going to identify layers
		for(int k = 2; k < nlp; k++)
			result(i, k) = _params(0, k);

		i++;
	}

	// Small optimization: Eliminate equal steps
	int newsteps = 0;
	MatrixXd finalResults = MatrixXd::Zero(points, nlp);
	finalResults.row(0) = result.row(0);

	for(int i = 1; i < points; i++) {
		// Similar layers, increment width
		if(fabs(result(i, 1) - result(i - 1, 1)) <= 1.0e-6) {
			finalResults(newsteps, 0) += result(i, 0);
		} else { // Different layers, create a new one
			newsteps++;
			finalResults.row(newsteps) = result.row(i);
		}
	}
	result = finalResults.block(0, 0, newsteps + 1, nlp);
	// END OF OPTIMIZATION

	//MessageMatrix(result);

	return result;
}


VectorXd EDPFunction::ComputeParamVector(Geometry *model, 
										 const VectorXd& oldParams,
										 const std::vector<double>& x,
										 int nLayers,
										 int& guessLayers) {
	int nlp = model->GetNumLayerParams(), 
		ep = model->GetNumExtraParams();

	_params = MatrixXd::Zero(nLayers, nlp);

	// Update the ED profile function
	for(int i = 0; i < nlp; i++)
		for(int j = 0; j < nLayers; j++)
			_params(j, i) = oldParams[i * nLayers + j];

	// Create the equivalent step function
	double areaUnused = 0.0;
	MatrixXd newp = MakeSteps(x[0], x[x.size() - 1], x[1] - x[0], areaUnused);


	// Under the assumption that the first parameter is the radius/thickness
	// and the second is the electron density profile
	guessLayers = (int)newp.rows();

	// Creating a new parameter vector
	VectorXd rnewp = VectorXd::Zero(guessLayers * nlp + ep);

	// Assuming there are two layer parameters (no more than thickness and
	// electron density)
	for(int i = 0; i < nlp; i++)
		for(int j = 0; j < guessLayers; j++)
			rnewp[i * guessLayers + j] = newp(j, i);

	// Copy the extra parameters
	for(int i = 0; i < ep; i++)
		rnewp[guessLayers * nlp + i] = oldParams[nLayers * nlp + i];

	// Commit the new, temporary parameter vector
	return rnewp;
}

double EDPFunction::GetLowerLimit() {
	return 0.0;
}

double EDPFunction::GetUpperLimit() {
	double res = 0.0;
	int nLayers = (int)_params.col(0).size();

	// Return the largest radius
	for(int i = 0; i < nLayers; i++)
		res += _params(i, 0);

	return res;
}

double GaussianED::Evaluate(double r) {
	int nlp = (int)_params.cols();
	VectorXd _r = _params.col(0), _ed = _params.col(1), 
			 _z = _params.col(nlp - 1);

	// Start with the solvent
	double res = _ed[0];

	int layers = (int)_r.size();

	// Add the contribution of all layers to this point
	for(int i = 0; i < layers; i++)		
		res += DenormGaussianFW(_r[i], _z[i], _ed[i] - _ed[0], 0.0, r);

	return res;
}

std::string GaussianED::GetEDParamName(int index) {
	switch(index) {
		default:
			return EDPFunction::GetEDParamName(index);
		case 0:
			return "ED Center";
	}
}

double GaussianED::GetUpperLimit() {
	int nlp = (int)_params.cols();
	int layers = (int)_params.rows();

	// Find the limits of the model: [max(z + 4*r)]
	VectorXd r = _params.col(0), z = _params.col(nlp - 1);

	double limit = z[1] + (4.0 * r[1]), tmp = 0.0;
	for(int i = 2; i < layers; i++) {
		tmp = z[i] + (4.0 * r[i]);
		if(tmp > limit)
			limit = tmp;
	}

	return limit;
}

double phi(double r) {
	return (tanh(r) + 1.0) / 2.0;
}

double phi(double r, double alpha, double center) {
	return (tanh(alpha * (r - center)) + 1.0) / 2.0;
}

EDPFunction *ProfileFromShape(ProfileShape shape, const MatrixXd& params) {
	switch(shape) {
		default:
		case DISCRETE:
			return NULL;

		case GAUSSIAN:
			return new GaussianED(params);

		case TANH:
			return new TanhED(params);
	}
}

double TanhED::Evaluate(double r) {
	int nlp = (int)_params.cols();
	VectorXd _r = _params.col(0), _ed = _params.col(1), 
		_s = _params.col(nlp - 1);

	// Start with the solvent
	double res = _ed[0];

	int layers = (int)_r.size();

	// Incremental radii
	double rad = _r[0];

	// Add the contribution of all layers to this point
	for(int i = 1; i < layers; i++) {
		res += (_ed[i] - _ed[0]) * (phi(r, _s[i - 1], rad) - phi(r, _s[i], rad + _r[i]));
		rad += _r[i];
	}

	return res;
}

std::string TanhED::GetEDParamName(int index) {
	switch(index) {
		default:
			return EDPFunction::GetEDParamName(index);
		case 0:
			return "Slope";
	}
}

double TanhED::GetUpperLimit() {
	int nlp = (int)_params.cols();
	int layers = (int)_params.rows();

	// Find the limits of the model: [max(r + 10/alpha)]
	VectorXd r = _params.col(0), s = _params.col(nlp - 1);
	
	double rad = 0.0;
	double limit = 0.2;
	for(int i = 0; i < layers; i++) {
		if(r[i] <= 0.0 || s[i] < 1.0e-6)
			continue;

		rad += r[i];
		limit = std::max(limit, rad + (10.0 / s[i]));
	}

	// DEBUG
	/*
	{
		AllocConsole();
		char a[256] = {0};
		sprintf(a, "Upper limit: %f\n", limit);
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), a, 256, NULL, NULL);		
	}
	*/

	return limit;
}

double StepED::Evaluate(double r) {
	VectorXd _r = _params.col(0), _ed = _params.col(1);

	int layers = (int)_r.size();

	// Incrementally find the current step we're in
	double cur = 0.0;
	for(int i = 0; i < layers; i++) {
		cur += _r[i];
		if(cur >= r)
			return _ed[i];		
	}
	return _ed[0];
}
