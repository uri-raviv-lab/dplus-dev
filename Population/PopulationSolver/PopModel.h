#ifndef __POP_MODEL_H
#define __POP_MODEL_H

#pragma once
#include "ParserSolver.h"
#include "annoying.h"

class PopModel {
protected:
	// Contains all the curves and x values referred to in the expression
	ArrayXXd		curves, xx;
	ArrayXd			*coeffs;
	// Will be filled with the final x data
	ArrayXd			x;
	ArrayXd			y;
	// The curve to which the curves and coefficients are to be fit
	ArrayXd			data;
	ParserSolver	*ps;

	// Parser variables
	vector<string> errs;
	string expr;

public:

	PopModel(const ArrayXXd &dataX, const ArrayXXd &dataY, string express, ArrayXd &coefficients);
	~PopModel();

	virtual bool IsLayerBased() {return false;}

	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	virtual void PreCalculate(VectorXd& p, int nLayers);
	void PreCalculate(ArrayXd& p, int nLayers);

protected:
	// Calculate the model's intensity for a given q
	virtual double Calculate(double q, int nLayers, VectorXd& p = VectorXd());

	inline VectorXd derF(const std::vector<double>& x, VectorXd& p, 
						 int nLayers, int ai, double h, double m);


public:

	// Calculates an entire vector. Default is in parallel using OpenMP,
	// or a GPU if possible
	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p = VectorXd());

	void GetModXY(ArrayXd& x, ArrayXd& y);

	void ChangeCoefficients(ArrayXd& coefs);

	ArrayXd CalculateExp();

	ArrayXd CalculateExp(ArrayXd& coefs);

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai);
};

#endif
