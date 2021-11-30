#pragma once
#include <string>
#include <vector>
#include "Eigen/Core" // For ArrayXd
#include "annoying.h"

#undef EXPORTED
	#ifdef _WIN32
		#ifdef EXPORTER
			#define EXPORTED __declspec(dllexport)
		#else
			#define EXPORTED __declspec(dllimport)
		#endif
	#else
		#define EXPORTED extern "C"
#endif


using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Array;
	
using namespace std;

enum OPERATION {ADD, SUBTRACT, MULTIPLY, DIVIDE};

/********************************************************************************
* This class parses simple mathematical expressions (including entire curves)	*
* and calculates the relevant curve using Eigen Arrays. The data matrices		*
* should have exactly 26 rows corresponding to 26 curves that each have a		*
* capital letter representing them (and may appear in the parsed expression).	*
* The same goes for the Array of coefficients (lower case).					*
********************************************************************************/
class EXPORTED ParserSolver {

protected:

	/* Holds the cropped y values */
	ArrayXXd	data;
	// Holds the x values of the rawData
	ArrayXXd	xdata;
	// Holds the y values that are passed
	ArrayXXd	rawData;
	// Holds the coeffiecients
	ArrayXd		coefs;
	// Holds the cropped x values
	ArrayXd		finalX;

	std::string expressionE;
	std::string dExpressionE;
	std::vector<std::string>* error;
	
	string simplifyExp(string exp);

	string multiplyExp(string l, string r);

	void breakUpString(const string str, string &r, string &rr, string &rest);

	ArrayXd CalculateExpression(std::string expres);
	ArrayXd CalculateExpression(std::string left, std::string right, OPERATION op);

	bool isOperation(char c);
	OPERATION op(char o);

	ArrayXd opVector(ArrayXd& left, OPERATION op, ArrayXd& right);

	bool cropData();
	double InterpolatePoint(double x0, const ArrayXd& x, const ArrayXd& y);

public:
	ArrayXd CalculateExpression();
	ArrayXd EvaluateExpression(ArrayXd &newX);
	ArrayXd EvaluateExpression(ArrayXd &newX, std::string expres);
	void ReplaceCoefficients(ArrayXd &coefficients);

	ArrayXd InterpolateCurve(ArrayXd xIn, ArrayXd yIn);

	ArrayXXd GetMatrix();

	bool CanExactDerive();

	ArrayXd Derive(char ch);
	
	ParserSolver(std::vector<std::string>& error, std::string expres, ArrayXXd& dataX, ArrayXXd& dataY, ArrayXd& coefficients);

	~ParserSolver();
};	// class
