#pragma once 

#ifndef __CALCULATION_EXT_H__
#define __CALCULATION_EXT_H__

#include "Eigen/Core"
#include "Eigen/LU"
#include "Eigen/SVD"
using namespace Eigen;

#include "..\PopulationSolver\ParserSolver.h"

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

#ifndef __CONSSTRUCT
#define __CONSSTRUCT
typedef struct ConsStruct {
	ArrayXi index, link;
	ArrayXd num;

	ConsStruct(int m) : index(ArrayXi::Constant(m, -1)), link(ArrayXi::Constant(m, -1)), num(ArrayXd::Zero(m)) {}
	ConsStruct()  {}
} cons;
#endif

EXPORTED int GetFitIterations();
EXPORTED bool isLogFitting();
EXPORTED void SetFitIterations(int value);
EXPORTED void SetLogFitting(bool value);

EXPORTED bool FitCoeffs(const ArrayXd& datay, const ArrayXd& datax, const std::string expression,
						const ArrayXXd& curvesX, const ArrayXXd& curvesY, ArrayXd& p, const ArrayXi& pmut,
						cons *pMin, cons *pMax, ArrayXd& paramErrors, ArrayXd& modelErrors);


#endif
