#ifndef __BACKGROUND_H
#define __BACKGROUND_H

#pragma once
#include "Common.h"
#include <string>
#include <vector>
#include <map>
#include "FrontendExported.h"

typedef struct {
	std::vector<double> x;
	std::vector<double> y;
} graphStruct;

// TODO::Baseline
//EXPORTED void AutoBaselineGen(const std::vector<double>& datax,
//							  const std::vector<double>& datay, std::vector<double>& bgy);

inline void LineFunction(double, double, double, double, double *, double *);

#endif // __BACKGROUND_H
