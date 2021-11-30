#ifndef __BGFITTING_H
#define __BGFITTING_H

#include <vector>
using std::vector;

#include "Common.h"

EXPORTED_BE bool FitBackground(const vector<double> bgx, const vector<double> bgy, 
						    vector<double>& resy, const vector<double>& signaly, const vector<bool>& mask, bgStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors);
EXPORTED_BE bool FitBackgroundU(const vector<double> bgx, const vector<double> bgy, 
						     vector<double>& resy, const vector<double>& signaly, const vector<bool>& mask, bgStruct *p, std::vector<double>& paramErrors, std::vector<double>& modelErrors, plotFunc GraphModify, 
						     int *pStop, progressFunc ProgressReport);

EXPORTED_BE bool GenerateBackground(const std::vector<double> bgx, std::vector<double>& genY,
				 				 bgStruct *p);
EXPORTED_BE bool GenerateBackgroundU(const std::vector<double> x, std::vector<double>& genY, 
								  const vector<double>& signaly, bgStruct *p, plotFunc GraphModify, int *pStop,
								  progressFunc ProgressReport);

#endif
