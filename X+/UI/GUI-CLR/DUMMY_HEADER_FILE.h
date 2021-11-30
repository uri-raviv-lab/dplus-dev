#ifndef __DUMMY_HEADER_FILE_h
#define __DUMMY_HEADER_FILE_h
#undef min
#undef max

#pragma once
#include <string>
#include <vector>
#include "Eigen/Core"
//#include "Common.h"
#include "FrontendExported.h"

#define DEFAULT_EDRES 157	// From EDProfile.h
#define DEFAULT_EDEPS 4

using std::vector;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

/****************************************************************************/
/* THIS IS A TEMPORARY FILE THAT IS TO BE USED ONLY FOR DEFINING STRUCTS	*/
/* OR METHODS THAT HAVE YET TO BE IMPLEMENTED IN THE FRONTEND OR BACKEND.	*/
/*																			*/
/* REMEMBER TO IMPLEMENT ITEMS BEFORE DELETEING THEM!!!!!!!!				*/
/****************************************************************************/

typedef struct {
	std::vector<double> x, y, tmpY;
} graphTable;

typedef struct ParamCons {
	double value, min, max;
	char mut;
	int minInd, maxInd, linkInd;

	ParamCons(double val = 0.0) : value(val), min(0.0), max(0.0), mut('N'),
		minInd(-1), maxInd(-1), linkInd(-1)  {}
} genParameter;

typedef struct {
	std::vector<genParameter> amp, fwhm, center, cailleSigma, cailleNDiffused;
} peakStruct;


typedef struct {
	std::vector<BGFuncType> type;
	std::vector<double> base, decay, center;
	std::vector<char> baseMutable, decayMutable, centerMutable;
	std::vector<double> basemin, basemax, decmin, decmax, centermin, centermax;
} bgStruct;

typedef struct {
	double a,b,c,gamma,alpha,beta, amin, bmin, cmin, gammamin,alphamin, betamin, 
		amax, bmax, cmax, gammamax,alphamax, betamax;

	char aM,bM,cM,gammaM,alphaM,betaM;

	double um, ummin, ummax;

	char umM;

	double qmax;
} phaseStruct;

typedef struct ConsStruct {
	VectorXi index, link;
	VectorXd num;

	ConsStruct(int m) : index(VectorXi::Constant(m, -1)), link(VectorXi::Constant(m, -1)), num(VectorXd::Zero(m)) {}
	ConsStruct() {}  
} cons;

double GetResolution();

void SetMinimumSig(double eefd);

bool hasGPUBackend();
bool isGPUBackend();
void SetGPUBackend(bool asdrg);

int vmin(const std::vector<double>& vec);

inline double vminfrom(vector<double>& v, int from, int *m);
inline double vmax(vector<double>& v, int *m);

double InterpolatePoint(double x0, const std::vector<double>& x, const std::vector<double>& y);
void ImportBackground(const wchar_t *filename, const wchar_t *datafile,
							   const wchar_t *savename, bool bFactor);
double GetMinimumSig();
void ClassifyQuadratureMethod(QuadratureMethod method) ;

void SetPDFunc(PeakType shape);
void GenerateBGLinesandFormFactor(const wchar_t *workspace, 
								  const wchar_t *datafile,
								  std::vector <double>& bglx,
								  std::vector <double>& bgly,
								  std::vector <double>& ffy, bool ang);
int GetPeakType();

int phaseDimensions(PhaseType phase);
void SetPeakType(PeakType sgh) ;

//documentation :: 
// q is the peak index we are dealing with
// &a are the params, ma is the phase and nd the dimention
void BuildAMatrixForPhases(VectorXd  &a, MatrixXd &g, PhaseType phase);
// Creating matrix of indices
MatrixXi GenerateIndicesMatrix(int dim, int length);
void uniqueIndices(const VectorXd& G_norm, std::vector<double>& result, std::vector<std::string> &locs);
std::vector <double> GenPhases (PhaseType phase , phaseStruct *p,
								std::vector<std::string> &locs);


#endif // __DUMMY_HEADER_FILE_h
