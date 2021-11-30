// Copyright Tal Ben-Nun 2012, 2014.

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <direct.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include "Optimization.h"
#include "CeresOpt.h"

#include <Eigen/Eigen>
using namespace Eigen;

#ifdef _WIN32
#include "dirent.h"
#else
#include <dirent.h>
#endif

#define NLP 6//13

// Common numerical factors
#ifndef ln2
#define ln2 0.69314718055995
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define START_Q_POINT 0.1
#define END_Q_POINT 3.0

template<typename T> inline T sq(const T& val) { return val * val; }

// Copied from X+
using std::vector;
using std::ifstream;

void ReadDataFile(const char *filename, vector<double>& x, vector<double>& y)
{
	int i = 0, n = -1;
	bool done = false, started = false;
	ifstream in (filename);

	if(!in) {
		fprintf(stderr, "Error opening file %s for reading\n", filename);
		exit(1);
	}

	while(!in.eof() && !done) {
		i++;
		if(n > 0 && i > n)
			break;

		std::string line;
		size_t pos, end;
		double curx = -1.0, cury = -1.0;
		getline(in, line);

		line = line.substr(0, line.find("#"));

		//Remove initial whitespace
		while(line[0]  == ' ' || line[0] == '\t' || line[0] == '\f')
			line.erase(0, 1);

		//Replaces whitespace with one tab
		for(int cnt = 1; cnt < (int)line.length(); cnt++) {
			if(line[cnt] == ' ' || line[cnt] == ',' || line[cnt] == '\t' || line[cnt] == '\f') {
				while(((int)line.length() > cnt + 1) && (line[cnt + 1] == ' ' || line[cnt + 1] == ',' || line[cnt + 1] == '\t' || line[cnt + 1] == '\f'))
					line.erase(cnt + 1, 1);
				line[cnt] = '\t';
			}
		}

		pos = line.find("\t");
		// Less than 2 words/columns
		if(pos == std::string::npos){
			if (started) done = true;
			continue;
		}
		end = line.find("\t", pos + 1);

		if(end == std::string::npos)
			end = line.length() - pos;
		else
			end = end - pos;

		if(end == 0) {	// There is no second word
			if (started) done = true;
			continue;
		}

		// Check to make sure the two words are doubles
		char *str = new char[line.length()];
		char *ptr;
		strcpy(str, line.substr(0, pos).c_str());
		strtod(str, &ptr);
		if(ptr == str) {
			if (started) done = true;
			delete[] str;
			continue;
		}
		strcpy(str, line.substr(pos + 1, end).c_str());
		strtod(str, &ptr);
		if(ptr == str) {
			if (started) done = true;
			delete[] str;
			continue;
		}
		delete[] str;

		curx = strtod(line.substr(0, pos).c_str(), NULL);
		cury = strtod(line.substr(pos + 1, end).c_str(), NULL);

		if(!started && !(fabs(cury) > 0.0)) continue;
		if(!started) started = true;

		if(curx < START_Q_POINT) // Skip bad data
			continue;
		if(curx > END_Q_POINT) // Skip bad data
			continue;

		x.push_back(curx);
		y.push_back(cury);
	}

	// Removes trailing zeroes
	if(y.empty()) return;
	while(y.back() == 0.0) {
		y.pop_back();
		x.pop_back();
	}
	//sort vectors
	for (unsigned int i = 0; i < x.size(); i++) {
		for(unsigned int j = i + 1; j < x.size(); j++) {
			if( x[j] < x[i]) {
				double a = x[j];
				x[j] = x[i];
				x[i] = a;
				a= y[j];
				y[j] = y[i];
				y[i] = a;
			}
		}
	}
}

void ReadDataFile(const char *filename, VectorXd& x, VectorXd& y)
{
	std::vector<double> sx, sy;
	ReadDataFile(filename, sx, sy);

	x.resize(sx.size());
	y.resize(sy.size());
	memcpy(x.data(), sx.data(), sizeof(double) * sx.size());
	memcpy(y.data(), sy.data(), sizeof(double) * sy.size());
}

MatrixXd ReadDirectoryToMatrix(const char *directory, VectorXd& qData) {
	DIR *dir;
	struct dirent *ent;
	dir = opendir(directory);
	if(!dir) {
		qData = VectorXd::Zero(1);
		return MatrixXd::Zero(1, 1);
	}

	std::set<std::string> frames;

	// Map frame files
	while ((ent = readdir (dir)) != NULL) {
		if(strstr(ent->d_name, ".dat") || strstr(ent->d_name, ".chi")) {
			char filepath[256] = {0};			
			sprintf(filepath, "%s\\%s", directory, ent->d_name);

			frames.insert(filepath);
		}
	}
	closedir(dir);	

	if(frames.size() < 1) {
		qData = VectorXd::Zero(1);
		return MatrixXd::Zero(1, 1);
	}

	// Read first file for q-range
	std::vector<double> x, y;
	ReadDataFile(frames.begin()->c_str(), x, y);

	// Initialize Xray matrix
	MatrixXd xrayMat = MatrixXd::Zero(frames.size(), x.size());

	// Initialize qData
	unsigned int i = 0;
	unsigned int qsize = x.size();
	qData = VectorXd::Zero(x.size());
	for(i = 0; i < x.size(); i++)
		qData[i] = x[i];

	// Read the data	
	i = 0;
	for(std::set<std::string>::iterator iter = frames.begin(); iter != frames.end(); ++iter) {
		x.clear(); y.clear();
		ReadDataFile(iter->c_str(), x, y);

		for(unsigned int j = 0; j < qsize; j++)
			xrayMat(i, j) = y[j];

		i++;
	}

	return xrayMat;
}

void WriteMatrixToFile(const char *filename, const MatrixXd& mat) {
	FILE *fp = fopen(filename, "wb");
	if(!fp)
		return;
	for(int i = 0; i < mat.rows(); i++) {
		for(int j = 0; j < mat.cols(); j++)
			fprintf(fp, "%f\t", mat(i, j));
		fprintf(fp, "\n");
	}

	fclose(fp);
}

void WriteMatrixToFiles(const char *prefix, const VectorXd& xAxis, const MatrixXd& mat) {
	char aaa[256] = {0};	

	for(int i = 0; i < mat.rows(); i++) {
		sprintf(aaa, "%s-%d.tsv", prefix, i);
		FILE *fp = fopen(aaa, "wb");
		if(!fp)
			return;

		for(int j = 0; j < mat.cols(); j++)
			fprintf(fp, "%f\t%f\n", xAxis[j], mat(i, j));

		fclose(fp);
	}
}

// Helper function to set initial values
void SetInitialParams(MatrixXd& paramVecs, VectorXd& lowerBound, VectorXd& upperBound)  {			
	// VALUES

	// Setting initial estimates
#ifdef INITIAL_ESTIMATES
	; // No...
#endif
	//////////////////////////////////////////////////////////////////////////

	// BOUNDS

	// Setting individual upper/lower bounds
/* // Membrane test...
	lowerBound[0] = 0.1;         upperBound[0] = 1.0;      // Membrane Tail Thickness (FF)
	lowerBound[1] = 0.1;         upperBound[1] = 1.0;	   // Membrane Head Thickness (FF) 
 	lowerBound[2] = 210.0;       upperBound[2] = 310.0;	   // Membrane Tail Electron Density (FF) 
 	lowerBound[3] = 400.0;       upperBound[3] = 500.0;    // Membrane Head Electron Density (FF)          
 	lowerBound[4] = 1.4;         upperBound[4] = 1.9;      // Membrane Head distance from origin (FF)      
 	lowerBound[5] = 2.0;         upperBound[5] = 25.0;	   // Number of Membranes in Domain (SF) 
 	lowerBound[6] = 15.0;        upperBound[6] = 65.0;	   // Membrane Repeat Distance (SF) 
 	lowerBound[7] = 1.0;         upperBound[7] = 17.0;	   // Debye-Waller Factor's <u^2> (SF) 
 	lowerBound[8] = 0.01;        upperBound[8] = 0.5;	   // Peak Amplitude (SF) 
	lowerBound[9] = 0.5;         upperBound[9] = 2.0;	   // Peak Addition Factor (SF)
	lowerBound[10] = 0.0000005; upperBound[10] = 0.0001;   // Form Factor Scale (BG) 
	lowerBound[11] = 0.13;      upperBound[11] = 0.2;      // Background Power (BG) 
	lowerBound[12] = 0.001;     upperBound[12] = 0.01;     // Background Amplitude (BG) 
*/
	lowerBound[0] = 0.001;       upperBound[0] = 10.0;     // Solvent Radius
	lowerBound[1] = 0.001;       upperBound[1] = 10.0;	   // Layer 1 Radius
	lowerBound[2] = 210.0;       upperBound[2] = 510.0;	   // Solvent ED
	lowerBound[3] = 200.0;       upperBound[3] = 500.0;    // Layer 1 ED
	lowerBound[4] = 0.1;         upperBound[4] = 10.0;     // Scale
	lowerBound[5] = 0.0;         upperBound[5] = 25.0;	   // Background

	// END of lower and upper bounds

	//////////////////////////////////////////////////////////////////////////
}

#define USE_GAUSSIAN // Use a Gaussian ED profile for Ph.D. model
#define USE_RELSSE

// Parameter macros
#define P_TAILTHICKNESS ff[0]  /* Membrane Tail Thickness (FF) */
#define P_HEADTHICKNESS ff[1]  /* Membrane Head Thickness (FF) */
#define P_TAILED        ff[2]  /* Membrane Tail Electron Density (FF) */
#define P_HEADED        ff[3]  /* Membrane Head Electron Density (FF) */
#define P_HEADZ         ff[4]  /* Membrane Head distance from origin (FF) */

#define P_N             sf[0]  /* Number of Membranes in Domain (SF) */
#define P_H2H           sf[1]  /* Membrane Repeat Distance (SF) */
#define P_DWU           sf[2]  /* Debye-Waller Factor's <u^2> (SF) */
#define P_PEAKAMP       sf[3]  /* Peak Amplitude (SF) */
#define P_PAF           sf[4]  /* Peak Addition Factor (SF) */

#define P_SCALE         bg[0] /* Form Factor Scale (BG) */
#define P_BPOWER        bg[1] /* Background Power (BG) */
#define P_BAMP          bg[2] /* Background Amplitude (BG) */

struct XRayResiduals
{
	XRayResiduals(const double *x, const double* y, int numResiduals)
		: x_(x), y_(y), numResiduals_(numResiduals) {}
	
	static ceres::CostFunction *GetCeresCostFunction(const double *x, const double *y,
		const std::vector<int>& params, int numResiduals)
	{
		ceres::DPlusDynamicNumericDiffCostFunction<XRayResiduals> *res = 
			new ceres::DPlusDynamicNumericDiffCostFunction<XRayResiduals>(new XRayResiduals(x, y, numResiduals));

		for(int k = 0; k < params.size(); k++)
			res->AddParameterBlock(params[k]);
		res->SetNumResiduals(numResiduals);

		return res;
	}

	/// NOTE! residual has to be used as both input and output.
	/// As input, the values should be the calculated intensity of the model.
	/// As output, the values will be (surprise) the residuals.
	//template <typename T> 
	bool operator()(double const* const* p, double* residual) const {
		for(int i = 0; i < numResiduals_; i++) {
			residual[i] -= y_[i];
		}

		return true;
	}

	const double* operator()() const {
		return x_;
	}

private:
	const double *x_;
	const double *y_;
	int numResiduals_;
};

struct XRayResidual {
  XRayResidual(double x, double y)
      : q(x), y_(y) {
		 std::cout << "Avi " << q << std::endl ;
  }

  static ceres::CostFunction *GetCeresCostFunction(double x, double y, const std::vector<int>& params, int numResiduals)
  {
	  // The Stride = 5 comes from the fact that there are actually 5 parameters per separable block (FF, SF, BG)
	  // BG has 3 but it's OK.
/*
	  ceres::DynamicAutoDiffCostFunction<XRayResidual, 5> *res = 
		new ceres::DynamicAutoDiffCostFunction<XRayResidual, 5>(new XRayResidual(x, y));
*/
	  ceres::DynamicNumericDiffCostFunction<XRayResidual> *res = 
		  new ceres::DynamicNumericDiffCostFunction<XRayResidual>(new XRayResidual(x, y));

	  // Set cost function structure
	  for(int k = 0; k < params.size(); k++)
		  res->AddParameterBlock(params[k]);
	  res->SetNumResiduals(numResiduals);

	  return res;
  }

  template <typename T> 
  bool operator()(T const* const* p, T* residual) const {
    T const* ff = p[0];
	T const* sf = p[1];
	T const* bg = p[2];

	T intensity = T(0.0);
						
	const T QMAX = T(4.40241289);

	// Formula: I(q)= A*FFOA*SFOA + Ba*q^-Bp
	const T edSolvent = T(333.0); // Solution is water ==> 333 e/nm^3

	// Computing FFOA	
#ifndef USE_GAUSSIAN
	const T& tTail = P_TAILTHICKNESS;
	const T tHead = P_HEADTHICKNESS + P_TAILTHICKNESS;		
	intensity += 2 * (P_TAILED - edSolvent) * (sin(q * tTail) - sin(q * 0 /*width[0]*/));
	intensity += 2 * (P_HEADED - edSolvent) * (sin(q * tHead) - sin(q * tTail));

	intensity *= intensity;

	intensity *= (2.0 / sq(sq(q)));	
#else
	const T tTail = P_TAILTHICKNESS * 2.0;
	const T& tHead = P_HEADTHICKNESS;		

	intensity += (P_TAILED - edSolvent) * tTail * exp(-q * q * tTail * tTail / (16.0 * ln2));

	intensity += 2.0 * (P_HEADED - edSolvent) * tHead * exp( -(q * q * tHead * tHead) / (16.0 * ln2))
				* cos(P_HEADZ * q);

	intensity = sq(intensity / q) * (PI / (2.0 * ln2));
#endif

	// SFOA
	T sfOA = T(0.0);

	T qMin = T(2.0 * M_PI) / P_H2H;
	T hMax = (QMAX / qMin);

	T debyeWallerFactor = P_PEAKAMP * exp(-sq(q) * P_DWU / 2.0);
	for(T i = T(1); i <= hMax; i = i + T(1)) {
		// (sin nx)/(sin x)
		/*double someRatio = (P_H2H / 2.0) * (q - (2 * M_PI * i / P_H2H));
		double ratioSine = sin(someRatio);
		if(ratioSine == 0.0)
			continue;		

		sfOA += debyeWallerFactor * sq(sin(P_N * someRatio) / ratioSine);*/

		// Gaussian model

		sfOA += sq(P_N) * exp(-sq(P_N * P_H2H * (q - (2.0 * M_PI * i / P_H2H))) / (4.0 * M_PI));
	}
	sfOA *= debyeWallerFactor;
	sfOA += P_PAF;
	
	intensity *= sfOA;

	// Scale and Background
	intensity *= P_SCALE;
	intensity += P_BAMP * pow(q, -P_BPOWER);

#ifdef USE_RELSSE
	if(fabs(y_) > 1e-7)
		residual[0] = ((T(y_) - intensity) / T(y_));
	else
		residual[0] = (T(y_) - intensity);
#else
	residual[0] = T(y_) - intensity;
#endif
    return true;
  }

 private:
  const double q;
  const double y_;
};

int main(int argc, char **argv) {
	if(argc < 2) {
		printf("XrayFit <DIRECTORY>\n");
		return 1;
	}

	google::GLOG_LEVEL = google::FATAL;

	int curTime = (int)time(NULL);

	// Randomize	
	srand(time(NULL));	
	//srand(11234);

	// 1. Read data to matrix
	VectorXd q;
	VectorXd data;
	ReadDataFile(argv[1], q, data);
	if(q.size() > 1) {
		printf("Read successfully. Sequence size: %d, q-range: %d.\n", 
			   data.rows(), q.size());
	} else {
		printf("ERROR: No data found\n");
		return 0;
	}

	// 2. Set initial estimate for bounds and parameters (if requested)
	// Generate parameter and mutability vectors for each dataset
	MatrixXd paramVecs = MatrixXd::Constant(data.rows(), NLP, 0.0);	
	VectorXd lowerBound = VectorXd::Constant(paramVecs.cols(), -std::numeric_limits<double>::infinity());
	VectorXd upperBound = VectorXd::Constant(paramVecs.cols(),  std::numeric_limits<double>::infinity());

	// Set the initial parameter values, mutabilities and lower/upper bounds
	SetInitialParams(paramVecs, lowerBound, upperBound);

// 	std::vector<int> paramsPerBlock (3);
// 	paramsPerBlock[0] = 5; paramsPerBlock[1] = 5; paramsPerBlock[2] = 3;

	std::vector<int> paramsPerBlock (1);
	paramsPerBlock[0] = NLP;

	std::vector<IOptimizationMethod *> optimizers;
/*
	for(int i = 0; i < data.rows(); i++) {
		optimizers.push_back(new CeresOptimizer(XRayResidual::GetCeresCostFunction,
			paramsPerBlock, 1, q, data.row(i), 20, lowerBound, upperBound,
			true, false, false));
	}
*/
	optimizers.push_back(new CeresOptimizer(XRayResiduals::GetCeresCostFunction,
		paramsPerBlock, data.size(), q, data, 20, lowerBound, upperBound,
		true, false, false));

/*
	VectorXd p = VectorXd::Zero(13);
	p[0] = 0.5; p[1] = 0.5; p[2] = 250.0; p[3] = 415.0; p[4] = 1.6; // FF
	p[5] = 10.0; p[6] = 30.0; p[7] = 1.5; p[8] = 0.04; p[9] = 0.5;  // SF
	p[10] = 0.000004; p[11] = 0.12; p[12] = 0.005;				    // BG
*/

	VectorXd p = VectorXd::Zero(NLP);
	p[0] = 0.5;		p[1] = 0.5;
	p[2] = 250.0;	p[3] = 415.0;
	p[4] = 1.6;		p[5] = 1.9;

	IOptimizationMethod *opt = optimizers[0];
	double gof = std::numeric_limits<double>::infinity();
	for(int i = 0; !opt->Convergence()/*i < 10*/; i++)
	{
		VectorXd pnew;
		gof = opt->Iterate(p, pnew);
		p = pnew;
		printf("Iteration %d: GoF = %f\n", i, gof);
	}

	char filename[256] = {0};
	sprintf(filename, "OUTCERES-%d.tsv", curTime);

	FILE *fp = fopen(filename, "wb");
	if(fp)
	{		
		double **params = new double*[3];
		params[0] = new double[5]; params[1] = new double[5]; params[2] = new double[3];
		for(int i = 0; i < 5; i++)
			params[0][i] = p[0 + i];
		for(int i = 0; i < 5; i++)
			params[1][i] = p[5 + i];
		for(int i = 0; i < 3; i++)
			params[2][i] = p[10 + i];

		for(int i = 0; i < q.size(); i++)
		{
			XRayResidual model (q[i], 0.0);			
			double residual = 0.0;
			model(params, &residual);

			fprintf(fp, "%f\t%f\n", q[i], -residual);
		}
		
		delete[] params[0]; delete[] params[1]; delete[] params[2];
		delete[] params;

		fclose(fp);
	}

	printf("FF: ");
	for(int i = 0; i < 5; i++)
		printf("%f ", p[i]);
	printf("\nSF: ");
	for(int i = 5; i < 10; i++)
		printf("%f ", p[i]);
	printf("\nBG: ");
	for(int i = 10; i < 13; i++)
		printf("%f ", p[i]);
	printf("\n");


	return 0;
}
