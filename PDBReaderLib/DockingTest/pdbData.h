#pragma once
#include <vector>
#include <string>
#include "Eigen\Core"

using std::vector;
using std::string;

using Eigen::MatrixXd;


enum PDB_ERRS {
	OK = 0,

	NO_FILE,

	GENERAL_ERROR
};

class pdbData
{
protected:
	vector<double> x, y, z, temperatureFactor;
	vector<char[6]> atom;
	string fn;

	MatrixXd trfm;
public:
	pdbData(void);
	~pdbData(void);
	double getAmplitude(double q, double theta, double phi);
	double getIntensity(double q, double theta, double phi);
	 
	PDB_ERRS readFromPDB(string filename);
	PDB_ERRS fTransform();
};
