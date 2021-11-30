#ifndef __MT_DELIX_MODEL_H
#define __MT_DELIX_MODEL_H

#ifndef PI
	#define PI 3.14159265358979323846264338327
#endif

#include "Model.h"

#define EXPORTER
#include "ModelContainer.h"

class MTDelixModel : public FFModel {
protected:
	VectorXd xTheta, wTheta, xPhi, wPhi;
	int thetaSteps, phiSteps, nHelices;
	double P, sphereR, helixR, ed, dw;

public:
	MTDelixModel(std::string st = "Infinite Uniform Delices", ProfileType edp = NONE, int thetaSteps = 100, int phiSteps = 100);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual ExtraParam GetExtraParameter(int index);

	virtual bool IsLayerBased();
};



#endif
