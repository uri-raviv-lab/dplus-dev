#ifndef __PROLATE_SPHERE_MODEL_H
#define __PROLATE_SPHERE_MODEL_H

#include "Model.h"

#define EXPORTER
#include "ModelContainer.h"

class ProlateSphereModel : public FFModel {
protected:
	VectorXd xx, ww;
	int steps;
	double a, b;
	VectorXd ed;

public:
	ProlateSphereModel(std::string st = "Prolate Sphere", ProfileType edp = NONE, int intSteps = 100);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual ExtraParam GetExtraParameter(int index);

};



#endif
