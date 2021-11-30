#ifndef __MT_DELIX_MODEL_H
#define __MT_DELIX_MODEL_H

#ifndef PI
	#define PI 3.14159265358979323846264338327
#endif

#include "Model.h"

#define EXPORTER
#include "ModelContainer.h"

#ifndef NINDICES
	#define NINDICES 24
#endif

class StackRings : public FFModel {
protected:
	Eigen::ArrayXd rtx;
	Eigen::VectorXd pars, xx, ww;
	double lz, d, T, ED, Ri, Ro, scale, bg;


public:
	StackRings(std::string st = "Stack of Rings", ProfileType edp = NONE, int steps = 500);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual std::string GetLayerName(int layer);

	virtual ExtraParam GetExtraParameter(int index);

	virtual bool IsLayerBased();
};

class FiniteHelixModel : public FFModel {
protected:
	Eigen::ArrayXd rtx, cs, sn;
	Eigen::VectorXd xx, ww;
	double R, P, Rg, N, ED, FFScale, FFBg;
	double rr2;

public:
	FiniteHelixModel(std::string st = "Finite Helix", ProfileType edp = NONE, int xSteps = 500);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual std::string GetLayerName(int layer);

	virtual ExtraParam GetExtraParameter(int index);

	virtual bool IsLayerBased();
};

class HelixWSF : public FFModel {
protected:
	Eigen::ArrayXd rtx, cs, sn;
	Eigen::ArrayXXd  A, lambda;
	Eigen::VectorXd xx, xPhi, ww, wPhi;
	double R, P, Rg, N, ED, FFScale, FFBg, SFScale, SFBg, a;
	int hMax, kMax;
	double rr2;

	double gHex(int h, int k);


public:
	HelixWSF(std::string st = "Helix w Lor Hex SF", ProfileType edp = NONE, int xSteps = 500, int phiSteps = 500);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual std::string GetLayerName(int layer);

	virtual ExtraParam GetExtraParameter(int index);

	virtual bool IsLayerBased();
};


class HCwLorSFModel : public FFModel {
protected:
	VectorXd xx, ww, phi, wphi;	// For numerical integral
	Eigen::ArrayXXd A, lambda;
	double a, ED, lf, debyeWaller, ro, ri, H, FFScale, FFBg, SFScale, SFBg;
	int kMax, hMax;
	
public:
	HCwLorSFModel(std::string st = "Uniform Hollow Cylinder with Hexagonal Structure Factor (Lorentzian)", ProfileType edp = NONE, int steps = 171);

	void PreCalculate(VectorXd& p, int nLayers);
	void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);
	virtual double Calculate(double q, int nLayers, VectorXd& p);
	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p);
	
	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	virtual double GetDefaultParamValue(int paramIndex, int layer);

	virtual std::string GetLayerParamName(int index);

	virtual std::string GetLayerName(int layer);

	virtual ExtraParam GetExtraParameter(int index);

	virtual bool IsLayerBased();

protected:
	double gHex(int h, int k);
	
	double hexSF2D(double q, int nLayers, VectorXd& p);
};
#endif
