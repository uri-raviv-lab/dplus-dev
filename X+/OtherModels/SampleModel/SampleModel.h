#ifndef __SAMPLE_MODEL_H
#define __SAMPLE_MODEL_H

#include "Geometry.h"

#define EXPORTER
#include "ModelContainer.h"
#include "RendererContainer.h"

class SampleModel : public FFModel {
public:
	SampleModel();

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	static ExtraParam GetExtraParameter(int index);

	virtual int GetNumDisplayParams();
	static std::string GetDisplayParamName(int index);
	static double GetDisplayParamValue(int index, const paramStruct *p);

protected:
	virtual double Calculate(double q, int nLayers, VectorXd& p);
};

class SquareWaveModel : public FFModel {
public:
	SquareWaveModel();

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
	static std::string GetLayerParamName(int index);

	// If index is out of bounds (bounds: [0,infinity)), returns N/A.
	// Usually returns "Solvent" or "Layer #"
	static std::string GetLayerName(int layer);

	virtual void PreCalculate(VectorXd &p, int nLayers);

	static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

protected:

	virtual double Calculate(double q, int nLayers, VectorXd& p);
};

class SineWaveModel : public FFModel {
public:
	SineWaveModel();

	virtual std::complex<double> CalculateFF(Vector3d qvec, 
		int nLayers, double w, double precision, VectorXd& p = VectorXd()  );

	// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
	static std::string GetLayerParamName(int index);

	// If index is out of bounds (bounds: [0,infinity)), returns N/A.
	// Usually returns "Solvent" or "Layer #"
	static std::string GetLayerName(int layer);

	virtual void PreCalculate(VectorXd &p, int nLayers);

	static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

protected:

	virtual double Calculate(double q, int nLayers, VectorXd& p);
};


#endif
