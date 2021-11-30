#ifndef __TRA_MODEL_H
#define __TRA_MODEL_H

#include "Geometry.h"


	// This class should be used when fitting many precalculated models and
	// their relative weights to a signal.
	// LaTeX: I_{tot} = \sum_i {a_i \times I_i}
	class TRAModel : public Geometry {
	protected:
		// Each of the embedded models must have its
		// intensity as part of the Ii matrix. The fitting
		// parameters will scale Ii[i] accordingly.
		std::vector<VectorXd> Ii;

		// The momentum transfer "vector." Each model must
		// be valid within this range
		std::vector<double> Qi;

		// The weights of each element in the model.
		VectorXd *alphai;

		double BG, scale;

		//FFModel *newModel;

		std::vector<FFModel*> models;

		// Holds the number of models incorporated
		int numModels;

		int qindex;

		int FindQ(int iNindex, double q);

	public:

		TRAModel(std::string st = "Find a better name", int extras = 2,
			int nlp = 1, int minlayers = -1, int maxlayers = -1,
			EDProfile edp = EDProfile(NONE));

		virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p);

		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		static std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		static std::string GetLayerName(int layer);

		virtual void AddModel(FFModel *mod);

		virtual void PreCalculate(VectorXd &p, int nLayers);

		virtual void RemoveModel(int index);

		virtual int GetModelIndex(FFModel *mod);

		virtual void CalculateModel(int index, int mnLayers, VectorXd& p);

		~TRAModel();

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd &p);
	};

	// This class is a "dummy" class that can be used to contain a precalculated
	// formfactor read from a file
	class PredefinedModel : public FFModel {
	protected:
		std::vector<double> Q, I;

		int qindex;

		int FindQ(int iNindex, double q);

	public:

		PredefinedModel(std::vector<double> qVector, std::vector<double> IVector, std::string st = "Needs a name", int extras = 2,
			int nlp = 0, int minlayers = -1, int maxlayers = 0,
			EDProfile edp = EDProfile(NONE));

		virtual void PreCalculate(VectorXd &p, int nLayers);

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w = 1.0, double precision = 1E-5, VectorXd* p = NULL);

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd &p);

	};


#endif
