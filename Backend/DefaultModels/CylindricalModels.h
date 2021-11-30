#ifndef __CYLINDRICAL_MODELS_H
#define __CYLINDRICAL_MODELS_H

#include "Geometry.h"

	class CylindricalModel : public FFModel {
	protected:
		int steps;

		double edSolvent;
		VectorXd ed;	// The electron density
		VectorXd t;		// The width of the layers
		double H;		// Half the height

		VectorXd xx, ww;
	public:
		CylindricalModel(int integralSteps = 1000, std::string str = "Abstract Cylindrical Model", ProfileShape eds = DISCRETE, int nlp = 2, int minlayers = 2, int maxlayers = -1, int extras = 2);

		virtual ExtraParam GetExtraParameter(int index);

		static ExtraParam GetExtraParameterStatic(int index);

		static std::string GetLayerNameStatic(int layer);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd& p) = 0;

	};


	class UniformHCModel : public CylindricalModel {
	public:
		UniformHCModel(int integralSteps = 1000, std::string str = "Uniform Hollow Cylinder", ProfileShape eds = DISCRETE, int nlp = 2, int minlayers = 2, int maxlayers = -1, int extras = 3);

		virtual std::string GetLayerName(int layer);

		static std::string GetLayerNameStatic(int layer);

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);


		virtual void PreCalculate(VectorXd& p, int nLayers);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculateFF(VectorXd& p, int nLayers);

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p);
	protected:
		virtual double Calculate(double q, int nLayers, VectorXd& p);

	};

	class GaussianHCModel : public CylindricalModel {
	protected:
		VectorXd r;	// The center of the Gaussian relative to the center of the cylinder
		int steps1, nonzero;
		MatrixXd xxR, wwR;

	public:
		GaussianHCModel(int heightSteps = 1000, int radiiSteps = 500, std::string name = "Gaussian Hollow Cylinder",
			int extras = 3, int nlp = 3, int minlayers = 2, int maxlayers = -1, EDProfile edp = EDProfile());

		virtual bool IsSlow();

		virtual std::string GetLayerName(int layer);

		static std::string GetLayerNameStatic(int layer);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculate(VectorXd& p, int nLayers);

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		static bool IsParamApplicable(int layer, int lpindex);

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p);

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd& p);
	};

#endif
