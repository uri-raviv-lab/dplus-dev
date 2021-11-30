#ifndef __SLAB_MODELS_H
#define __SLAB_MODELS_H

#include "Geometry.h"


	class SlabModel : public FFModel {
	protected:
		double edSolvent;
		VectorXd ED;
		VectorXd width, centers;
		double xDomain, yDomain;	// Parameters for finite sized

	public:
		SlabModel(std::string st = "Slab model - do not use? WIP", EDProfile edp = EDProfile(SYMMETRIC, DISCRETE),
			int nlp = 2, int maxlayers = -1);

		static bool IsParamApplicable(int layer, int lpindex);

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerName(int layer);

		static std::string GetLayerNameStatic(int layer);

		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		virtual ExtraParam GetExtraParameter(int index);

		static ExtraParam GetExtraParameterStatic(int index);

		virtual bool IsSlow() { return false; }

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p) = 0;

	};

	class UniformSlabModel : public SlabModel {
	public:
		UniformSlabModel(std::string st = "Symmetric Uniform Slabs", ProfileType t = SYMMETRIC);


		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculate(VectorXd& p, int nLayers);

		virtual void PreCalculateFF(VectorXd& p, int nLayers);

		//virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, 
		//							int nLayers, int ai);

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd& p);
	};

#endif
