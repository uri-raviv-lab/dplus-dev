#ifndef __HELICAL_MODELS_H
#define __HELICAL_MODELS_H

#include "Geometry.h"

	class HelicalModel : public FFModel {
	protected:
		double rHelix;
		double P, edSolvent;
		VectorXd delta;
		VectorXd deltaED;
		VectorXd rCs;

	public:
		HelicalModel(std::string st = "Helical model - do not use", int extraParams = 5);

		virtual ExtraParam GetExtraParameter(int index);

		static ExtraParam GetExtraParameterStatic(int index);

		static bool IsParamApplicable(int layer, int lpindex);

		virtual bool IsLayerBased();

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerName(int layer);

		static std::string GetLayerNameStatic(int layer);

		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p) = 0;

	};

	class HelixModel : public HelicalModel {
	protected:
		int steps, steps1, subSteps;

		double height;

		VectorXd xIn, wIn, csxIn, xOut, wOut;

		Eigen::ArrayXXf space;
		Eigen::Matrix4Xd voxel_COMs;
		Eigen::ArrayXd voxel_contrast;
		int number_xy_voxels;
		double step_size;
		int number_z_voxels;
		int xy_origin, z_origin;
	public:

		HelixModel(std::string st = "Helix", int integralStepsIn = 1000, int integralStepsOut = 1000, int extras = 5); // We need to add a parameter for handedness when using the amplitude. I'm not sure how to do this...

		virtual std::complex<double> CalculateFF(Vector3d qvec,
			int nLayers, double w, double precision, VectorXd* p);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual bool IsSlow();

		virtual void PreCalculate(VectorXd& p, int nLayers);

		virtual void PreCalculateFF(VectorXd& p, int nLayers);

		virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param,
			int nLayers, int ai);

	protected:
		virtual double Calculate(double q, int nLayers, VectorXd& p);

	};


#endif
