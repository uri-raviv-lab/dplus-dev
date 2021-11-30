#ifndef __SPHERICAL_MODELS_H
#define __SPHERICAL_MODELS_H

#include "../GPU/GPUInterface.h"
#include "Geometry.h"

	// The abtract class that the Uniform and Gaussian electron densities should inherit from
	class SphericalModel : public FFModel {
	protected:
		double edSolvent;
		VectorXd r;		// Radii
		VectorXd ED;	// Electron density

	public:
		SphericalModel(std::string st = "Spherical model - do not use", int nlp = 2, ProfileShape edp = DISCRETE, int exParams = 2);

		static bool IsParamApplicable(int layer, int lpindex);

		virtual ExtraParam GetExtraParameter(int index);

		static ExtraParam GetExtraParameterStatic(int index);

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual std::string GetLayerName(int layer);
	protected:
		virtual double Calculate(double q, int nLayers, Eigen::VectorXd &p) = 0;	// Ensure that this class cannot be used

	};

	class UniformSphereModel : public SphericalModel, public IGPUCalculable,
		public IGPUGridCalculable
	{

	public:
		UniformSphereModel(std::string st = "Uniform Sphere");

		virtual ExtraParam GetExtraParameter(int index);

		static ExtraParam GetExtraParameterStatic(int index);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculate(Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculateFF(Eigen::VectorXd &p, int nLayers);

		virtual std::complex<double> CalculateFF(Eigen::Vector3d qvec, int nLayers, double w, double precision, VectorXd* p);

		virtual bool SetModel(Workspace& workspace);

		virtual bool SetParameters(Workspace& workspace, const double *params, unsigned int numParams);

		virtual bool ComputeOrientation(Workspace& workspace, float3 rotation);

		virtual void CorrectLocationRotation(double& x, double& y, double& z,
			double& alpha, double& beta, double& gamma);

		virtual bool CalculateGridGPU(GridWorkspace& workspace);

		virtual bool SetModel(GridWorkspace& workspace);

		virtual bool ImplementedHybridGPU();

	protected:
		virtual double Calculate(double q, int nLayers, Eigen::VectorXd &p);
	};

	class GaussianSphereModel : public SphericalModel{
	protected:
		VectorXd z0;		// Distance from center of the sphere
		VectorXd xx, ww;	// For the numerical integration
		int steps;

	public:
		GaussianSphereModel(std::string st = "Gaussian Sphere", ProfileShape edp = GAUSSIAN);

		static bool IsParamApplicable(int layer, int lpindex);

		virtual std::string GetLayerName(int layer);

		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);

		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		virtual void OrganizeParameters(const Eigen::VectorXd &p, int nLayers);

		virtual void PreCalculate(Eigen::VectorXd &p, int nLayers);

		virtual std::complex<double> CalculateFF(Eigen::Vector3d qvec, int nLayers, double w, double precision, VectorXd* p);

	protected:
		virtual double Calculate(double q, int nLayers, Eigen::VectorXd &p);

	};

#endif
