#ifndef __SYMMETRIES_H
#define __SYMMETRIES_H

#include "../Backend/Symmetry.h"

//forward declare:
class JsonWriter;

	class GridSymmetry : public Symmetry {
	protected:
		// For each dimension
		// Angles in degrees
		Radian alpha, beta, gamma; // Angle
		FACC da, db, dc; // Distance
		FACC Na, Nb, Nc; // Average repetitions

		Vector3d av, bv, cv;

	public:
		GridSymmetry() : alpha(0.f), beta(0.f), gamma(0.f) {}
		virtual ~GridSymmetry() {}

		virtual std::string Hash() const;
		virtual std::string GetName() const;

		// Organize parameters from the parameter vector into the matrix and vector defined earlier.
		virtual void OrganizeParameters(const VectorXd& p, int nLayers);

		virtual void PreCalculate(VectorXd& p, int nLayers);

		virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz);

		virtual bool Populate(const VectorXd& p, int nLayers);
		virtual unsigned int GetNumSubLocations();
		virtual LocationRotation GetSubLocation(int index);

		// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);
		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);


		// If index is out of bounds (bounds: [0,infinity)), returns N/A.
		// Usually returns "Solvent" or "Layer #"
		virtual std::string GetLayerName(int layer);
		static std::string GetLayerNameStatic(int layer);

		// Returns the requested extra parameter's specifications, when index
		// is out of bounds, returns a parameter with name N/A.
		virtual ExtraParam GetExtraParameter(int index);
		static ExtraParam GetExtraParameterStatic(int index);

		// Returns the default value of a layer parameter by its index and layer
		// (spanning from 0 to NumParamLayers)
		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		// Returns false iff the layer and layer parameter index are not 
		// applicable
		static bool IsParamApplicable(int layer, int lpindex);

		// Returns the title of the displayed parameter
		static std::string GetDisplayParamName(int index);

		// Returns the value of the displayed parameter, according to the
		// current parameters of the model
		static double GetDisplayParamValue(int index, const paramStruct *p);

		virtual void GetHeader(unsigned int depth, std::string &header);
		virtual void GetHeader(unsigned int depth, JsonWriter &writer);


		virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL, void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);

		virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi);
		virtual std::complex<FACC> getAmplitudeAtPoint(FACC q, FACC theta, FACC phi);

	};


	// Formerly known as DOLs, only now with (mutable) parameters...
	class ManualSymmetry : public Symmetry, public IGPUGridCalculable {
	protected:
		std::vector<Matrix3d> rot;
		std::vector<Vector3d> trans, rotVars;
		std::map<Eigen::Matrix3d, std::vector<Eigen::Vector3d>, matrix3d_less> translationsPerOrientation;

	public:
		ManualSymmetry() : Symmetry() {}
		virtual ~ManualSymmetry() {}

		virtual std::string Hash() const;
		virtual std::string GetName() const;

		// Called before each series of q-calculations
		// If implemented and dependent on parameters, should set status to UNINITIALIZED
		virtual void PreCalculate(VectorXd& p, int nLayers);

		virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL,
			void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);

		virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz);

		virtual bool Populate(const VectorXd& p, int nLayers);
		virtual unsigned int GetNumSubLocations();
		virtual LocationRotation GetSubLocation(int index);

		// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
		virtual std::string GetLayerParamName(int index, EDPFunction *edpfunc);
		static std::string GetLayerParamNameStatic(int index, EDPFunction *edpfunc);

		// If index is out of bounds (bounds: [0,infinity)), returns N/A.
		// Usually returns "Solvent" or "Layer #"
		virtual std::string GetLayerName(int layer);
		static std::string GetLayerNameStatic(int layer);

		// Returns the default value of a layer parameter by its index and layer
		// (spanning from 0 to NumParamLayers)
		static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc);

		// Returns the requested extra parameter's specifications, when index
		// is out of bounds, returns a parameter with name N/A.
		virtual ExtraParam GetExtraParameter(int index);
		static ExtraParam GetExtraParameterStatic(int index);


		virtual void GetHeader(unsigned int depth, std::string &header);
		virtual void GetHeader(unsigned int depth, JsonWriter &writer);

		virtual bool CalculateGridGPU(GridWorkspace& workspace);

		virtual bool SetModel(GridWorkspace& workspace);

		virtual bool ImplementedHybridGPU();

		virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi);
		virtual std::complex<FACC> getAmplitudeAtPoint(FACC q, FACC theta, FACC phi);

	};


#endif
