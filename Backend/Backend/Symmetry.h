#ifndef __SYMMETRY_H
#define __SYMMETRY_H

#include <vector>
#include "Amplitude.h"
#include "PDBAmplitude.h"

class matrix3d_less : public std::binary_function<Eigen::Matrix3d, Eigen::Matrix3d, bool>
{
public:
	matrix3d_less(double arg_ = 0.0001) : threshhold(arg_) {}
	bool operator()(const Eigen::Matrix3d &left, const Eigen::Matrix3d &right) const
	{
		if (left.isApprox(right, threshhold))
			return false;

		for (int i = 0; i < left.size(); i++)
		{
			if (*(left.data() + i) + threshhold < *(right.data() + i))
				return true;
			if (*(left.data() + i) > *(right.data() + i) + threshhold)
				return false;
		}

		// They're equal, but for some reason, isApprox didn't catch it?
		return false;
	}
	double threshhold;
};

class EXPORTED_BE Symmetry : public ISymmetry, public Amplitude {
protected:
	Symmetry() {}

	std::vector<Amplitude *> _amps;
	std::vector<VectorXd> _ampParams;
	std::vector<int> _ampLayers;

public:	
	virtual ~Symmetry() {}


	// Organize parameters from the parameter vector into the matrix and vector defined earlier.
	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	// Called before each series of q-calculations
	// If implemented and dependent on parameters, should set status to UNINITIALIZED
	virtual void PreCalculate(VectorXd& p, int nLayers);

	virtual void calculateGrid(FACC qMax, int sections = 150, progressFunc progFunc = NULL, 
		void *progArgs = NULL, double progMin = 0.0, double progMax = 1.0, int *pStop = NULL);


	virtual std::complex<FACC> calcAmplitude(FACC qx, FACC qy, FACC qz) = 0;

	virtual bool GetHasAnomalousScattering();
	

	virtual int GetNumSubAmplitudes();
	virtual Amplitude *GetSubAmplitude(int index);
	virtual void SetSubAmplitude(int index, Amplitude *subAmp);
	virtual void AddSubAmplitude(Amplitude *subAmp);
	virtual void RemoveSubAmplitude(int index);
	virtual void ClearSubAmplitudes();
	virtual void GetSubAmplitudeParams(int index, VectorXd& params, int& nLayers);

	virtual PDB_READER_ERRS CalculateSubAmplitudeGrids(double qMax, int gridSize, 
		progressFunc progFunc, void *progArgs, double progMin, double progMax, int *pStop);

	virtual bool SavePDBFile(std::ostream &output);
	virtual bool AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs, bool electronPDB=false);
	
	virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi) = 0;

	/**
	 * @name	GetUseGridWithChildren
	 * @brief	This method determines if from this point down in the model tree all models are set to use a grid.
	 * @ret True iff this model and all children are set to use a grid/lookup table (bUseGrid == true)
	*/
	virtual bool GetUseGridWithChildren() const;
	
	/**
	* @name	GetUseGridWithChildren
	* @brief	This method determines if any descendant in the model tree uses a grid.
	* @ret True iff this model or any descendants are set to use a grid/lookup table (bUseGrid == true)
	*/
	virtual bool GetUseGridAnyChildren() const;

	
	//////////////////////////////////////////////////////////////////////////
	// Model information retrieval procedures
	//////////////////////////////////////////////////////////////////////////

	// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
	static std::string GetLayerParamName(int index, EDPFunction *edpfunc) { return "N/A"; }

	// If index is out of bounds (bounds: [0,infinity)), returns N/A.
	// Usually returns "Solvent" or "Layer #"
	static std::string GetLayerName(int layer) { return "N/A"; }

	// Returns the requested extra parameter's specifications, when index
	// is out of bounds, returns a parameter with name N/A.
	static ExtraParam GetExtraParameter(int index) { return ExtraParam(); }

	// Returns the default value of a layer parameter by its index and layer
	// (spanning from 0 to NumParamLayers)
	static double GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) { return 0.0; }

	// Returns false iff the layer and layer parameter index are not 
	// applicable
	static bool IsParamApplicable(int layer, int lpindex) { return true; }

	// Returns the title of the displayed parameter
	static std::string GetDisplayParamName(int index) { return "N/A"; }

	// Returns the value of the displayed parameter, according to the
	// current parameters of the model
	static double GetDisplayParamValue(int index, const paramStruct *p) { return 0.0; }
};

//forward declare
class JsonWriter;

class LuaSymmetry : public Symmetry, public IGPUGridCalculable{
protected:
	std::string script;
	void *context;
	bool bUsingExternalContext;


	int nlp;
	MatrixXd dol;
	MatrixXd luaParams;

	std::vector<Matrix3d> rot;
	std::vector<Vector3d> trans, rotVars;
	std::complex<FACC> Im;
	std::map<Eigen::Matrix3d, std::vector<Eigen::Vector3d>, matrix3d_less> translationsPerOrientation;

public:
	LuaSymmetry(const std::string& scr, void *luaContext);
	virtual ~LuaSymmetry();

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

	bool VerifyAndObtainData();
	void SetTableFromParamVector( VectorXd& p, int nLayers );

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	virtual bool CalculateGridGPU(GridWorkspace& workspace);

	virtual bool SetModel(GridWorkspace& workspace);

	virtual bool ImplementedHybridGPU();

	virtual ArrayXcX getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi);

};

template <typename T>
inline float3 f3(T d)
{
	float3 f;
	f.x = (float)(d.x);
	f.y = (float)(d.y);
	f.z = (float)(d.z);
	return f;
}
template<>
inline float3 f3(Vector3d d)
{
	float3 f;
	f.x = (float)(d.x());
	f.y = (float)(d.y());
	f.z = (float)(d.z());
	return f;
}

template <typename T>
inline float4 f4(T d)
{
	float4 f;
	f.x = (float)(d.x);
	f.y = (float)(d.y);
	f.z = (float)(d.z);
	f.w = 0.f;
	return f;
}
template<>
inline float4 f4(Vector3d d)
{
	float4 f;
	f.x = (float)(d.x());
	f.y = (float)(d.y());
	f.z = (float)(d.z());
	f.w = 0.f;
	return f;
}

#endif
