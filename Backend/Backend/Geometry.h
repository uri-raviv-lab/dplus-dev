#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include "Model.h"

//////////////////////////////////////////////////////////////
/////////////////// Geometry Abstract Class //////////////////
//////////////////////////////////////////////////////////////

class EXPORTED_BE Geometry : public IModel  {

protected:
	/// A pointer to the name of GPU kernel that calculates
	/// the model, if applicable
	const char *GPUKernel;

	/// The number of parameters per layer
	int nLayerParams;

	/// Minimal and maximal amount of layers (if maxLayers is -1, layers can be
	/// infinite)
	int minLayers, maxLayers;

	/// The number of extra parameters
	int nExtraParams;

	/// The electron density profile specifier
	EDProfile profile;
	EDPFunction *profileFunc;

	/// The number of displayed parameters
	int displayParams;

	/// The display name of this model
	std::string modelName;

	///a Matrix and a vector that contain the parameter structure in a logical way.
	MatrixXd *parameters;
	VectorXd *extraParams;

	/// q-range of the vector to be generated
	double qmin, qmax;

	/// A flag indicating whether or not a coarse parallelization is possible
	/// Override CalculateVector to set as false
	bool bParallelizeVector;

	/// Calculate the model's intensity for a given q
	virtual double Calculate(double q, int nLayers, VectorXd& p) = 0;

	/// Calculate an entire vector using a GPU, if applicable
	virtual VectorXd GPUCalculate(const std::vector<double>& q, int nLayers, VectorXd& p);

	virtual bool GetHasAnomalousScattering();


	Geometry(std::string name = "Abstract Model - DO NOT USE", 
		  int extras = 2, int nlp = 2, int minlayers = 2, 
		  int maxlayers = -1, EDProfile edp = EDProfile(),
		  int disp = 0);	
	
public:

	// Destructor
	virtual ~Geometry();


	///// Get/Set Methods

	// TODO::dox Cleanup methods and fields

	// Returns this model's display name
	virtual std::string GetName();

	// Returns the electron density profile specification
	virtual EDProfile GetEDProfile();

	// Sets a new electron density profile
	virtual void SetEDProfile(EDProfile edp);

	// Returns true if the calculation of the vector is parallelizable
	virtual bool ParallelizeVector();	

	// Set the global pointer that is set when stop is requested
	virtual void SetStop(int *stop);

	// Gets a vector of all the parameters
	virtual std::vector<double> GetAllParameters();

	///// Calculation Methods
	//////////////////////////////////////////////////////////////////////////

	// Organize parameters from the parameter vector into the matrix and vector defined earlier.
	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	// Called before each series of q-calculations
	virtual void PreCalculate(VectorXd& p, int nLayers);


	// Calculates an entire vector. Default is in parallel using OpenMP,
	// or a GPU if possible
	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p,
								 	 progressFunc progress = NULL, void *progressArgs = NULL);

	// Computes the derivative of the model on an entire vector. Default
	// is numerical derivation (may use analytic derivation)
	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, 
								int nLayers, int ai);

	// Geometry information
	//////////////////////////////////////////////////////////////////////////

	// Returns the minimal amount of layers for this model
	virtual int GetMinLayers();

	// Returns the maximal amount of layers for this model
	virtual int GetMaxLayers();

	// Returns the number of layer parameters
	virtual int GetNumLayerParams();

	// Returns the number of extra parameters
	virtual int GetNumExtraParams();

	// Returns true iff the model is layer-based, capable of having an electron
	// density profile
	virtual bool IsLayerBased();

	// Returns the number of displayed parameters
	virtual int GetNumDisplayParams();

	// Geometry information procedures
	//////////////////////////////////////////////////////////////////////////

	// If index is out of bounds (bounds: [0,nLayerParams)), returns N/A.
	static std::string GetLayerParamName(int index, EDPFunction *edpfunc);

	// If index is out of bounds (bounds: [0,infinity)), returns N/A.
	// Usually returns "Solvent" or "Layer #"
	static std::string GetLayerName(int layer);

	// Returns the requested extra parameter's specifications, when index
	// is out of bounds, returns a parameter with name N/A.
	static ExtraParam GetExtraParameter(int index);

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


};

//////////////////////////////////////////////////////////////
/////////////////// Special Geometry Types //////////////////////
//////////////////////////////////////////////////////////////

// Custom scripted model
class LuaModel : public Geometry {
protected:
	void *luactx;
	bool bContextCreated;
	std::string modelCode;

	VectorXd parVec;	// BG funcs need to save the parameters to the obj
	typedef double (*modelFunction)(double q, VectorXd& p, int ma, int nd);

public:
	LuaModel(std::string script, void *luaContext);

	virtual ~LuaModel();

	virtual void PreCalculate(VectorXd &p, int nLayers);

protected:
	virtual double Calculate(double q, int nLayers, VectorXd& p);
};

//////////////////////////////////////////////////////////////
/////////////////// Form Factor Models ///////////////////////
//////////////////////////////////////////////////////////////

// Performs orientation average on a given model for a given q
class FFModel;
double OrientationAverage(double q, FFModel *model, int nLayers, VectorXd& p);
// A model containing a form factor
class EXPORTED_BE FFModel : public Geometry {
protected:
	FFModel(std::string name = "Abstract FF Model - DO NOT USE",
		int extras = 2, int nlp = 2, int minlayers = 2, int maxlayers = -1,
		EDProfile edp = EDProfile()) : 
			Geometry(name, extras, nlp, minlayers, maxlayers, edp) {}

	// Calculate the model's intensity for a given q. Default
	// is numerical orientation average of the |FF|^2.
	virtual double Calculate(double q, int nLayers, VectorXd& p) {
		return OrientationAverage(q, this, nLayers, p);
	}

public:
	// Virtual destructor
	virtual ~FFModel() {}

	// Called before each series of q-calculations
	virtual void PreCalculateFF(VectorXd& p, int nLayers) {}

	// Calculate the model's form factor for a given q vector = 
	// (qx,qy,qz) in Cartesian coordinates.
	// Because many times the FFCalculate contains a Dirac Delta Function, we need to 
	// take it into account: after discretization: w - is the weight of the point at
	// the integral in which the function is used and precision is the precision of the delta function
	virtual std::complex<double> CalculateFF(Vector3d qvec, 
											 int nLayers, double w = 1.0, double precision =1E-5, 
                                                 VectorXd* p = NULL) = 0;

	// Returns true iff this form factor has a special structure
	// factor function
	virtual bool HasSpecializedSF() { return false; }

	// Returns a special structure factor function (Geometry),
	// such as Caille in membranes/slabs
	virtual Geometry *GetSpecializedSF() { return NULL; }

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, int nLayers, int ai);

	virtual std::string GetLayerName(int layer) = 0;
	static std::string GetLayerNameStatic(int layer);

	virtual ExtraParam GetExtraParameter(int index) = 0;
	static ExtraParam GetExtraParameterStatic(int index);

};



#endif
