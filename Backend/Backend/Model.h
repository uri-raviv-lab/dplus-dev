#ifndef __MODEL_H
#define __MODEL_H

// #undef EXPORTED
// #ifdef _WIN32
// #ifdef CALCULATION
// #define EXPORTED __declspec(dllexport)
// #else
// #define EXPORTED __declspec(dllimport)
// #endif
// #else
// #define EXPORTED extern "C"
// #endif
// 
// Avoid "need dll-interface" warnings (the easy way)
#pragma warning (disable: 4251)
/*
extern template class EXPORTED std::vector<Parameter>;
extern template class EXPORTED std::vector< std::vector<Parameter> >;
*/

#include "Eigen/Core" // For VectorXd
#include <vector> // For std::vector
#include <set>
#include <map>
#include <complex> // For std::complex
#include <limits> // For inf/-inf
#include <cstring> // For strncpy

#include "Common.h"
#include "EDProfile.h" // For EDProfile

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXi;

// Maximal amount of additives in a composite model
#define MAX_COMPOSITE_ADDITIVES 1024

//////////////////////////////////////////////////////////////
///////////////////// Specifications /////////////////////////
//////////////////////////////////////////////////////////////
/********************************************************************************/
/* Model Class Diagram:                *                                        */
/*             IModel                  *                 Amplitude              */
/* Geometry CompositeModel DomainModel * FileAmplitude  GeometricAmp  Symmetry  */
/* +FFModel                            * +PDBAmplitude                          */
/* +SFModel                            * +AmpGrid                               */
/* +BGModel                            *                                        */
/*                                     *                                        */
/*                                     *                                        */
/********************************************************************************/

// Forward declarations
class Amplitude;
class CompositeModel;

//////////////////////////////////////////////////////////////
//////////////// IModel Abstract Interface ///////////////////
//////////////////////////////////////////////////////////////
// An abstract interface presenting a computable model
class EXPORTED_BE IModel {
protected:
	// The global "should we stop" variable
	int *pStop;

	IModel() {}

public:
	virtual ~IModel() {}

	// Set the global pointer that is set when stop is requested
	virtual void SetStop(int *stop) = 0;

	///// Calculation Methods
	//////////////////////////////////////////////////////////////////////////

	// Organize parameters from the parameter vector into the matrix and vector defined earlier.
	virtual void OrganizeParameters(const VectorXd& p, int nLayers) = 0;

	// Called before each series of q-calculations
	virtual void PreCalculate(VectorXd& p, int nLayers) = 0;



	// Calculates an entire vector. Default is in parallel using OpenMP,
	// or a GPU if possible
	virtual VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p,
		progressFunc progress = NULL, void *progressArgs = NULL) = 0;

	// Computes the derivative of the model on an entire vector. Default
	// is numerical derivation (may use analytic derivation)
	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, 
		int nLayers, int ai) = 0;

	/// Returns true if the model or any of its descendants has anomalous scattering
	virtual bool GetHasAnomalousScattering() = 0;

protected:
	// Calculate the model's intensity for a given q
	virtual double Calculate(double q, int nLayers, VectorXd& p) = 0;

	// Calculate an entire vector using a GPU, if applicable
	virtual VectorXd GPUCalculate(const std::vector<double>& q,int nLayers, VectorXd& p) = 0;

	friend class CompositeModel;
};

class EXPORTED_BE ISymmetry {
protected:
	ISymmetry() {}
public:
	virtual ~ISymmetry() {}

	virtual int GetNumSubAmplitudes() = 0;
	virtual Amplitude *GetSubAmplitude(int index) = 0;
	virtual void SetSubAmplitude(int index, Amplitude *subAmp) = 0;
	virtual void AddSubAmplitude(Amplitude *subAmp) = 0;
	virtual void RemoveSubAmplitude(int index) = 0;
	virtual void ClearSubAmplitudes() = 0;
	virtual void GetSubAmplitudeParams(int index, VectorXd& params, int& nLayers) = 0;

	// Stateful function (modifies the symmetry object)
	virtual bool Populate(const VectorXd& p, int nLayers) = 0;
	virtual unsigned int GetNumSubLocations() = 0;
	virtual LocationRotation GetSubLocation(int index) = 0;

	/**
	 * @name	SavePDBFile
	 * @brief	Saves a PDB file of the entire substructure
	 *			Saves a PDB file of all descendant PDBs.
				Any non-PDB/symmetry structures are not saved.
				None of the original headers are saved.
	 * @param	std::wstring & savefilename The path where to save the file
	 * @ret		True if a file was saved
	*/
	virtual bool SavePDBFile(std::ostream &output) = 0;

	/**
	 * @name	AssemblePDBFile
	 * @brief	Prepares the lines and locations of all descendant-PDB structures
	 * @param[in,out]	std::vector<std::string> & lines
									Contains the textual "ATOM  " lines
	 * @param[inout]	std::vector<Eigen::Vector3f> & locs
									Contains the rotated and translated locations of all the sub atoms
	 * @ret		True if no errors were encountered
	*/
	virtual bool AssemblePDBFile(std::vector<std::string> &lines, std::vector<Eigen::Vector3f> &locs, bool electronPDB=false) = 0;
};

// A special model representing a model which is polydisperse in one 
// parameter. This class may be nested in itself for N polydisperse
// parameters.
class PolydisperseModel : public IModel {
protected:
	IModel *model;
	double polySigma;
	int polyInd;
	bool bDeleteInner;
	VectorXd p;
	int pdResolution;
	PeakType pdFunction;

	int *pStop;

public:
	PolydisperseModel(IModel *m, int index, double sigma, const VectorXd& bigP,
		int pdRes, PeakType pdFunc, bool deleteInnerModel = false) : 
		model(m), polyInd(index), polySigma(sigma),
		p(bigP), pdResolution(pdRes), pdFunction(pdFunc), bDeleteInner(deleteInnerModel) {}

	virtual ~PolydisperseModel() { if(model && bDeleteInner) delete model; }	

	virtual VectorXd Derivative(const std::vector<double>& x, VectorXd param, 
								int nLayers, int ai);

	virtual void SetStop(int *stop) { pStop = stop; }
	virtual void OrganizeParameters(const VectorXd& p, int nLayers) {}
	virtual void PreCalculate(VectorXd& p, int nLayers) {}
	virtual bool GetHasAnomalousScattering();



	virtual VectorXd GPUCalculate(const std::vector<double>& q,int nLayers, VectorXd& p);

	virtual Eigen::VectorXd CalculateVector(const std::vector<double>& q, int nLayers, VectorXd& p,
		progressFunc progress = NULL, void *progressArgs = NULL);

	virtual IModel *GetInnerModel();

protected:
	virtual double Calculate(double q, int nLayers, VectorXd& p);
};

// An interface for models that ceres-solver can fit
class EXPORTED_BE ICeresModel
{
protected:
	std::map<int, double*> mutParams;
	VectorXd* pVecCopy;

	ICeresModel() {}
public:
	virtual ~ICeresModel() {}

	virtual bool CalculateVectorForCeres(const double* qs, double const* const* p,
										 double* residual, int points) = 0;

	virtual void SetInitialParamVecForCeres(VectorXd* p, std::vector<int> &mutIndices) = 0;
	virtual void GetMutatedParamVecForCeres(VectorXd* p) = 0;
	virtual void SetMutatedParamVecForCeres(const VectorXd& p) = 0;
};

// A model that computes a weighted average of sub-models, with optional multipliers. Ceres-friendly.
// NOTE: THIS MODEL DOES NOT MANAGE MEMORY, DELETE THE INNER MODELS YOURSELF!

//forward declare:
class JsonWriter;

class EXPORTED_BE CompositeModel : public IModel, public ICeresModel
{
	// Parameters: Weights (avg. population size per IModel)
protected:
	std::vector<IModel *> _additives;
	std::vector<VectorXd> _params;	
	std::vector<int> _layers;
	
	std::vector< std::vector<IModel *> > _multipliers; // Mapping multiplier models to indices
	std::vector<IModel *> _fullMults; // Each multiplier appears once here
	std::map<IModel *, VectorXd> _multParams;
	std::map<IModel *, int> _multLayers;

	std::string _previous_hashes;
	Eigen::ArrayXd _previous_parameters;
	Eigen::ArrayXd _previous_intensity;
	Eigen::ArrayXd _previous_q_values;


	std::vector<double> _weights; // Weights for each additive
	double scale, _previous_scale;
	double constant, _previous_constant;

	virtual double Calculate(double q, int nLayers, VectorXd& p);

	virtual bool GetHasAnomalousScattering();

	// The global "should we stop" variable
	int *pStop;

public:
	CompositeModel() : scale(1.0), constant(0.0) {}
	virtual ~CompositeModel() {}	

	std::vector<IModel *> GetSubModels() const { return _additives; }
	void SetSubModels(const std::vector<IModel *>& models) { _additives = models; _multipliers.resize(models.size()); }

	void ClearMultipliers() { _multipliers.clear(); _multipliers.resize(_additives.size());
							  _fullMults.clear(); _multParams.clear(); _multLayers.clear(); }
	void AddMultiplier(IModel *mult, const std::vector<unsigned int>& indices);
	std::vector<IModel *> GetMultipliers() const { return _fullMults; }

	// IModel implementers 
	virtual void OrganizeParameters(const VectorXd& p, int nLayers);

	virtual void SetStop(int *stop);

	virtual void PreCalculate( VectorXd& p, int nLayers );


	virtual VectorXd CalculateVector( const std::vector<double>& q, int nLayers , VectorXd& p, progressFunc progress = NULL, void *progressArgs = NULL);

	virtual VectorXd Derivative( const std::vector<double>& x, VectorXd param, int nLayers, int ai );

	virtual VectorXd GPUCalculate( const std::vector<double>& q,int nLayers, VectorXd& p);

	virtual void GetHeader(unsigned int depth, std::string &header);
	virtual void GetHeader(unsigned int depth, JsonWriter &writer);

	// ICeresModel implementers
	virtual bool CalculateVectorForCeres(const double* qs, double const* const* p,
										 double* residual, int points);

	virtual void SetInitialParamVecForCeres(VectorXd* p, std::vector<int> &mutIndices);
	virtual void GetMutatedParamVecForCeres(VectorXd* p);
	virtual void SetMutatedParamVecForCeres(const VectorXd& p);
};

VectorXd NumericalDerivative(IModel *mod, const std::vector<double>& x, VectorXd param, int nLayers, int ai,
							 double epsilon);

#endif
