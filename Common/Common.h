#ifndef __COMMON_H
#define __COMMON_H

// Where all the definitions are

// Determine which project is including this file and how to treat import/export
// of functions from/to dynamicly loaded libraries.
#undef EXPORTED
#undef EXPORTED_BE
#ifdef _WIN32
	#ifdef CALCULATION
		#ifdef BACKEND
			#define EXPORTED_BE __declspec(dllexport)
			#define EXPORTED_BE_FUNCTION
			#define EXPORTED
		#else
			#define EXPORTED __declspec(dllexport)
			#define EXPORTED_BE_FUNCTION
			#define EXPORTED_BE
		#endif // BACKEND
	#else
		#ifdef BACKEND
			#define EXPORTED_BE __declspec(dllimport)
			#define EXPORTED_BE_FUNCTION
			#define EXPORTED
		#else
			#define EXPORTED __declspec(dllimport)
			#define EXPORTED_BE_FUNCTION
			#define EXPORTED_BE
		#endif // BACKEND

	#endif
#else
	#ifdef CALCULATION
		#ifdef BACKEND
			#define EXPORTED_BE __attribute__ ((visibility ("default")))
			#define EXPORTED_BE_FUNCTION __attribute__ ((visibility ("default")))
			#define EXPORTED
		#else
			#define EXPORTED __attribute__ ((visibility ("default")))
			#define EXPORTED_BE_FUNCTION __attribute__ ((visibility ("default")))
			#define EXPORTED_BE
		#endif // BACKEND
	#else
		#ifdef BACKEND
			#define EXPORTED_BE __attribute__ ((visibility ("default")))
			#define EXPORTED_BE_FUNCTION __attribute__ ((visibility ("default")))
			#define EXPORTED
		#else
			#define EXPORTED __attribute__ ((visibility ("default")))
			#define EXPORTED_BE_FUNCTION __attribute__ ((visibility ("default")))
			#define EXPORTED_BE
		#endif // BACKEND

	#endif
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef _SCL_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#endif

// Defines that are used for CUDA, that if not in a CUDA file should be ignored
#ifndef __host__
#define __host__
#define __device__
#define __forceinline__ inline
#endif

#define __HDFI__ __host__ __device__ __forceinline__

#undef min
#undef max

#undef OUT
/// Used to indicate that a passed parameter will be changed by the function
#define OUT

#undef IN
/// Used to indicate that a passed parameter will be used as input by the function
#define IN

#include <string>
#include <vector>
#include <limits>
#include <fstream>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <locale.h>

using std::abs;

/// Converts a std::wstring (\p wstr) to a std::string and returns the result
inline std::string wstringtoutf8(const std::wstring& wstr)
{
    setlocale(LC_ALL, "C.UTF-8");
    int utflen = (int)wcstombs(NULL, wstr.c_str(), 0) + 1;
    if(utflen == 0)
        return ""; // If there were invalid characters (non UTF-8)
        
    char *utfstr = new char[utflen];

    wcstombs(utfstr, wstr.c_str(), utflen);

    std::string result = utfstr;
    
    delete[] utfstr;

    return result;
}

//////////////////////////////////////////////////////////////////////////
// Enumerations and Definitions
//////////////////////////////////////////////////////////////////////////

// Positive/Negative infinity definitions
#define POSINF (std::numeric_limits<double>::infinity())
#define NEGINF (-std::numeric_limits<double>::infinity())

/// Rounds a double (\p val) to the nearest int and returns it
inline int dbltoint(double val) { return (int)floor(val + 0.5); }
/// Rounds a double (\p val) to the nearest unsigned int and returns it
inline unsigned int dbltouint(double val) { return (unsigned int)floor(val + 0.5); }

// Floating point ACCuracy
typedef double FACC;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;

typedef unsigned int JobPtr;
typedef unsigned int ModelPtr;


#define FIRST_MACRO_PAR(Enum, String, NumParameters)  Enum,
#define SECOND_MACRO_PAR_CASE(Enum, String, NumParameters) case Enum: return String;
#define THIRD_MACRO_PAR_CASE(Enum, String, NumParameters) case Enum: return NumParameters;

#define CREATE_ENUM_MACRO(ENUM_NAME, DEFINE_NAME)	\
	typedef enum {	DEFINE_NAME(FIRST_MACRO_PAR) ENUM_NAME##_SIZE, } ENUM_NAME##_Enum;

#define CREATE_ToCString_MACRO(ENUM_NAME, DEFINE_NAME)						\
	static const char* ENUM_NAME##ToCString(ENUM_NAME##_Enum type) {		\
		switch(type) { DEFINE_NAME(SECOND_MACRO_PAR_CASE) }; return NULL;	\
	};

#define CREATE_NumParameters_MACRO(ENUM_NAME, DEFINE_NAME)					\
	static int ENUM_NAME##NumberOfParameters(ENUM_NAME##_Enum type) {	\
		switch(type) { DEFINE_NAME(THIRD_MACRO_PAR_CASE) };	return -1;		\
	};

static bool STRCMP(const char *a, const char *b)
{
	while(*a != '\0' && *b != '\0' && *a == *b) {
		a++; b++;
	}
	return (*a == '\0' && *b == '\0');
}

#define CREATE_cStringtoEnum_MACRO(ENUM_NAME, DEFINE_NAME)						\
	static ENUM_NAME##_Enum ENUM_NAME##fromCString(const char *name) {	\
	  for(int i = 0; i < (int) ENUM_NAME##_SIZE ; i++) \
			if(STRCMP(name, ENUM_NAME##ToCString(ENUM_NAME##_Enum(i))))			\
				return ENUM_NAME##_Enum(i);										\
		return ENUM_NAME##_SIZE;												\
	}

#define CREATE_ENUM_AND_CSTRING(ENUM_NAME, DEFINE_NAME)	\
	CREATE_ENUM_MACRO(ENUM_NAME, DEFINE_NAME);						\
	CREATE_ToCString_MACRO(ENUM_NAME, DEFINE_NAME);					\
	CREATE_cStringtoEnum_MACRO(ENUM_NAME, DEFINE_NAME);

#define CREATE_ENUM_CSTRING_AND_NUMPARAMS(ENUM_NAME, DEFINE_NAME)	\
	CREATE_ENUM_AND_CSTRING(ENUM_NAME, DEFINE_NAME);				\
	CREATE_NumParameters_MACRO(ENUM_NAME, DEFINE_NAME);

enum ErrorCode {
	OK = 0,
	
	// Job-related
	ERROR_STOPPED, // 1 The job was manually stopped
	ERROR_JOBRUNNING, //2 The job is already running
	ERROR_JOBNOTFOUND,  //3 The job cannot be found
	ERROR_OUTOFMODELS, //4 Too many models have been allocated
	
	// Backend-related
	ERROR_BACKEND, //5 Error accessing backend
	ERROR_FILENOTFOUND, //6 The file cannot be found
	ERROR_INVALIDCONTAINER, //7 Bad model/renderer container
	ERROR_UNSUPPORTED, //8 The backend does not support this operation	
	ERROR_ILLEGAL_JSON, //9 Illegal JSON passed to Backend
	ERROR_NO_MUTABLES_FOR_FIT, //10 No variables are marked as mutable for fitting process
	ERROR_PARAMETER_NOT_WITHIN_CONSTRAINTS, //11 There is at least one mutable parameter with infeasible bounds
	ERROR_CERES_INITIALIZATION_FAILURE, //12 Fitting process failed to initialize. Please check all parameters
	ERROR_INSUFFICIENT_MEMORY, //13 Insufficient memory allocating a grid

	// Model-related
	ERROR_MODELNOTFOUND, //14 The model was not found within the container or job
	ERROR_INVALIDMODEL, //15 The requested model is of an incorrect type
	ERROR_INVALIDMODELTYPE, //16 The specified model type is invalid or unknown
	ERROR_INCOMPATIBLEMODEL, //17 The sub-model or parent model are incompatible with each other
	ERROR_UNFITTABLEMODEL, //18 The selected model cannot be fit using the model fitter
	ERROR_INVALIDPARAMTREE, //19 The input parameter tree is invalid
	ERROR_UNIMPLEMENTED_CPU, //20 The specific set of operations contains something that hasn't been implemented (VEGAS with spacefilling symmetry)
	
	
	// General
	ERROR_INVALIDARGS, //21 The function or method has been called with invalid arguments
	ERROR_NOMUTABLES, //22 A fitting job was created, but there are no parameters to change
	ERROR_UNSUCCESSFULLFITTING, //23 The fitter was unable to find a better set of parameters
	ERROR_GENERAL, //24 A general error has occurred (crash/unknown error)	
	ERROR_NO_ACCELERATORS, //25Accelerators do not exist on computer (is this a GPU-capable machine?)

	// Web
	ERROR_NO_RESPONSE, //26
	ERROR_BAD_TOKEN, //27
	ERROR_INTERNAL_SERVER, //28

	//Registry
	ERROR_TDR_LEVEL, //29
	AVX, // 30
	ERROR_NO_GPU// 31
};

static const char *g_errorStrings[] = {
	"OK",

	// Job-related
	"The job was manually stopped",
	"A job is already running",
	"The job cannot be found",
	"Too many models have been allocated",

	// Backend-related
	"Error accessing backend",
	"The file cannot be found",
	"Bad model or renderer container",
	"The backend does not support this operation",
	"Illegal JSON passed to Backend",
	"No variables are marked as mutable for fitting process",
	"There is at least one mutable parameter with infeasible bounds",
	"Fitting process failed to initialize. Please check all parameters",
	"Insufficient memory allocating a grid",

	// Model-related
	"The model was not found within the container or job",
	"The requested model is of an incorrect type",
	"The specified model type is invalid or unknown",
	"The sub-model or parent model are incompatible with each other",
	"The selected model cannot be fit using the model fitter",
	"The input parameter tree is invalid",
	"The current integration method was not implemented for CPU usage (probably means you selected VEGAS and a space-filling symmetry, which haven't been implemented on a GPU).",


	// General
	"The function or method has been called with invalid arguments",
	"A fitting job was created, but there are no parameters to change",
	"The fitter was unable to find a better set of parameters",
	"A general error has occurred",
	"Accelerators do not exist on computer (is this a GPU-capable machine?)",

	//Web-related
	"No response from server",
	"Invalid activation code",
	"Internal server error",

	//Registry
	"The server GPU is not configured properly",
	"You're using an old CPU, CPU requires an Intel CPU from 2011 or later",
	"This computer doesn't have a GPU"
};

enum JobType {
	JT_NONE,
	JT_GENERATE,
	JT_FIT
};

// Model types (bitmap, can be ORed)
enum ModelType { 
	MT_FORMFACTOR = 0x01,      // 0001
	MT_STRUCTUREFACTOR = 0x02, // 0010
	MT_BACKGROUND = 0x04,      // 0100
	MT_SYMMETRY = 0x08,        // 1000
	
	// Shorthands
	MT_FFSF = 0x03, // MT_FORMFACTOR | MT_STRUCTUREFACTOR
	MT_FFBG = 0x05, // MT_FORMFACTOR | MT_STRUCTUREFACTOR
	MT_SFBG = 0x06, // MT_STRUCTUREFACTOR | MT_BACKGROUND
	MT_ALL  = 0x07, // MT_FORMFACTOR | MT_STRUCTUREFACTOR | MT_BACKGROUND
};

/// Electron density profile type
enum ProfileType {
	NONE,        ///< E.D. profile is N/A (i.e. cuboid/helices)
	SYMMETRIC,   ///< Symmetric E.D. profile
	ASYMMETRIC,  ///< Asymmetric E.D. profile
};

/// Electron density profile function shape. Each shape (except DISCRETE) should
/// implement a corresponding EDPFunction subclass.
enum ProfileShape {
	DISCRETE,    ///< Discrete E.D. profile
	GAUSSIAN,    ///< Gaussian E.D. profile
	TANH,        ///< Hyperbolic-tangent smooth E.D. profile
};

/// File-based amplitude types
enum AmpFileType { 
	AF_PDB,
	AF_AMPGRID,
};

/// The level of detail to be displayed in the 3D graphics pane in the GUI
enum LevelOfDetail {
	LOD_NONE    = 1, ///< Display nothing
	LOD_VERYLOW = 2, ///< Display polygons instead of PDBs, really low quality quadrics
	LOD_LOW     = 3, ///< Low quality quadrics
	LOD_MEDIUM  = 4, ///< Medium quality quadrics
	LOD_HIGH    = 5, ///< High (old) quality
};

/// Possible errors relating to reading PDB files
enum PDB_READER_ERRS {
	PDB_OK = 0,

	UNINITIALIZED,

	ERROR_WITH_GPU_CALCULATION_TRY_CPU,

	NO_FILE,
	FILE_ERROR,
	ERROR_IN_PDB_FILE,
	NO_ATOMS_IN_FILE,

	MEMORY_ERROR,
	OUT_OF_RANGE,

	BAD_ATOM,

	UNIMPLEMENTED,
	MISMATCHED_DATA,	

	STOPPED,

	GENERAL_ERROR,

	MISSING_HARDWARE,
};

/// Possible errors relating to the status of an amplitude grid (Grid and
/// its descendants)
enum AMPLITUDE_GRID_STATUS {
	AMP_READY = 0,
	AMP_NOT_USED,
	AMP_UNINITIALIZED,
	AMP_OUT_OF_DATE,
	AMP_CACHED,
	AMP_HAS_INVALID_NUMBER
};

/// The different atomic radii sets
enum ATOM_RADIUS_TYPE {
	RAD_UNINITIALIZED,
	RAD_VDW,
	RAD_EMP,
	RAD_CALC,
	RAD_DUMMY_ATOMS_ONLY,
	RAD_DUMMY_ATOMS_RADII,
	RAD_SIZE
};

/// Default Polydispersity resolution
#define DEFAULT_PDRES 15

enum PeakType { SHAPE_GAUSSIAN, SHAPE_GAUSSIAN_FWHM, SHAPE_LORENTZIAN, SHAPE_LORENTZIAN_SQUARED, SHAPE_CAILLE };

enum PhaseType{	PHASE_NONE, PHASE_LAMELLAR_1D, PHASE_2D,PHASE_RECTANGULAR_2D,
PHASE_CENTERED_RECTANGULAR_2D, PHASE_SQUARE_2D, PHASE_HEXAGONAL_2D, PHASE_3D, PHASE_RHOMBO_3D, 
				PHASE_HEXA_3D, PHASE_MONOC_3D,PHASE_ORTHO_3D,PHASE_TETRA_3D,PHASE_CUBIC_3D};

enum QuadratureMethod { QUAD_MONTECARLO, QUAD_GAUSSLEGENDRE, QUAD_SIMPSON };

enum BGFuncType { BG_EXPONENT, BG_LINEAR, BG_POWER };

enum FitMethod { FIT_LM, FIT_DE, FIT_RAINDROP, FIT_LBFGS };

/// The available orientation averaging (integration) methods
//enum OAMethod { OA_MC = 0, OA_ADAPTIVE_GK, OA_CUBICSPLINE, OA_DIRECT_GPU };
#define DEFINE_OAMETHOD_MACRO(X) \
        X(OA_MC, "Monte Carlo (Mersenne Twister)", 0)			\
	X(OA_ADAPTIVE_MC_VEGAS, "Adaptive (VEGAS) Monte Carlo", 0)	\
	X(OA_ADAPTIVE_GK, "Adaptive Gauss Kronrod", 0)			\
//	X(OA_DIRECT_GPU, "Direct Computation - MC", 0)			\ // Hidden for review; should be reimplemented when cpu/gpu unification is complete
//	X(OA_MC_SOBOL, "Monte Carlo (Sobol) - unimplemented", 0)

CREATE_ENUM_AND_CSTRING(OAMethod, DEFINE_OAMETHOD_MACRO);
#undef DEFINE_OAMETHOD_MACRO
#ifdef _WIN32
#define STDCALL __stdcall
#else
#define STDCALL
#endif

/// Used for passing progress information between threads/machines
typedef void (STDCALL *progressFunc)(void *args, double progress);
typedef void (STDCALL *notifyCompletionFunc)(void *args, int error);

//////////////////////////////////////////////////////////////////////////
// Data Structures (Portable)
//////////////////////////////////////////////////////////////////////////

// Data structure alignment
#pragma pack(push)
#pragma pack(16)

/// Electron density profile specifier
struct EDProfile {
	ProfileType type;
	ProfileShape shape;

	// Constructor
	EDProfile(ProfileType t = SYMMETRIC, ProfileShape s = DISCRETE) :
	type(t), shape(s) {}
};

/**
 * A data structure representing all the information known about a certain model
 *
 * @param name The name of the model
 * @param category The ID of the category to which the model belongs
 * @param modelIndex The index of the model in the container
 * @param isLayerBased Is the model layer-based (for electron density profile computations)
 * @param nlp The number of layer parameters
 * @param minLayers The minimal amount of layers for this model
 * @param maxLayers The maximal amount of layers for this model
 * @param nExtraParams Number of extra parameters for this model
 * @param nDispParams Number of display parameters for this model
 * @param isGPUCompatible Is there a GPU implementation of this model?
 * @param isSlow Is the model slow?
 * @param relatedModels An array representing the indices of the models related to this one.
                        Array terminator: -1
 * @param ffImplemented Is CalculateFF properly implemented? (does this model have a separate form factor?)
**/
struct ModelInformation {
	char name[256];
	int category, modelIndex;
	bool isLayerBased;
	int nlp, minLayers, maxLayers;
	int nExtraParams, nDispParams;
	bool isGPUCompatible, isSlow;
	bool ffImplemented;
	
	ModelInformation(const char *modelname = "Abstract Model - DO NOT USE",
					 int cat = -1, int ind = -1, bool layerbased = true, 
					 int layerparams = 2, int minlayer = 1, int maxlayer = 2, 
					 int exparams = 2, int disps = 0, bool gpu = false,
					 bool slow = false, bool calcff = false, bool datarequired = false) : 
					 category(cat), modelIndex(ind),
					 isLayerBased(layerbased), nlp(layerparams), minLayers(minlayer),
					 maxLayers(maxlayer), nExtraParams(exparams), nDispParams(disps),
					 isGPUCompatible(gpu), isSlow(slow), ffImplemented(calcff)
					 {
						 memset(name, 0, sizeof(char) * 256);
						 if(modelname)
							 strncpy(name, modelname, 256);						 
					 }
	ModelInformation(const ModelInformation& copy) {
		memset(name, 0, sizeof(char) * 256);
		strncpy(name, copy.name, 256);
		category		= copy.category;	
		modelIndex		= copy.modelIndex;
		isLayerBased	= copy.isLayerBased;
		nlp				= copy.nlp;
		minLayers		= copy.minLayers;
		maxLayers		= copy.maxLayers;
		nExtraParams	= copy.nExtraParams;
		nDispParams		= copy.nDispParams;
		isGPUCompatible	= copy.isGPUCompatible;
		isSlow			= copy.isSlow;
		ffImplemented   = copy.ffImplemented;
	}
};

/// A data structure representing a model category in a model container.
/// Cannot have more than 16 related models in a single category.
struct ModelCategory { 
	char name[256];
	ModelType type;
	int models[16];
};

// Forward declaration
struct Radian;

/**
brief A struct representing a floating point number with units of degrees.
This struct represents a degree (i.e. [0-360) of a circle). The class should NOT
be able to convert implicitely to a number as that would allow it to be used as
sin(someDegree) which would (usually) be incorrect. As such, the constructors are
explicit as well. For the sake of expliciteness, we won't overload the assignment
operator from a Radian as well.
*/
struct Degree 
{
	/// The numerical value of the Degree object
	float deg;

	/// Constructor from a numerical value
	__HDFI__ explicit Degree(float angle = 0.f) : deg(angle) {}
	/// Constructor from a Radian struct
	__HDFI__ explicit Degree(const Radian& rad);
	/// Nothing to destroy
	__HDFI__ ~Degree() {}

	__HDFI__ Degree& operator=(const Degree& rhs) {
		if(this != &rhs)
			deg = rhs.deg;
		return *this;
	}
};

/**
brief A struct representing a floating point number with units of radians.
This struct represents radians (i.e. [0-2\pi) of a circle). The class can
be implicitely cast to a number so that sin(someRadian) will work. The
constructors are explicit so that the programmer will be able to ensure that
it is indeed radians and not degrees.
*/
struct Radian
{
	/// The numerical value
	float rad;

	/// Constructor from a floating point number
	__HDFI__ explicit Radian(float angle = 0.f) : rad(angle) {}
	/// Constructor from a Degree struct
	__HDFI__ explicit Radian(const Degree& deg) : rad (deg.deg * 0.017453292519943295769f) {}
	/// Nothing to destroy
	__HDFI__ ~Radian() {}

	__HDFI__ Radian& operator=(const Radian& rhs) {
		if(this != &rhs)
			rad = rhs.rad;
		return *this;
	}

	/// Constructor from double, to remove infinite warnings
	__HDFI__ explicit Radian(double angle) : rad((float)angle) {}

	__HDFI__ operator float() const{
		return rad;
	}
/*	If defined, will be ambiguous when sending to a stream.*/
/*
	__HDFI__ operator double() const{
		return this->operator float();
	}
*/

};

__HDFI__ Degree::Degree(const Radian& rad) : deg(rad.rad * 57.295779513082320877f) {}

/*
__HDFI__ Degree& Degree::operator=( const Radian &rad ) {
	deg = rad.rad * 57.295779513082320877;
	return *this;
}*/

/**********************/
/* URI AND AVI REACHED HERE!
***********************/
struct LocationRotation {
	double x, y, z;

	// In radians
	Radian alpha, beta, gamma;

	__HDFI__ LocationRotation(double _x = 0.0, double _y = 0.0, double _z = 0.0,
		double _alpha = 0.0, double _beta = 0.0, double _gamma = 0.0) : 
	x(_x), y(_y), z(_z), alpha(_alpha), beta(_beta), gamma(_gamma) {}
};

// The data structure representing a Model parameter
struct Parameter {
	static const int ELEMENTS = 9;

	// The parameter's initial value
	double value; 

	// Determines whether this parameter may change during fitting
	bool isMutable;

	// True iff the value is, during fitting, constrained
	// to be between consMin and consMax
	bool isConstrained;
	double consMin, consMax;

	// If this parameter is dynamically constrained to
	// other parameters, each of the values are the constraint
	// parameter indices. Otherwise, the value is -1.
	int consMinIndex, consMaxIndex;

	// NOTE: consMin and consMax may be -inf 
	// and inf respectively	

	// If this parameter is linked to another, this value is its
	// index. Otherwise, it is -1.
	int linkIndex;

	// If this parameter is poly-dispersed, this value will be larger than
	// 0.0 and will mean the std. deviation of the parameter.
	double sigma;

	// Constructor
	Parameter(double val = 0.0, bool bMutable = false, bool bCons = false, 
		double consmin = NEGINF, double consmax = POSINF,
		int minInd = -1, int maxInd = -1, int linkInd = -1, 
		double stddev = 0.0) :
	value(val), isMutable(bMutable), isConstrained(bCons),
		consMin(consmin), consMax(consmax), consMinIndex(minInd),
		consMaxIndex(maxInd), linkIndex(linkInd), sigma(stddev) {}
	
	/**
	 * Method that fills a preallocated array with the fields of the Parameter
	 * @param res A preallocated double array to which the values of all the fields
	 *				of the Parameter are written
	 **/
	void ParameterToDoubleArray(double* res) {
		res[0] = value;
		res[1] = isMutable ? 1.0 : 0.0;
		res[2] = isConstrained ? 1.0 : 0.0;
		res[3] = consMin;
		res[4] = consMax;
		res[5] = double(consMinIndex) + double(abs(consMinIndex)/consMinIndex)*0.0000001;
		res[6] = double(consMaxIndex) + double(abs(consMaxIndex)/consMaxIndex)*0.0000001;
		res[7] = double(linkIndex) + double(abs(linkIndex)/linkIndex)*0.0000001;
		res[8] = sigma;
	}
	bool ParamValidateConstraint()
	{
		if (consMax < consMin)
			return false;
		return true;
	}
	void SwapMinMaxValue()
	{
		double tmp = consMax;
		consMax = consMin;
		consMin = tmp;
	}
	bool operator == (const Parameter& rhs) const {
		if(fabs((1.0 - this->value / rhs.value)) > 1.0e-9)
			return false;
		if(fabs((1.0 - this->consMin / rhs.consMin)) > 1.0e-9)
			return false;
		if(fabs((1.0 - this->consMax / rhs.consMax)) > 1.0e-9)
			return false;
		if(fabs((1.0 - this->sigma / rhs.sigma)) > 1.0e-9)
			return false;
		
		if(this->isMutable != rhs.isMutable)
			return false;
		if(this->isConstrained != rhs.isConstrained)
			return false;
		if(this->consMinIndex != rhs.consMinIndex)
			return false;
		if(this->consMaxIndex != rhs.consMaxIndex)
			return false;
		if(this->linkIndex != rhs.linkIndex)
			return false;

		return
			true;

	}
	bool operator != (const Parameter& rhs) const {
		return !(*this == rhs);
	}

};

// The data structure representing an extra parameter specification
struct ExtraParam {
	bool isIntegral;    // True iff accepts only integer values
	int decimalPoints;  // Number of decimal points to show/set	

	bool isRanged;      // True iff the value has to be between 
	// rangeMin and rangeMax
	double rangeMin, rangeMax; 
	// NOTE: rangeMin and rangeMax may be -inf 
	// and inf respectively

	bool isAbsolute;    // Convenience setting so that negative values
	// will be automatically turned to positive

	bool canBeInfinite; // True iff value can be infinite

	char name[256];     // The display name of the parameter

	double defaultVal;  // The default value of the parameter

	// Constructor
	ExtraParam(const char *pName = NULL, double defval = 0.0, bool bInf = false, 
		bool bAbs = false, bool bRange = false, double minval = NEGINF, 
		double maxval = POSINF, bool bInt = false, int decPoints = 12) : 

	isIntegral(bInt), decimalPoints(decPoints), isRanged(bRange),
		rangeMin(minval), rangeMax(maxval), isAbsolute(bAbs), 
		canBeInfinite(bInf), defaultVal(defval) {
			if(bInt)
				decimalPoints = 0;
			if(pName)
				strncpy(name, pName, 256);
	}
};

// The data structure passed from the UI to the frontend/backend. 
// Specifies all the parameters of a model.
struct paramStruct {
	// The outer vector represents the layer parameters,
	// while the inner represents the layer itself. i.e., params[i][j]
	// is the ith parameter of the jth layer.
	std::vector< std::vector<Parameter> > params;
	std::vector<Parameter> extraParams;

	Parameter x, y, z, alpha, beta, gamma; // Location and rotation (for domains)
	
	bool bConstrain;
	bool bSpecificUseGrid;

	int layers, nlp;
	int nExtraParams;

	paramStruct(ModelInformation mi) : layers(0), nlp(mi.nlp), 
		nExtraParams(mi.nExtraParams), bConstrain(false), bSpecificUseGrid(true) {}

	paramStruct() : layers(0), nlp(0), nExtraParams(0), bConstrain(false), bSpecificUseGrid(true) {}
	

	paramStruct(const Parameter *paramArray, unsigned int length, int nLayerParams,
				int extraParamCount) : layers(0), nlp(nLayerParams), nExtraParams(extraParamCount), 
									   bConstrain(false) {
		if(!paramArray)
			return;

		unsigned int ctr = 0;

		x = paramArray[ctr++]; y = paramArray[ctr++]; z = paramArray[ctr++];
		alpha = paramArray[ctr++]; beta = paramArray[ctr++]; gamma = paramArray[ctr++];

		bSpecificUseGrid = (paramArray[ctr++].value != 0.0);

		layers = (int)paramArray[ctr++].value;
		
		params.resize(nLayerParams);		
		for(int i = 0; i < nLayerParams; i++) {
			params[i].resize(layers);
			
			for(int j = 0; j < layers; j++)
				params[i][j] = paramArray[ctr++];
		}

		extraParams.resize(extraParamCount);
		for(int i = 0; i < extraParamCount; i++)
			extraParams[i] = paramArray[ctr++];
	}

	/**
	 * Returns the Parameter-array representation of the paramstruct's values for transferring over
	 * the network.
	 * Called with a NULL parameter, returns the length of the corresponding array.
	 * NOTE: This method does NOT allocate "result".
	 * @param result The output array, should be at least the size of the paramstruct
	 *		 (get size by calling with NULL parameter and multiplying by sizeof(Parameter)).
	 * @return The number of elements in the resulting array.
	 **/
	unsigned int Serialize(Parameter *result) const {
		unsigned int resSize = 0;
		unsigned int ctr = 0;

		resSize += 6;
		if(result) { // Location and rotation
			result[ctr++] = x; result[ctr++] = y; result[ctr++] = z;
			result[ctr++] = alpha; result[ctr++] = beta; result[ctr++] = gamma;
		}

		resSize++;
		if(result) { // Use grid at this level
			result[ctr++] = Parameter(bSpecificUseGrid ? 1.0 : 0.0);
		}

		resSize++;
		if(result)
			result[ctr++] = Parameter(layers);

		for(unsigned int i = 0; i < params.size(); i++) {
			resSize += (unsigned int)params[i].size();
			if(result) {
				for(unsigned int j = 0; j < params[i].size(); j++)
					result[ctr++] = params[i][j];
			}
		}

		resSize += (unsigned int)extraParams.size();
		if(result) {
			for(unsigned int i = 0; i < extraParams.size(); i++)
				result[ctr++] = extraParams[i];
		}

		return resSize;
	}

	/**
	 * Returns the parameter vector representation of the paramstruct's values for
	 * use in the fitter.
	 * Called with a NULL parameter, returns the length of the corresponding array.
	 * NOTE: This method does NOT allocate "result".
	 * @param result The output array, should be at least the size of the paramstruct
	 *		 (get size by calling with NULL parameter and multiplying by sizeof(Parameter)).
	 * @return The number of elements in the resulting array.
	 **/
	unsigned int ToParamVector(Parameter *result) const {
		unsigned int resSize = 0;
		unsigned int ctr = 0;

		resSize += 6;
		if(result) { // Location and rotation
			result[ctr++] = x; result[ctr++] = y; result[ctr++] = z;
			result[ctr++] = alpha; result[ctr++] = beta; result[ctr++] = gamma;
		}

		resSize++;
		if(result) { // Use grid at this level
			result[ctr++] = Parameter(bSpecificUseGrid ? 1.0 : 0.0);
		}

		for(unsigned int i = 0; i < params.size(); i++) {
			resSize += (unsigned int)params[i].size();
			if(result) {
				for(unsigned int j = 0; j < params[i].size(); j++)
					result[ctr++] = params[i][j];
			}
		}

		resSize += (unsigned int)extraParams.size();
		if(result) {
			for(unsigned int i = 0; i < extraParams.size(); i++)
				result[ctr++] = extraParams[i];
		}

		return resSize;
	}
};

#define SIZEOFPARAMETERARRAY (Parameter::ELEMENTS)
#define LOCINARR(first, a, i) (first + a * SIZEOFPARAMETERARRAY + i)

/**
 * This class contains the tree structure of the parameters in a hierarchical structure.
 **/
class ParameterTree {
protected:
	// Should we replace the following with std::shared_ptr (Smart Pointers)?
	std::vector<ParameterTree*> children;

	// Node parameters/pointers
	ModelPtr model;
	std::vector<Parameter> nodeParams;	

	// Tree pointers
	ParameterTree *parent;
	unsigned int level;

public:
	ParameterTree() {
		parent = NULL;
		level = 0;
	}

	// Copy constructor. Creates new children for this tree.
	ParameterTree(const ParameterTree& other) : 
		parent(NULL), model(other.model), nodeParams(other.nodeParams), level(0) {
		
		// Copy the children
		children.resize(other.children.size(), NULL);
		for(unsigned int i = 0; i < children.size(); i++) {
			if(other.children[i]) {
				children[i] = new ParameterTree(*other.children[i]);
				children[i]->parent = this;
				children[i]->level = children[i]->parent->GetLevel() + 1;
			}
		}
			

	}

	ParameterTree(ParameterTree* par) {
		parent = par;

		level = this->parent->GetLevel() + 1;
	}

	~ParameterTree() {
		while(!children.empty()) {
			delete children.back();
			children.pop_back();
		}
	}

	ParameterTree &operator=(const ParameterTree& other) { 
		if(&other == this)
			return *this;

		parent = NULL;
		model = other.model;
		nodeParams = other.nodeParams;
		level = 0;
		
		// Delete the current children
		for(unsigned int i = 0; i < children.size(); i++)
			if(children[i])
				delete children[i];

		// Copy the children
		children.resize(other.children.size(), NULL);
		for(unsigned int i = 0; i < children.size(); i++) {
			if(other.children[i]) {
				children[i] = new ParameterTree(*other.children[i]);
				children[i]->parent = this;
			}
		}
			

		return *this;
	}

	// Tree methods
	//////////////////////////////////////////////////////////////////////////
	void PrintTree() {
		std::string spaces;
		spaces.insert(0, 2*level+1, ' ');
		printf("%sModel\t%d\n",spaces.c_str(), model);
		for(int i = 0; i < nodeParams.size(); i++) {
			printf("%sParameter[%d]\t%f\n",spaces.c_str(), i, nodeParams[i].value);
		}
		for(int i = 0; i < children.size(); i++) {
			children[i]->PrintTree();
		}
	}

	unsigned int GetLevel() {
		return level;
	}
	const ParameterTree *GetParent() const {
		return parent;
	}
	ParameterTree *GetParent() {
		return parent;
	}
	ModelPtr GetNodeModel() const {
		return model;
	}
	void SetNodeModel(ModelPtr mod) {
		model = mod;
	}
	paramStruct GetNodeParameters(int nlp, int nExtraParameters) const {
		return paramStruct(&nodeParams[0], 
                                   (unsigned int)(nodeParams.size()), nlp, 
                                   nExtraParameters);
	}
	void SetNodeParameters(const paramStruct& p) {
		nodeParams.resize(p.Serialize(NULL));
		p.Serialize(&nodeParams[0]);		
	}
	int GetNumSubModels() const {
		return (int)children.size();
	}
	const ParameterTree *GetSubModel(int ind) const {
		return children[ind];
	}
	ParameterTree *GetSubModel(int ind) {
		return children[ind];
	}
	ParameterTree *AddSubModel(ModelPtr aMod = (ModelPtr)NULL, const paramStruct &pars = paramStruct()) {
		ParameterTree *res = new ParameterTree(this);

		children.push_back(res);
		res->model = aMod;

		res->nodeParams.resize(pars.Serialize(NULL));
		pars.Serialize(&(res->nodeParams[0]));



		return res;
	}
	
	inline bool operator != (const ParameterTree& rhs) const {
		return !(*this == rhs);
	}

	inline bool operator == (const ParameterTree& rhs) const {
		const ParameterTree* lhs = this;

		// It's better to test the equality manually rather than depending on
		// Serialize, which may be implemented wrong
		/*{ // I think this code works
			std::vector<double> pvr;
			std::vector<double> pvl;
			pvr.resize(rhs.Serialize(), 0.0);
			rhs.Serialize(&pvr[0]);
			pvl.resize(lhs->Serialize(), 0.0);
			lhs->Serialize(&pvl[0]);

			if(pvl.size() == pvr.size()) {
				for(int kk = 0; kk < pvl.size(); kk++) {
					if(fabs(1.0-pvl[kk]/pvr[kk]) > 1.0e-8) {
						return false;
					}
				}
			}
			return true;		
		}*/

		if(lhs->model != rhs.model)
			return false;

		if(lhs->nodeParams.size() != rhs.nodeParams.size()) {
			return false;
		}

		for(unsigned int i = 0; i < lhs->nodeParams.size(); i++ ) {
			if(lhs->nodeParams[i] != rhs.nodeParams[i]) {
				return false;
			}
		}
		
		if(lhs->children.size() != rhs.children.size())
			return false;

		for(unsigned int i = 0; i < lhs->children.size(); i++) {
			// null/not null checks
			if(!lhs->children[i] && rhs.children[i])
				return false;
			if(lhs->children[i] && !rhs.children[i])
				return false;

			if(*lhs->children[i] != *rhs.children[i]) {
				return false;
			}
		}

		return true;
	}

	// Tree to/from parameter vector methods
	//////////////////////////////////////////////////////////////////////////

	/**
	 * Method that takes all the parameters of the entire tree and places it into
	 * a single vector to be used in the fitter.
	 * 
	 * Array Structure:
	 * =================
	 * [# of children][child 1 position]...[child N position][    params   ][@c1p  c1 params ]...[@cNp  cN params]
	 * where params = [x][y][z][alpha][beta][gamma][    actual parameters   ]
	 *
	 * @param arr Pre-allocated double* array of the appropriate size. If NULL, 
	 *            only returns size.
	 * @return int the size of arr (or the size that is should be)
	 **/
	size_t ToParamVector(double *arr = NULL) const {
		// Compute size
		if(!arr) {
			size_t res = 0; 

			res += 1; // Number of children indicator
			res += children.size(); // Children locations
			res += nodeParams.size(); // Actual parameters

			// Size of children
			for(unsigned int i = 0; i < children.size(); i++)
				res += children[i]->ToParamVector();

			return res;
		}

		size_t offset = 0;

		arr[offset] = (double)children.size(); // Number of children indicator
		offset++;

		// Skip children for now
		offset += children.size();

		// Actual parameters
		for(size_t i = 0; i < nodeParams.size(); i++)
			arr[offset + i] = nodeParams[i].value;
		offset += nodeParams.size();

		// Children parameters and locations
		for(size_t i = 0; i < children.size(); i++) {
			arr[1 + i] = (double)offset; // Pointer to beginning of sub-vector
			size_t sz = children[i]->ToParamVector(arr + offset);
			offset += sz;
		}

		return offset;
	}

	/**
	 * Method that takes all the parameter mutability values of the entire tree and places it into
	 * a single vector to be used in the fitter.
	 * 
	 * Array Structure:
	 * =================
	 * [# of children][child 1 position]...[child N position][    params   ][@c1p  c1 params ]...[@cNp  cN params]
	 * where params = [x][y][z][alpha][beta][gamma][    actual parameters   ]
	 *
	 * @param arr Pre-allocated int* array of the appropriate size. If NULL, 
	 *            only returns size.
	 * @return int the size of arr (or the size that is should be)
	 **/
	size_t ToMutabilityVector(int *arr = NULL) const {
		// Compute size
		if(!arr)
			return ToParamVector();

		size_t offset = 0;

		arr[offset] = 0; // Number of children indicator
		offset++;

		// Skip children for now
		offset += children.size();

		// Actual parameters
		for(size_t i = 0; i < nodeParams.size(); i++)
			arr[offset + i] = nodeParams[i].isMutable ? 1 : 0;
		offset += nodeParams.size();

		// Children parameters and locations
		for(size_t i = 0; i < children.size(); i++) {
			arr[1 + i] = 0; // Pointer to beginning of sub-vector
			size_t sz = children[i]->ToMutabilityVector(arr + offset);
			offset += sz;
		}

		return offset;
	}

	enum ConstraintType {
		CT_MINVAL,
		CT_MAXVAL,
		CT_MININD,
		CT_MAXIND,
		CT_LINK,
		CT_SIGMA
	};

	protected:
	static double GetCType(const Parameter& val, ConstraintType type) {
		switch (type)
		{
		default:
		case CT_MINVAL:
			return val.consMin;
		case CT_MAXVAL:
			return val.consMax;
		case CT_MININD:
			return (double)val.consMinIndex;
		case CT_MAXIND:
			return (double)val.consMaxIndex;
		case CT_LINK:
			return (double)val.linkIndex;
		case CT_SIGMA:
			return val.sigma;
		}
	}

	static double GetCDefVal(ConstraintType type) {
		switch (type)
		{
		default:
		case CT_MINVAL:
			return NEGINF;
		case CT_MAXVAL:
			return POSINF;
		case CT_MININD:
		case CT_MAXIND:
		case CT_LINK:
			return -1;
		case CT_SIGMA:
			return 0.0;
		}
	}
	public:

	/**
	 * Method that takes all the parameter constraint values of the entire tree and places it into
	 * a single vector to be used in the fitter.
	 * 
	 * Array Structure:
	 * =================
	 * [# of children][child 1 position]...[child N position][    params   ][@c1p  c1 params ]...[@cNp  cN params]
	 * where params = [x][y][z][alpha][beta][gamma][    actual parameters   ]
	 *
	 * @param arr Pre-allocated int* array of the appropriate size. If NULL, 
	 *            only returns size.
	 * @return int the size of arr (or the size that is should be)
	 **/
	size_t ToConstraintVector(double *arr = NULL, ConstraintType type = CT_MINVAL) const {
		// Compute size
		if(!arr)
			return ToParamVector();

		size_t offset = 0;

		arr[offset] = GetCDefVal(type); // Number of children indicator
		offset++;

		// Skip children for now
		offset += children.size();

		// Actual parameters
		for(unsigned int i = 0; i < nodeParams.size(); i++)
			arr[offset + i] = GetCType(nodeParams[i], type);
		offset += nodeParams.size();

		// Children parameters and locations
		for(unsigned int i = 0; i < children.size(); i++) {
			arr[1 + i] = GetCDefVal(type); // Pointer to beginning of sub-vector
			size_t sz = children[i]->ToConstraintVector(arr + offset, type);
			offset += sz;
		}

		return offset;
	}

    /**
	 * Method that takes the parameter vector from the fitter (obtained by calling ToParamVector)
	 * and modifies this parameter tree's values accordingly.
	 * NOTE: This does NOT create a new parameter tree, only modifies its values!
	 * @param arr The data array
	 * @return True iff succeeded.
	 **/
	bool FromParamVector(const double *arr, size_t sz) {
		// Compute size
		if(!arr)
			return false;

		size_t offset = 0;

		int chl = (int)(arr[0] + 0.1);
		if(children.size() != chl) // Invalid tree
			return false;
		offset++;

		for(int i = 0; i < chl; i++) {
			size_t subsize;

			// Final child requires computation with the total size
			if(i == (chl - 1))
				subsize = sz - (int)(arr[offset + i] + 0.1);
			else
				subsize = (int)(arr[offset + i + 1] + 0.1) - (int)(arr[offset + i] + 0.1);

			// Modify children
			children[i]->FromParamVector(arr + (int)(arr[offset + i] + 0.1), subsize);
		}
		offset += chl;

		// Actual parameters
		for(unsigned int i = 0; i < nodeParams.size(); i++)
			nodeParams[i].value = arr[offset + i];

		return true;
	}

	/**
	 * GetNodeParamVec: Returns the sub-vector that represents the node's
	 * parameters.
	 * @param vec The input main vector.
	 * @param totalsz The size of "vec".
	 * @param sz The resulting size of the sub-vector.
	 * @param bWithModelInformation If true, returns the location and children data.
	 * @return The pointer to the sub-vector, or NULL if the arguments were invalid.
	 **/
	static inline const double *GetNodeParamVec(const double *vec, 
												 const size_t& totalsz, 
												 size_t& sz,
												 bool bWithModelInformation) {		
		if(!vec)
			return NULL;		
		
		int chl = (int)(vec[0] + 0.1);

		// Size of node parameter vector = (position of first child - start position of node parameter vector)
		size_t posFirstChild = 1;
		if(chl > 0)
			posFirstChild = dbltouint(vec[1]);

		sz = posFirstChild - (1 + chl);

		if(bWithModelInformation)
			return (vec + 1 + chl);
		else {
			sz -= 7;
			return (vec + 1 + chl + 7);
		}
	}

	/**
	 * GetChildParamVec: Returns the sub-vector that represents a child model's
	 * parameters.
	 * @param vec The input main vector.
	 * @param totalsz The size of "vec".
	 * @param child The index of the requested child.
	 * @param sz The resulting size of the sub-vector.
	 * @param bWithModelInformation If true, returns the location and children data.
	 * @return The pointer to the sub-vector, or NULL if the arguments were invalid.
	 **/
	static inline const double *GetChildParamVec(const double *vec, 
												 const size_t& totalsz, 
												 int child, size_t& sz,
												 bool bWithModelInformation) {		
		if(!vec)
			return NULL;		
		
		int chl = (int)(vec[0] + 0.1);

		// Verify that the child is correct
		if(child < 0 || child >= chl)
			return NULL;

		// Get child size
		if(child == (chl - 1))
			sz = totalsz - (int)(vec[1 + child] + 0.1);
		else
			sz = (int)(vec[1 + child + 1] + 0.1) - (int)(vec[1 + child] + 0.1);

		// Return child
		if(bWithModelInformation)
			return (vec + (int)(vec[1 + child] + 0.1));
		else { // Return child without location and number of children
			// #children + [each child's pointer] + [x,y,z,alpha,beta,gamma,bUseGrid]			
			size_t offset = 1 + dbltouint(vec[dbltouint(vec[1 + child])]) + 7;

			// If a model has children they will still be included in "sz"
			sz -= offset;

			return (vec + dbltouint(vec[1 + child]) + offset);
		}
	}

	static inline int GetChildNumChildren(const double *vec, int child) {
		if(!vec)
			return -1;		

		int chl = dbltoint(vec[0]);

		// Verify that the child is correct
		if(child < 0 || child >= chl)
			return -2;

		// #children + [each child's pointer] + [x,y,z,alpha,beta,gamma]-->bUseGrid?
		size_t offset = dbltouint(vec[1 + child]);		

		return dbltoint(vec[offset]);
	}

	static inline double GetChildNLayers(const double *vec, int child) {
		if(!vec)
			return -1.0;		

		int chl = (int)(vec[0] + 0.1);
		 
		// Verify that the child is correct
		if(child < 0 || child >= chl)
			return -2.0;

		size_t offset = dbltouint(vec[1 + child]);
		// #children + [each child's pointer] + [x,y,z,alpha,beta,gamma,bUseGrid]	
		offset += 1 + dbltouint(vec[dbltouint(vec[1 + child])]) + 7;

		return vec[offset];
	}

	static inline void GetChildLocationData(const double *vec, int child,
											LocationRotation& locrot
											)
	{
		if(!vec)
			return;		

		int chl = (int)(vec[0] + 0.1);
		 
		// Verify that the child is correct
		if(child < 0 || child >= chl)
			return;

		size_t offset = dbltouint(vec[1 + child]);
		// #children + [each child's pointer]
		offset += 1 + dbltouint(vec[dbltouint(vec[1 + child])]);

		// Assign the values
		locrot.x = vec[offset++]; locrot.y = vec[offset++]; locrot.z = vec[offset++];
		locrot.alpha = Radian(vec[offset++]); locrot.beta = Radian(vec[offset++]);
		locrot.gamma = Radian(vec[offset++]);
	}

	static inline void GetChildUseGrid(const double *vec, int child,
		bool& bUseGridFromHere) {
			if(!vec)
				return;		

			int chl = (int)(vec[0] + 0.1);

			// Verify that the child is correct
			if(child < 0 || child >= chl)
				return;

			size_t offset = dbltouint(vec[1 + child]);
			// #children + [each child's pointer]
			offset += 1 + dbltouint(vec[dbltouint(vec[1 + child])]);
			// Location variables
			offset += 6;
			// Assign the value
			bUseGridFromHere = vec[offset] != 0;
	}

	/** 
	 * Helper macro for Serialize that writes a Parameter to the array.
	 **/
#define SERIALIZE_PARAMETER(arr, offset, par)				\
	do {													\
		arr[offset + 0] = par.value;						\
		arr[offset + 1] = par.isMutable ? 1.0 : 0.0;		\
		arr[offset + 2] = par.isConstrained ? 1.0 : 0.0;	\
		arr[offset + 3] = par.consMin;						\
		arr[offset + 4] = par.consMax;						\
		arr[offset + 5] = (double)par.consMinIndex;			\
		arr[offset + 6] = (double)par.consMaxIndex;		    \
		arr[offset + 7] = (double)par.linkIndex;		    \
		arr[offset + 8] = par.sigma;					    \
		offset += Parameter::ELEMENTS;						\
	} while(0)

	/** 
	 * Helper macro for Deserialize that reads a Parameter from the array.
	 **/
#define DESERIALIZE_PARAMETER(arr, offset, par)									\
	do {																		\
		par = Parameter(arr[offset + 0], (fabs(arr[offset + 1] - 1.0) <= 1e-6),	\
						(fabs(arr[offset + 2] - 1.0) <= 1e-6), arr[offset + 3],	\
						arr[offset + 4], dbltoint(arr[offset + 5]),				\
						dbltoint(arr[offset + 6]), dbltoint(arr[offset + 7]),	\
						arr[offset + 8]);										\
		offset += Parameter::ELEMENTS;											\
	} while(0)

	/**
	 * Method that takes all the parameters of the entire tree and places it into
	 * a single pre-allocated double array that can be transfered and used to 
	 * rebuild an equivalent tree.
	 * @param vec Pre-allocated double* array of the appropriate size
	 * @return int the size of vec (or the size that is should be)
	 
	size_t Serialize(double *arr = NULL) const {
		// Compute size
		if(!arr) {
			size_t res = 0; 

			res += 1; // Model pointer
			res += 1; // Number of children indicator
			res += children.size(); // Children locations
						
			res += nodeParams.size() * Parameter::ELEMENTS; // Actual parameters

			// Size of children
			for(unsigned int i = 0; i < children.size(); i++)
				res += children[i]->Serialize();

			return res;
		}

		size_t offset = 0;

		arr[offset] = (double)model; // Model pointer
		offset++;

		arr[offset] = (double)children.size(); // Number of children indicator
		offset++;

		// Skip children for now
		offset += children.size();

		// Actual parameters
		for(unsigned int i = 0; i < nodeParams.size(); i++)
			SERIALIZE_PARAMETER(arr, offset, nodeParams[i]);

		// Children parameters and locations
		for(size_t i = 0; i < children.size(); i++) {
			arr[2 + i] = (double)offset; // Pointer to beginning of sub-vector
			size_t sz = children[i]->Serialize(arr + offset);
			offset += sz;
		}

		return offset;
	}**/

	void Deserialize(const std::vector<double> &pars) {		
		Deserialize(&pars[0], pars.size());
	}

	bool Deserialize(const double *arr, size_t sz) {
		// Compute size
		if(!arr)
			return false;

		size_t offset = 0;
		
		model = (ModelPtr)dbltouint(arr[offset]);
		offset++;

		// Clear existing children and parameters first
		nodeParams.clear();
		for(unsigned int i = 0; i < children.size(); i++)
			delete children[i];
		children.clear();
		
		unsigned int chl = dbltouint(arr[offset]);
		offset++;

		size_t firstChildOffset = offset;

		for(unsigned int i = 0; i < chl; i++) {
			size_t subsize;

			// Final child requires computation with the total size
			if(i == (chl - 1))
				subsize = sz - (int)(arr[offset + i] + 0.1);
			else
				subsize = (int)(arr[offset + i + 1] + 0.1) - (int)(arr[offset + i] + 0.1);

			// Re-create children
			this->AddSubModel(0);
			children[i]->Deserialize(arr + (int)(arr[offset + i] + 0.1), subsize);
		}
		offset += chl;

		// Actual parameters
		size_t numElements = 0;
		if(chl == 0) // There are no children, the number of elements is determined
					 // by the total size of the array
			numElements = (sz - offset) / Parameter::ELEMENTS;
		else // Children determine where the parameters end
			numElements = (size_t(arr[firstChildOffset] + 0.1) - offset) / Parameter::ELEMENTS;

		nodeParams.resize(numElements);
		for(unsigned int i = 0; i < nodeParams.size(); i++)
			DESERIALIZE_PARAMETER(arr, offset, nodeParams[i]);			

		return true;
	}

	/* New functions for JSON serialization (which is done in another class) */
	int GetNumParameters() const {
		return (int)nodeParams.size();
	}
	const Parameter &GetParameter(int ind) const {
		return nodeParams.at(ind);
	}

	void AddSubModel(const ParameterTree &pt) {
		ParameterTree *copy = new ParameterTree(pt);
		children.push_back(copy);
	}
	void AddParameter(const Parameter &param) {
		nodeParams.push_back(param);
	}
};

#define LOSS_FUNCTION(X)			\
	X(TRIVIAL_LOSS,		"Trivial Loss",		0)	\
	X(HUBER_LOSS,		"Huber Loss",		1)	\
	X(SOFTLONE_LOSS,	"Soft L One Loss",	1)	\
	X(CAUCHY_LOSS,		"Cauchy Loss",		1)	\
	X(ARCTAN_LOSS,		"Arctan Loss",		1)	\
	X(TOLERANT_LOSS,	"Tolerant Loss",	2)
							

CREATE_ENUM_CSTRING_AND_NUMPARAMS(LossFunction, LOSS_FUNCTION);

#define MINIMIZER_TYPE_ENUM_DEFINE(X)	\
	X(LINE_SEARCH,	"Line Search",	2)	\
	X(TRUST_REGION,	"Trust Region",	1)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(MinimizerType, MINIMIZER_TYPE_ENUM_DEFINE);

#define LINE_SEARCH_DIRECTION_TYPE_DEFINE(X)			\
	X(STEEPEST_DESCENT,				"Steepest Descent",				0)	\
	X(NONLINEAR_CONJUGATE_GRADIENT,	"Nonlinear Conjugate Gradient",	1)	\
	X(LBFGS,						"L-BFGS",						0)	\
	X(BFGS,							"BFGS",							0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(LineSearchDirectionType, LINE_SEARCH_DIRECTION_TYPE_DEFINE);

#define LINE_SEARCH_TYPE_DEFINE(X)			\
X(ARMIJO,	"Armijo",	0)					\
X(WOLFE,	"Wolfe",	0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(LineSearchType, LINE_SEARCH_TYPE_DEFINE);

#define TRUST_REGION_STRATEGY_TYPE_DEFINE(X)			\
	X(LEVENBERG_MARQUARDT, "Levenberg-Marquardt", 0)	\
	X(DOGLEG, "Dogleg", 0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(TrustRegionStrategyType, TRUST_REGION_STRATEGY_TYPE_DEFINE);

#define DOGLEG_TYPE_DEFINE(X)							\
	X(TRADITIONAL_DOGLEG,	"Traditional Dogleg",	0)	\
	X(SUBSPACE_DOGLEG,		"Subspace Dogleg",		0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(DoglegType, DOGLEG_TYPE_DEFINE);

#define NONLINEAR_CONJUGATE_GRADIENT_TYPE(X)	\
	X(FLETCHER_REEVES,	"Fletcher Reeves",	0)	\
	X(POLAK_RIBIRERE,	"Polak Ribirere",	0)	\
	X(HESTENES_STIEFEL,	"Hestenes Stiefel",	0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(NonlinearConjugateGradientType, NONLINEAR_CONJUGATE_GRADIENT_TYPE);


#define XRAY_RESIDUALS_TYPES(X)					\
	X(NORMAL_RESIDUALS,	"Normal Residuals",	0)	\
	X(RATIO_RESIDUALS,	"Ratio Residuals",	0)	\
	X(LOG_RESIDUALS,	"Log Residuals",	0)

CREATE_ENUM_CSTRING_AND_NUMPARAMS(XRayResidualsType, XRAY_RESIDUALS_TYPES);

struct CeresProperties
{
	/// Ceres parameters
	MinimizerType_Enum					minimizerType = TRUST_REGION;
	LineSearchDirectionType_Enum		lineSearchDirectionType = LBFGS;
	LineSearchType_Enum					lineSearchType = WOLFE;
	TrustRegionStrategyType_Enum		trustRegionStrategyType = LEVENBERG_MARQUARDT;
	DoglegType_Enum						doglegType = TRADITIONAL_DOGLEG;
	NonlinearConjugateGradientType_Enum	nonlinearConjugateGradientType = FLETCHER_REEVES;
	LossFunction_Enum					lossFuncType = TRIVIAL_LOSS;
	double								lossFunctionParameters[2]
// MS hasn't yet implemented C++11 initializers
#if (defined(_MSC_VER ) && _MSC_VER < 1900) || (!defined(_MSC_VER) && __cplusplus < 201103L)
		;
#else
		= { 0.5, 0.5 };
#endif
	XRayResidualsType_Enum				residualType					= NORMAL_RESIDUALS;

	// Non-ceres specific parameters

	/// This is the maximum fraction step size used when calculating an
	/// adaptive derivative
	double derivativeStepSize = 0.01;
	/// Instead of waiting for the derivative/gradient to be 0, when the
	/// cost function reaches this, call it a day (or a fit)
	double fittingConvergence = 0.1;
	/// 
	double derivativeEps = 0.1;
};

struct FittingProperties {
	// Fitting properties
	double minSignal;
	double resolution;
	int fitIterations;
	FitMethod method;
	bool usingGPUBackend;

	CeresProperties ceresProps;

	// Fitting accuracy settings
	bool accurateFitting;
	bool accurateDerivative;
	bool wssrFitting;
	bool logScaleFitting;

	bool liveFitting;
	bool bProgressReport;

	// Progress update interval (in milliseconds)
	int msUpdateInterval;

	// Controls live graph data generation
	bool liveGenerate;

	int orientationIterations;
	OAMethod_Enum orientationMethod;
	
	int gridSize;

	bool bUseGrid;
};

#pragma pack(pop)

typedef void (*previewRenderFunc)();
typedef void (*renderFunc)(const paramStruct& p, const EDProfile& profile, LevelOfDetail lod, bool bColorCoded);

// Returns an array of structs corresponding to the locations and rotations of the underlying 
// objects, relative to the symmetry itself (i.e., 0,0,0,0,0,0 doesn't change a thing)
typedef std::vector<LocationRotation> (*symmetryRenderFunc)(const paramStruct& p);

#endif

