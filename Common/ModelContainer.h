#ifndef __MODEL_CONTAINER_H
#define __MODEL_CONTAINER_H

/////////////////////////////////////////////////////////////////////
// Prelude:                                                        //
// This .h file has to be implemented by every DLL model container //
/////////////////////////////////////////////////////////////////////

#include "Common.h" // For ModelInformation

// Forward declarations
class IModel;
 
#undef EXPORTED
 #ifdef _WIN32
   #ifdef EXPORTER
     #define EXPORTED __declspec(dllexport)
   #else
     #define EXPORTED __declspec(dllimport)
   #endif
 #else
   #ifdef EXPORTER
     #define EXPORTED __attribute__ ((visibility ("default")))
   #else
     #define EXPORTED
   #endif
 #endif

#ifdef _WIN32
	// Since std::string is a C++ type and we are exporting C-type declarations,
	// we disable the "C++ type in C declaration" warning (so, so hacky)
	#pragma warning(push)
	#pragma warning(disable: 4190)
#endif


#ifdef __cplusplus    // If used by C++ code, 
extern "C" {          // we need to export the C interface
#endif
	
enum InformationProcedure {
	IP_UNKNOWN = 0,
	
	IP_LAYERPARAMNAME,
	IP_LAYERNAME,
	IP_EXTRAPARAMETER,
	IP_DEFAULTPARAMVALUE,
	IP_ISPARAMAPPLICABLE,
	IP_DISPLAYPARAMNAME,
	IP_DISPLAYPARAMVALUE,

	IP_UNKNOWN2,
};

// Information procedure definition
class EDPFunction;
typedef std::string (*fGetLayerParamName)(int index, EDPFunction *edpfunc);
typedef std::string (*fGetLayerName)(int layer);
typedef ExtraParam (*fGetExtraParameter)(int index);
typedef double (*fGetDefaultParamValue)(int paramIndex, int layer, EDPFunction *edpfunc);
typedef bool (*fIsParamApplicable)(int layer, int lpindex);
typedef std::string (*fGetDisplayParamName)(int index);
typedef double (*fGetDisplayParamValue)(int index, const paramStruct *p);


// Methods


// Returns the number of model categories in this container
EXPORTED int GetNumCategories();

// Returns the relevant category information, or "N/A" if index is invalid.
EXPORTED ModelCategory GetCategoryInformation(int catInd);

// Returns the number of models in this container
EXPORTED int GetNumModels();

// Returns the model's information structure from its index.
EXPORTED ModelInformation GetModelInformation(int index);

// Returns the model object that matches the index. Supposed to return NULL
// for indices that are out of bounds
EXPORTED IModel *GetModel(int index);

// TODO::OneStepAhead: Replace this with a script for the frontend
// Returns the requested information retrieval procedure
EXPORTED void *GetModelInformationProcedure(int index, InformationProcedure type);

#ifdef __cplusplus
}
#endif

#ifdef _WIN32
	#pragma warning(pop)
#endif

#endif
