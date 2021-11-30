#ifndef __MODELUI_H
#define __MODELUI_H

#pragma once
#include <string>
#include <vector>
#include "CommProtocol.h"
// #undef min
// #undef max
#include "Eigen/Core" // For ArrayXXi

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

using std::string;
using std::vector;
using Eigen::ArrayXXi;
using Eigen::MatrixXd;

// (Later, if necessary): A C-wrapper for both ModelUI and ModelRenderer 
//                             (and every other class and/or non-C function)
// Must be STRICT C (no std::vectors and such) in order to support multiple UI
// languages (C++, C#, Java)

EXPORTED enum EXTRA_PARAM_TYPE {
	EPT_DOUBLE,
	EPT_INTEGER,
	EPT_MULTIPLE_CHOICE,
	EPT_CHECKBOX,
	EPT_PLACEHOLDER
};

class EXPORTED ModelRenderer {
protected:	
	renderFunc renderFunction;
	previewRenderFunc previewFunction;
	symmetryRenderFunc symmetryFunction;

public:
	/**
	 * ModelRenderer: Creates a model renderer from the container's name and model index.
	 *                NOTE: Seeks renderers in: 1. The model container itself
	 *                                          2. Same filename with .xrn extension
	 *
	 * @param container The container name (no extension)
	 * @param index The model's index
	 **/
	ModelRenderer(const wchar_t *container, int index);
	
	/**
	 * GetRenderer: Returns the model render function for use inside the fitting window
	 *
	 * @return The model render function
	 **/
	renderFunc GetRenderer();

	/**
	 * GetPreview: Returns the model's preview render for use in the opening window
	 *
	 * @return The model preview render function
	 **/
	previewRenderFunc GetPreview();

	/**
	 * GetSymmetryRenderer: Returns the symmetry render function that specifies
	 *                      where the sub-amplitudes reside.
	 *
	 * @return The symmetry render function
	 **/
	symmetryRenderFunc GetSymmetryRenderer();
};

/**
 * The ModelUI class retrieves model layer/parameter information for a 
 * single ModelType from the local or remote backend
 */
class EXPORTED ModelUI {
protected:
	// TODO::EDP Add GetEDProfile
	EDProfile edp;
	std::vector<string> layerNames, layerParamNames;
	std::vector<string> displayParamNames;
	ModelInformation mi;
	std::vector<string> relatedModelNames;
	std::vector<int> relatedModelIndices;
	int relatedModels;
	ModelCategory mc;
	std::string modelName;
	Eigen::ArrayXXi isParamApplicable;
	Eigen::MatrixXd defVals;
	std::vector<ExtraParam> extraParamsInfo;
	std::vector<EXTRA_PARAM_TYPE> extraParamsTypes;
	std::vector<std::vector<string>> extraParamOptions;
	// Communication variables
	FrontendComm *_com;
	ModelInformation _info;
	wchar_t *_container;
	wchar_t _containerstr[MAX_PATH];

	/**
	 * Retrieves model information using existing communication variables
	 * from backend and stores a local copy.
	 * 
	 * @param startInd First layer index to retrieve [startInd, endInd]
	 * @param endInd Last layer index to retrieve [startInd, endInd]
	 * @return true if successful, false otherwise.
	 */
	virtual bool retrieveMoreLayers(int startInd, int endInd);

	virtual int GetNumRetrievedLayers();

public:

	~ModelUI();
	
	ModelUI();

	/**
	 * Retrieves model information from backend and stores a local copy.
	 * @param *com FrontendComm to communicate with backend.
	 * @param job JobPtr indicating which job the request is for.
	 * @param type Which model type within the job the request is for.
	 *					Cannot be multiple types.
	 * @param info ModelInformation of the model requested.
	 * @param endInd Number (-1) of layers to retrieve [0, endInd]
	 * @return true if successful, false otherwise.
	 */
	virtual bool setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd);

	virtual bool IsParamApplicable(int layer, int iPar);

	virtual string GetLayerParamName(int index);

	virtual string GetLayerName(int index);

	virtual double GetDefaultParamValue(int layer, int iPar);

	virtual ExtraParam GetExtraParameter(int index);

	virtual int GetNumLayerParams();

	virtual int GetNumExtraParams();

	virtual int GetNumDisplayParams();

	virtual string GetDisplayParamName(int index);

	virtual EDProfile GetEDProfile();
	
	virtual string GetName();

	virtual bool IsLayerBased();

	virtual bool IsSlow();

	virtual int GetMaxLayers();

	virtual int GetMinLayers();

	virtual int GetNumRelatedModels();

	virtual string GetRelatedModelName(int index);
	
	virtual int GetRelatedModelIndex(int index);

	virtual wchar_t *GetContainer(wchar_t *res);

	virtual ModelInformation GetModelInformation();

	virtual EXTRA_PARAM_TYPE GetExtraParamType(int ind);

	virtual std::vector<std::string> GetExtraParamOptionStrings(int ind);

 };

class EXPORTED PDBModelUI : public ModelUI {
protected:
	// Does nothing here
	virtual bool retrieveMoreLayers(int startInd, int endInd) { return true; }
	virtual int GetNumRetrievedLayers() { return 0; }

public:
	PDBModelUI(const char *tName);

	~PDBModelUI();
	string GetLayerParamName(int index);
	string GetLayerName(int index);
	bool setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd);
};	

class EXPORTED AMPModelUI : public ModelUI {
protected:
	// Does nothing here
	virtual bool retrieveMoreLayers(int startInd, int endInd) { return true; }
	virtual int GetNumRetrievedLayers() { return 0; }

public:
	AMPModelUI(const char *tName);

	~AMPModelUI() {};
	string GetLayerParamName(int index);
	string GetLayerName(int index);
	bool setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd);
};	

typedef const char * (__stdcall *GetNameFunc)(int index);
typedef double (__stdcall *GetDefaultParamValueFunc)(int row, int col);
typedef double (__stdcall *GetDefaultExtraParamValueFunc)(int index);
typedef bool (__stdcall *GetParamApplicableFunc)(int row, int col);

// Handles scripted model UI
class EXPORTED ScriptedModelUI : public ModelUI {
protected:
	GetNameFunc layernameFunc, layerparamnameFunc, extraParamNameFunc;
	GetDefaultParamValueFunc dpvFunc;
	GetParamApplicableFunc paFunc;
	GetDefaultExtraParamValueFunc extraParamValueFunc;

	// Does nothing here
	virtual bool retrieveMoreLayers(int startInd, int endInd) { return true; }
	virtual int GetNumRetrievedLayers() { return 0; }

public:

	~ScriptedModelUI();
	
	ScriptedModelUI();

	virtual void SetHandlerCallbacks(GetNameFunc layername, GetNameFunc layerparam, 
									 GetDefaultParamValueFunc dpv, GetParamApplicableFunc ipa);

	/**
	 * Retrieves model information from script. ONLY USES info
	 * @param info ModelInformation of the model requested.
	 * @return true if successful, false otherwise.
	 */
	virtual bool setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd);

	virtual bool IsParamApplicable(int layer, int iPar);

	virtual string GetLayerParamName(int index);

	virtual string GetLayerName(int index);

	virtual double GetDefaultParamValue(int layer, int iPar);

	virtual ExtraParam GetExtraParameter(int index);	

	virtual int GetNumExtraParams();

	virtual int GetNumDisplayParams();

	virtual string GetDisplayParamName(int index);

	virtual int GetNumRelatedModels();

	virtual string GetRelatedModelName(int index);
	
	virtual int GetRelatedModelIndex(int index);
 };


#endif
