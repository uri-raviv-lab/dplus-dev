#include "ModelUI.h"

/**
 * @file ModelUI.cpp
 * @brief Implements the ModelUI, ScriptedModelUI, PDBModelUI, and AMPModelUI classes, which provide
 *        frontend model metadata, parameter management, and UI integration for model selection and configuration.
 *
 * The ModelUI module is responsible for:
 *  - Managing model metadata and parameter information for frontend display and editing.
 *  - Providing access to layer and extra parameter names, types, default values, and applicability.
 *  - Supporting dynamic retrieval of layer information from the backend as needed.
 *  - Handling related models, display parameters, and container associations for each model.
 *  - Supporting specialized UI models for scripted, PDB, and amplitude-only models.
 *
 * Key Concepts:
 *  - ModelUI: Main class for representing a model's UI metadata, parameter names, types, and values.
 *  - ScriptedModelUI: UI class for models defined by scripts, with callback-based parameter and layer naming.
 *  - PDBModelUI: UI class for PDB-based models, with fixed extra parameters and options.
 *  - AMPModelUI: UI class for amplitude-only models, with minimal parameterization.
 *  - Dynamic Layer Retrieval: Layer names and parameter applicability are fetched from the backend as needed.
 *  - Extra Parameters: Support for different types (double, checkbox, multiple choice) and options.
 *
 * Fields:
 *  - ModelInformation mi:
 *      Holds model metadata (name, index, parameter counts, flags, etc.).
 *  - std::vector<std::string> layerParamNames:
 *      Names of layer parameters for the model.
 *  - std::vector<std::string> layerNames:
 *      Names of layers for the model.
 *  - std::vector<ExtraParam> extraParamsInfo:
 *      Information about extra (non-layer) parameters.
 *  - std::vector<EXTRA_PARAM_TYPE> extraParamsTypes:
 *      Types of extra parameters (e.g., EPT_DOUBLE, EPT_CHECKBOX, EPT_MULTIPLE_CHOICE).
 *  - std::vector<std::vector<std::string>> extraParamOptions:
 *      Option strings for extra parameters (for multiple choice, etc.).
 *  - std::vector<std::string> displayParamNames:
 *      Names of display parameters for the model.
 *  - ArrayXXi isParamApplicable:
 *      Matrix indicating applicability of each parameter to each layer.
 *  - MatrixXd defVals:
 *      Matrix of default parameter values for each layer and parameter.
 *  - int relatedModels:
 *      Number of related models.
 *  - std::vector<std::string> relatedModelNames:
 *      Names of related models.
 *  - std::vector<int> relatedModelIndices:
 *      Indices of related models.
 *  - EDProfile edp:
 *      Electron density profile information for the model.
 *  - FrontendComm* _com:
 *      Pointer to the frontend communication interface.
 *  - const wchar_t* _container:
 *      Pointer to the container name for the model.
 *  - wchar_t _containerstr[MAX_PATH]:
 *      Buffer for storing the container name.
 *  - ModelCategory mc:
 *      Category information for the model.
 *  - (ScriptedModelUI) Function pointers for custom naming and parameter logic:
 *      - GetNameFunc layernameFunc, layerparamnameFunc
 *      - GetDefaultParamValueFunc dpvFunc
 *      - GetParamApplicableFunc paFunc
 *
 * Main Methods:
 *  - setModel: Initializes the UI model with backend and metadata information.
 *  - retrieveMoreLayers: Dynamically fetches additional layer information from the backend.
 *  - GetLayerParamName / GetLayerName / GetExtraParameter: Accessors for parameter and layer names/info.
 *  - GetNumLayerParams / GetNumExtraParams / GetNumDisplayParams: Accessors for parameter counts.
 *  - GetDefaultParamValue / IsParamApplicable: Accessors for parameter values and applicability.
 *  - GetRelatedModelName / GetRelatedModelIndex: Accessors for related model information.
 *  - GetContainer: Returns the container name for the model.
 *  - ScriptedModelUI::SetHandlerCallbacks: Sets custom callbacks for scripted model UI logic.
 *
 * Threading and Safety:
 *  - Not inherently thread-safe; designed for use in the frontend UI thread.
 *  - Dynamic layer retrieval and backend communication are performed synchronously.
 *
 * Error Handling:
 *  - Returns false on backend communication errors or invalid indices.
 *  - Catches and handles backend errors when retrieving parameter or model information.
 *
 * Dependencies:
 *  - ModelUI.h for class and method declarations.
 *  - FrontendComm for backend communication.
 *  - ModelInformation, ExtraParam, EDProfile, and related data structures.
 *  - Eigen for matrix and array types (MatrixXd, ArrayXXi).
 *
 * See ModelUI.h for class and method declarations.
 */

ModelUI::~ModelUI() {
}

ExtraParam ModelUI::GetExtraParameter(int index) {
	return extraParamsInfo[index];
}

string ModelUI::GetLayerParamName(int index) {
	return layerParamNames[index];
}

string ModelUI::GetLayerName(int index) {
	if(index >= (int)layerNames.size()) {
		retrieveMoreLayers(GetNumRetrievedLayers(), 2 * GetNumRetrievedLayers());
		//return string("Need to retrieve more layers from backend");
	//	retrieveMoreLayers()
	}
	return layerNames[index];
}

int ModelUI::GetNumRetrievedLayers() {
	return (int)layerNames.size();
}

int ModelUI::GetNumExtraParams() {
	return mi.nExtraParams;
}

int ModelUI::GetNumLayerParams() {
	return mi.nlp;
}

string ModelUI::GetDisplayParamName(int index) {
	return displayParamNames[index];
}

bool ModelUI::retrieveMoreLayers(int startInd, int endInd) {
	if(_com) {
		ErrorCode err = OK;
		int nlp = _info.nlp;

		int *app = new int[nlp];
		double *dVals = new double[nlp];

		// Layer names, parameter applicability, parameter default values
		MatrixXd dVtmp = defVals;
		ArrayXXi iPAtmp = isParamApplicable;

		defVals.resize(nlp, endInd+1);
		isParamApplicable.resize(nlp, endInd+1);
		layerNames.resize(endInd+1);

		if (dVtmp.cols() > 0) {
			defVals.block(0,0, nlp,dVtmp.cols()) = dVtmp;
			isParamApplicable.block(0,0, nlp,iPAtmp.cols()) = iPAtmp;
		}

		for(int ind = startInd; ind <= endInd && !err; ind++) {
			char name[256] = {0};
			err = _com->GetLayerInfo(_container, mi.modelIndex, ind, name, app, dVals, nlp);
			if(err)
				break;

			layerNames[ind] = std::string(name);
			for(int i = 0; i < nlp; i++) {
				isParamApplicable(i,ind) = app[i];
				defVals(i,ind) = dVals[i];
			}
		}

		delete[] app;
		delete[] dVals;

		return true;
	}

	return false;
}

bool ModelUI::setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd ) {
	_com = com;
	_info = info;
	if(container) {
		wcsncpy(_containerstr, container, MAX_PATH);
		_container = _containerstr;
	} else
		_container = NULL;

	// Hack to render asymmetric models correctly. I'm not sure where we would want to
	// obtain this info from (presumable the backend, although there is no reason the
	// backend should know about rendering...
	const std::string t("Asymmetr");
	if (std::string(info.name).compare(0, t.length(), t) == 0)
		edp.type = ASYMMETRIC;

	if(com) {
		ErrorCode err = OK;

		mi = info;
		int nlp = mi.nlp;

		char **lpNames = new char*[nlp];
		for(int i = 0; i < nlp; i++)
			lpNames[i] = new char[256];

		// Layer parameter names
		err = com->GetLayerParamNames(_container, mi.modelIndex, lpNames, nlp);
		if(err == OK) {
			layerParamNames.clear();
			for(int i = 0; i < nlp; i++)
				layerParamNames.push_back(std::string(lpNames[i]));

			// Layer names, default values, applicability
			err = retrieveMoreLayers(0, endInd) ? OK : ERROR_GENERAL;
		}

		if(!err && mi.nExtraParams > 0) {
			ExtraParam *ep = new ExtraParam[mi.nExtraParams];

			// Extra parameters
			err = com->GetExtraParamInfo(_container, mi.modelIndex, ep, mi.nExtraParams);
			if(!err) {
				extraParamsInfo.clear();
				for (int i = 0; i < mi.nExtraParams; i++)
					extraParamsInfo.push_back(ep[i]);
			}

			delete[] ep;

			extraParamsTypes.resize(mi.nExtraParams, EPT_DOUBLE);
			extraParamOptions.resize(mi.nExtraParams, std::vector<std::string>());
			// TODO Talk to backend to determine how these should be filled.
		}

		if (!err && mi.nDispParams > 0) {
			int nDisplayParams = mi.nDispParams;
			char **dpsNames = new char*[nDisplayParams];
			for(int i = 0; i < nDisplayParams; i++)
				dpsNames[i] = new char[256];
			err = com->GetDisplayParamInfo(_container, mi.modelIndex, 
												dpsNames, nDisplayParams);

			for(int i = 0; i < nDisplayParams; i++)
				displayParamNames.push_back(std::string(dpsNames[i]));

			for(int i = 0; i < nDisplayParams; i++)
				delete[] dpsNames[i];
			delete[] dpsNames;
		}

		for(int i = 0; i < nlp; i++)
			delete[] lpNames[i];
		delete[] lpNames;

		if(!err) {
			mc = com->QueryCategory(container, mi.category);
			for(relatedModels = 0; relatedModels < 16; relatedModels++) {
				if(mc.models[relatedModels] == -1)
					break;
				if(mc.models[relatedModels] == mi.modelIndex)
					continue;
				ModelInformation tMi = com->QueryModel(container, relatedModels);
				relatedModelNames.push_back(string(tMi.name));
				relatedModelIndices.push_back(tMi.modelIndex);
			}
			// Removing the model itself from the count
			relatedModels--;
		}

		return (err == OK);
	}

	return false;
}

ModelUI::ModelUI() {
}

string ModelUI::GetName() {
	return string(mi.name);
}

bool ModelUI::IsParamApplicable(int layer, int iPar) {
	if(layer >= (int)layerNames.size())
		retrieveMoreLayers(GetNumRetrievedLayers(), 2 * GetNumRetrievedLayers());
	return (isParamApplicable(iPar, layer) == 1);
}

EDProfile ModelUI::GetEDProfile() {
	return edp;	// TODO::EDP
}

bool ModelUI::IsLayerBased() {
	return mi.isLayerBased;
}

bool ModelUI::IsSlow() {
	return mi.isSlow;
}

int ModelUI::GetMaxLayers() {
	return mi.maxLayers;
}

int ModelUI::GetMinLayers() {
	return mi.minLayers;
}

int ModelUI::GetNumRelatedModels() {
	return relatedModels;
}

string ModelUI::GetRelatedModelName(int index) {
	return relatedModelNames[index];
}

wchar_t *ModelUI::GetContainer(wchar_t *res) {
	if(res && _container)
		wcsncpy(res, _container, MAX_PATH);

	if(!_container)
		return NULL;

	return res;
}

int ModelUI::GetRelatedModelIndex(int index) {
	return relatedModelIndices[index];
}

double ModelUI::GetDefaultParamValue(int layer, int iPar) {
	if(layer >= (int)layerNames.size())
		retrieveMoreLayers(GetNumRetrievedLayers(), 2 * GetNumRetrievedLayers());
	return defVals(iPar, layer);
}

int ModelUI::GetNumDisplayParams() {
	return (int)displayParamNames.size();
}

ModelInformation ModelUI::GetModelInformation() {
	return mi;
}

EXTRA_PARAM_TYPE ModelUI::GetExtraParamType(int ind) {
	return extraParamsTypes[ind];
}

std::vector<std::string> ModelUI::GetExtraParamOptionStrings(int ind) {
	return extraParamOptions[ind];
}

//////////////////////////////////////////////////////////////////////////

ScriptedModelUI::~ScriptedModelUI() {
}

ScriptedModelUI::ScriptedModelUI() {
	layernameFunc = NULL;
	layerparamnameFunc = NULL;
	dpvFunc = NULL;
	paFunc = NULL;
	extraParamNameFunc = NULL;
	extraParamValueFunc = NULL;

	extraParamsTypes.resize(1, EPT_DOUBLE);	// scale

}

bool ScriptedModelUI::setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd) {
	// Do nothing except this
	mi = info;
	return true;
}

bool ScriptedModelUI::IsParamApplicable(int layer, int iPar) {
	if(!paFunc)
		return true;

	return paFunc(layer, iPar);
}

string ScriptedModelUI::GetLayerParamName(int index) {
	if(!layerparamnameFunc) {
		std::stringstream ss;
		ss << "Parameter " << (index + 1);	
		return ss.str();
	}

	return layerparamnameFunc(index);
}

string ScriptedModelUI::GetLayerName(int index) {
	if(!layernameFunc) {
		std::stringstream ss;
		ss << "Layer " << (index + 1);	
		return ss.str();
	}

	return layernameFunc(index);
}

double ScriptedModelUI::GetDefaultParamValue(int layer, int iPar) {
	if(!dpvFunc)
		return 0.0;

	return dpvFunc(layer, iPar);
}

ExtraParam ScriptedModelUI::GetExtraParameter(int index) {
	// TODO::Lua (later): Use the function, if exists
/*
	if(!extraParamValueFunc)
		return ExtraParam();

	return extraParamValueFunc(index);
*/
	if(index == 0)
	{
		return ExtraParam("Scale", 1.0);
	}
	return ExtraParam();
}

int ScriptedModelUI::GetNumExtraParams() {
	return 1;
}

int ScriptedModelUI::GetNumDisplayParams() {
	// TODO::Lua (later): Use the function, if exists
	return 0;
}

string ScriptedModelUI::GetDisplayParamName(int index) {
	// TODO::Lua: Use the function, if exists
	// TODO: What about displayparamvalue?
	return "TODO";
}

int ScriptedModelUI::GetNumRelatedModels() {
	return 0;
}

string ScriptedModelUI::GetRelatedModelName(int index) {
	return "N/A";
}

int ScriptedModelUI::GetRelatedModelIndex(int index) {
	return -1;
}

void ScriptedModelUI::SetHandlerCallbacks(GetNameFunc layername, 
										  GetNameFunc layerparam, 
										  GetDefaultParamValueFunc dpv, 
										  GetParamApplicableFunc ipa) {
	layernameFunc = layername;
	layerparamnameFunc = layerparam;
	dpvFunc = dpv;
	paFunc = ipa;
}

//////////////////////////////////////////////////////////////////////////

PDBModelUI::PDBModelUI(const char *tName) {
	mi.nExtraParams = 10;

	this->extraParamsInfo.resize(mi.nExtraParams);
	extraParamsInfo[0] = ExtraParam("Scale", 1.0, false, false, false, NEGINF, POSINF, false, 12);
	extraParamsInfo[1] = ExtraParam("Solvent ED", 334.0);
	extraParamsInfo[2] = ExtraParam("C1", 1.0, false, true, true, 0.95, 1.05);
	extraParamsInfo[3] = ExtraParam("Solvent Voxel Size", 0.2);
	extraParamsInfo[4] = ExtraParam("Solvent Probe Radius", 0.14);
	extraParamsInfo[5] = ExtraParam("Solvation Thickness", 0.28, false, true, true, 0.0);
	extraParamsInfo[6] = ExtraParam("Outer Solvent ED");
	extraParamsInfo[7] = ExtraParam("Fill Holes", 0.0, false, false, true, 0.0, 1.0, true);
	extraParamsInfo[8] = ExtraParam("Solvent Only", 0.0, false, false, true, 0.0, 1.0, true);
	extraParamsInfo[9] = ExtraParam("Solvent method", 4.0, false, false, true, 0.0, 4.0, true);
	
	extraParamsTypes.resize(mi.nExtraParams, EPT_DOUBLE);
	extraParamOptions.resize(mi.nExtraParams, std::vector<std::string>());

	extraParamsTypes[7] = EPT_CHECKBOX;
	extraParamsTypes[8] = EPT_CHECKBOX;
	extraParamsTypes[9] = EPT_MULTIPLE_CHOICE;

	extraParamOptions[9].resize(RAD_SIZE);
	extraParamOptions[9][0] = "No Solvent";
	extraParamOptions[9][1] = "Van der Waals";
	extraParamOptions[9][2] = "Empirical";
	extraParamOptions[9][3] = "Calculated";
	extraParamOptions[9][4] = "Dummy Atoms";
	extraParamOptions[9][5] = "Dummy Atoms (voxelized)";

	mi = ModelInformation(tName, -1, -1, false, 0, 0, 0, mi.nExtraParams, 0, false /*TODO::GPU*/, true, true, true/*even though this is never used*/);

}

bool PDBModelUI::setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd) {
	// Do nothing except this
	return true;
}

PDBModelUI::~PDBModelUI() {

}
string PDBModelUI::GetLayerParamName(int index) {
	return "N/A";
}

string PDBModelUI::GetLayerName(int index) {
	return "N/A";
}

AMPModelUI::AMPModelUI(const char *tName) {
	mi.nExtraParams = 1;

	this->extraParamsInfo.resize(mi.nExtraParams);
	extraParamsInfo[0] = ExtraParam("Scale", 1.0);

	extraParamsTypes.resize(mi.nExtraParams, EPT_DOUBLE);
	extraParamOptions.resize(mi.nExtraParams, std::vector<std::string>());

	mi = ModelInformation(tName, -1, -1, false, 0, 0, 0, 1, 0, false /*TODO::GPU*/, false, false, true/*even though this is never used*/);
}

string AMPModelUI::GetLayerParamName(int index) {
	return "N/A";
}

string AMPModelUI::GetLayerName(int index) {
	return "N/A";
}

bool AMPModelUI::setModel(FrontendComm *com, const wchar_t *container, ModelInformation info, int endInd) {
	return true;
}

