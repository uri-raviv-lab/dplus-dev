#include "LocalBackendParameterTree.h"
#include <string>
#include <iostream>
#include <sstream>
#include "LocalBackend.h"
#include <algorithm>

using namespace std;

#include "JobManager.h"
#include <rapidjson/document.h>
using namespace rapidjson;

/**
 * @file LocalBackendParameterTree.cpp
 * @brief Implements the LocalBackendParameterTreeConverter class and related logic for model mapping,
 *        parameter tree conversion, and backend model lifecycle management in the local backend.
 *
 * The LocalBackendParameterTree module is responsible for:
 *  - Defining the LocalBackendParameterTreeConverter class, which manages the mapping between frontend (state) models
 *    and backend (internal) models for a given job.
 *  - Handling creation, mapping, and destruction of backend models based on frontend parameter tree state.
 *  - Supporting conversion between frontend parameter tree JSON and backend model representations.
 *  - Managing model caching, reuse, and cleanup to optimize backend resource usage.
 *  - Providing utility functions for model creation from JSON, composite/domain model instantiation, and model lookup.
 *  - Integrating with the LocalBackend for model creation, destruction, and amplitude handling.
 *  - Ensuring consistency between frontend and backend model hierarchies during parameter tree updates.
 *
 * Key Concepts:
 *  - Model Mapping: Maintains bidirectional maps between frontend (state) ModelPtr and backend (internal) ModelPtr.
 *  - Model Lifecycle: Handles creation, reuse, and destruction of backend models as the parameter tree changes.
 *  - Parameter Tree Conversion: Converts frontend parameter tree JSON to backend ParameterTree objects.
 *  - Model Caching: Avoids redundant backend model creation by caching and reusing models when possible.
 *  - Amplitude and Geometry Handling: Supports creation of amplitude, geometry, and symmetry models from JSON.
 *  - Error Handling: Throws backend_exception on invalid model types, missing files, or mapping errors.
 *
 * Fields:
 *  - LocalBackend* _pBackend: Pointer to the backend instance used for model creation and destruction.
 *  - JobPtr _job: Handle to the current job, used for backend operations and model association.
 *  - std::map<ModelPtr, ModelPtr> _stateToInternal: Maps frontend (state) ModelPtr to backend (internal) ModelPtr.
 *  - std::map<ModelPtr, ModelPtr> _internalToState: Maps backend (internal) ModelPtr to frontend (state) ModelPtr.
 *  - std::set<ModelPtr> _usedModels: Tracks state models used during the current parameter tree conversion, for cleanup.
 *
 * Main Methods:
 *  - LocalBackendParameterTreeConverter (constructor): Initializes the converter with backend and job context.
 *  - StateToInternal / InternalToState: Maps between frontend and backend model pointers.
 *  - StateFromAmp: Retrieves the frontend model pointer associated with a given amplitude.
 *  - CreateCompositeModel / CreateDomainModel: Creates or retrieves composite/domain backend models for a given state model.
 *  - MapModel: Establishes bidirectional mapping between state and internal models.
 *  - ClearUnusedModels: Destroys backend models that are no longer referenced in the current parameter tree.
 *  - FromStateJSON: Converts frontend parameter tree JSON to backend ParameterTree, updating model mappings.
 *  - CreateModelFromJSON / ModelFromJsonString: Creates backend models from JSON definitions, supporting various model types.
 *  - GetStateModels: Returns a list of all frontend (state) models currently mapped.
 *
 * Threading and Safety:
 *  - This class is not inherently thread-safe; external synchronization is required if used concurrently.
 *  - Model mapping and destruction are managed per job and parameter tree update.
 *
 * Error Handling:
 *  - Throws backend_exception for invalid model types, missing files, or mapping errors.
 *  - Ensures backend models are destroyed when no longer referenced to prevent resource leaks.
 *
 * Dependencies:
 *  - LocalBackend for backend model creation and destruction.
 *  - JobManager for job and amplitude-to-model mapping.
 *  - rapidjson for JSON parsing and model definition extraction.
 *  - ParameterTreeConverter for base parameter tree conversion logic.
 *
 * See LocalBackendParameterTree.h for class and method declarations.
 */

LocalBackendParameterTreeConverter::LocalBackendParameterTreeConverter(LocalBackend *backend, JobPtr job)
{
	_pBackend = backend;
	_job = job;
}


ModelPtr LocalBackendParameterTreeConverter::StateToInternal(ModelPtr stateModel) const
{

	try
	{
		return _stateToInternal.at(stateModel);
	}
	catch (...)
	{
		throw new backend_exception(ERROR_MODELNOTFOUND, "Failed to map model pointer to internal model pointer");
	}

}

ModelPtr LocalBackendParameterTreeConverter::StateFromAmp(Amplitude * ampl) const
{
	Job j = JobManager::GetInstance().GetJobInformation(_job);
	ModelPtr internal = j.ampToUid[ampl];
	return InternalToState(internal);
}

ModelPtr LocalBackendParameterTreeConverter::InternalToState(ModelPtr internalModel) const
{
	return _internalToState.at(internalModel);
}

/* Model creation functions */
ModelPtr LocalBackendParameterTreeConverter::CreateCompositeModel(ModelPtr stateModel)
{
	//Check the map if backend already has the model cached
	map<ModelPtr, ModelPtr>::iterator i = _stateToInternal.find(stateModel);
	ModelPtr compositeModel;

	if (i == _stateToInternal.end()) {
		//not found in map, so create the model
		compositeModel = _pBackend->HandleCreateCompositeModel(_job);
		MapModel(stateModel, compositeModel);
	}
	else {
		//found in map, just return the internal ModelPtr		
		compositeModel = i->second;
	}

	_usedModels.insert(stateModel);
	return compositeModel;
}

ModelPtr LocalBackendParameterTreeConverter::CreateDomainModel(ModelPtr stateModel)
{
	//Check the map if backend already has the model cached
	map<ModelPtr, ModelPtr>::iterator i = _stateToInternal.find(stateModel);
	ModelPtr domainModel;

	if (i == _stateToInternal.end()) {
		//not found in map, so create the model
		domainModel = _pBackend->HandleCreateDomainModel(_job);
		MapModel(stateModel, domainModel);
	}
	else {
		//found in map, just return the internal ModelPtr
		domainModel =  i->second;
	}

	_usedModels.insert(stateModel);
	return domainModel;
}

void LocalBackendParameterTreeConverter::MapModel(ModelPtr stateModel, ModelPtr internalModel)
{
	_stateToInternal[stateModel] = internalModel;
	_internalToState[internalModel] = stateModel;
}


void	 LocalBackendParameterTreeConverter::ClearUnusedModels()
{
	set<ModelPtr> toBeDeleted;
	for (auto itr = _stateToInternal.begin(); itr != _stateToInternal.end(); ++itr)
	{
		//check if any of the models in the map are not used
		if (_usedModels.find((*itr).first) == _usedModels.end())
		{
			toBeDeleted.insert((*itr).first);
		}
	}

	for (auto itr = toBeDeleted.begin(); itr != toBeDeleted.end(); ++itr)
	{
		ModelPtr stateModelPtr = (*itr);
		ModelPtr internalModelPtr = _stateToInternal[stateModelPtr];
		_pBackend->HandleDestroyModel(_job, internalModelPtr);
		_stateToInternal.erase(_stateToInternal.find(stateModelPtr));
		_internalToState.erase(_internalToState.find(internalModelPtr));
	}
}

ParameterTree LocalBackendParameterTreeConverter::FromStateJSON(const rapidjson::Value &json)
{
	_usedModels.clear();
	ParameterTree pt = ParameterTreeConverter::FromStateJSON(json);
	ClearUnusedModels();

	return pt;
}




ModelPtr LocalBackendParameterTreeConverter::CreateModelFromJSON(const std::string &str, const rapidjson::Value &model, bool bAmp, ModelPtr stateModel)
{
	//Check the map if backend already has the model cached
	map<ModelPtr, ModelPtr >::iterator i = _stateToInternal.find(stateModel);
	ModelPtr internalModel;

	if (i == _stateToInternal.end()) {
		//not found in map, so create the model
		internalModel = ModelFromJsonString(str, model, bAmp);
		MapModel(stateModel, internalModel);
	}
	else {
		//found in map, just return the internal ModelPtr
		internalModel = i->second;
	}

	_usedModels.insert(stateModel);
	return internalModel;
}


ModelPtr LocalBackendParameterTreeConverter::ModelFromJsonString(const std::string &strMixedCase, const rapidjson::Value &model, bool bAmp)
{
	std::string str = strMixedCase;
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);

	if (str.empty()) {

		throw backend_exception(ERROR_INVALIDARGS, "Invalid model, no type (model.Type) defined!");
		return 0;
	}

	size_t indexOfComma = str.find_first_of(",");

	if (indexOfComma == string::npos) //the default models have a comma added, and a numerical model id. the other options are as follow here
	{
		if (str == "scripted model")
		{
			string cscript = GetScript(model);
			return _pBackend->HandleCreateScriptedModel(_job, cscript.c_str(), (unsigned)cscript.size());
		}
		else if (str == "scripted geometry")
		{
			//TODO - ignore?? assert if we get here or not?
			string cscript = GetScript(model);
			ModelPtr geometry = 0;
			return _pBackend->HandleCreateGeometricAmplitude(_job, geometry);
		}
		else if (str == "scripted symmetry")
		{
			string cscript = GetScript(model);
			return _pBackend->HandleCreateScriptedSymmetry(_job, cscript.c_str(), (unsigned)cscript.size());
		}
		else if (str == "pdb" || str == "epdb")
		{
			if (!bAmp) {

				throw backend_exception(ERROR_INVALIDARGS, "A PDB file cannot be chosen as a geometry, only amplitude.");
			}

			Value::ConstMemberIterator Centered = model.FindMember("Centered");

			bool centered = Centered->value.GetBool();
			string fname = GetFilename(model);
			string anomfname = GetString(model, "AnomFilename");
			if (iterationMethod != OA_MC)
			{
				if (!anomfname.empty())
					throw backend_exception(ERROR_INVALIDPARAMTREE, "You are running a calculation with an anomalous PDB and Vegas or Gauss-Kronrod integration selected. D+ only supports anomalous PDBs with Monte Carlo integration.");
			}
			if (fname.empty()) {

				throw backend_exception(ERROR_INVALIDARGS, "Invalid PDB, no file(model.Filename) defined!");
			}

			bool electronPDB = (str == "epdb");

			return _pBackend->HandleCreateFileAmplitude(_job, AF_PDB, StringToWideCharT(fname), centered, StringToWideCharT(anomfname), electronPDB);

		}
		else if (str == "amp")
		{
			if (!bAmp) {

				throw backend_exception(ERROR_INVALIDARGS, "An Amplitude Grid cannot be chosen as a geometry, only amplitude.");
			}

			Value::ConstMemberIterator Centered = model.FindMember("Centered");
			bool centered = Centered->value.GetBool();
			//if (Centered->value.Empty() || !Centered->value.IsBool()) {

			//	throw backend_exception(ERROR_INVALIDARGS, "Invalid centered");
			//}

			string fname = GetFilename(model);
			if (fname.empty()) {

				throw backend_exception(ERROR_INVALIDARGS, "Invalid Amplitude Grid, no file (model.Filename) defined!");
			}

			return _pBackend->HandleCreateFileAmplitude(_job, AF_AMPGRID, StringToWideCharT(fname), centered);
		}
		else
		{
			throw backend_exception(ERROR_INVALIDARGS, "Invalid model type");
		}
	}

	//what if indexOfComma < 0.. we'll get to some of this code...


	if (bAmp) { // Create a geometric amplitude
		ModelPtr geometry = 0;

		try
		{
			if (indexOfComma == 0) // Default models
				geometry = _pBackend->HandleCreateModel(_job, NULL, atoi(str.substr(indexOfComma + 1).c_str()), EDProfile());
			else {          // With container
				geometry = _pBackend->HandleCreateModel(_job, StringToWideCharT(str.substr(0, indexOfComma)), atoi(str.substr(indexOfComma + 1).c_str()), EDProfile());
			}
			return _pBackend->HandleCreateGeometricAmplitude(_job, geometry);
		}
		catch (backend_exception)
		{
			// This is not a gemoetry, move on
		}

		// If it is not a geometry, it may be a symmetry
		if (indexOfComma == 0) { // Default models 
			geometry = _pBackend->HandleCreateSymmetry(_job, NULL, atoi(str.substr(indexOfComma + 1).c_str()));
		}
		else  {   // With container
			geometry = _pBackend->HandleCreateSymmetry(_job, StringToWideCharT(str.substr(0, indexOfComma)), atoi(str.substr(indexOfComma + 1).c_str()));
		}
		return geometry;
	}
	else {
		if (indexOfComma == 0) // Default models
			return _pBackend->HandleCreateModel(_job, NULL, atoi(str.substr(indexOfComma + 1).c_str()), EDProfile());
		else           // With container
			return _pBackend->HandleCreateModel(_job, StringToWideCharT(str.substr(0, indexOfComma)), atoi(str.substr(indexOfComma + 1).c_str()), EDProfile());
	}
}

std::vector<ModelPtr> LocalBackendParameterTreeConverter::GetStateModels()
{
	std::vector<ModelPtr> states;

	for (auto itr = _stateToInternal.begin(); itr != _stateToInternal.end(); ++itr)
	{
		states.push_back((*itr).first);
	}
	return states;
}
