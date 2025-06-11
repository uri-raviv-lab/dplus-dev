/*
 * This file contains the implementation of the MetadataRepository::SingleContainerRepository class
 */

#include "MetadataRepository.h"
using namespace std;

/**
 * @file SingleContainerRepository.cpp
 * @brief Implements the MetadataRepository::SingleContainerRepository class for managing model metadata in a single container.
 *
 * The SingleContainerRepository module is responsible for:
 *  - Parsing and storing model categories, models, extra parameters, and layer information from a JSON metadata source.
 *  - Providing query and access methods for model categories, model information, extra parameters, and layer details.
 *  - Supporting metadata-driven UI and backend operations by exposing structured model information.
 *
 * Key Concepts:
 *  - Metadata Parsing: Reads and interprets model metadata from a rapidjson::Value, including categories, models, extra parameters, and layers.
 *  - Category and Model Management: Maintains mappings of model categories and models for efficient lookup and retrieval.
 *  - Layer and Parameter Information: Stores and provides access to layer parameter names, applicability, and default values for each model.
 *  - Error Handling: Returns error codes for invalid queries or missing data, supporting robust integration with frontend and backend components.
 *
 * Fields:
 *  - std::wstring _containerName: Name of the model container.
 *  - std::map<int, ModelCategory> _categories: Mapping from category index to ModelCategory.
 *  - std::map<int, CompleteModelInformation> _models: Mapping from model index to CompleteModelInformation.
 *
 * Main Methods:
 *  - SingleContainerRepository(const rapidjson::Value& json):
 *      Constructor. Parses container name, categories, and models from JSON.
 *  - void ParseModelCategory(const rapidjson::Value& json):
 *      Parses and stores a model category from JSON.
 *  - void ParseModel(const rapidjson::Value& json):
 *      Parses and stores complete model information from JSON.
 *  - void ParseExtraParams(CompleteModelInformation&, const rapidjson::Value& json):
 *      Parses extra parameters for a model.
 *  - void ParseLayers(CompleteModelInformation&, const rapidjson::Value& json):
 *      Parses layer information for a model.
 *  - int QueryCategoryCount() const:
 *      Returns the number of model categories.
 *  - ModelCategory QueryCategory(int ind) const:
 *      Returns the ModelCategory for a given index.
 *  - int QueryModelCount() const:
 *      Returns the number of models.
 *  - ModelInformation QueryModel(int ind) const:
 *      Returns the ModelInformation for a given model index.
 *  - ErrorCode GetLayerParamNames(int index, char** lpNames, int nlp) const:
 *      Retrieves layer parameter names for a model.
 *  - ErrorCode GetExtraParamInfo(int index, ExtraParam* ep, int nEP) const:
 *      Retrieves extra parameter information for a model.
 *  - ErrorCode GetLayerInfo(int index, int layerIndex, char* layerName, int* applicability, double* defaultValues, int nlp) const:
 *      Retrieves detailed layer information for a model.
 *
 * Error Handling:
 *  - Returns specific error codes (e.g., ERROR_MODELNOTFOUND, ERROR_INVALIDARGS) for invalid queries or missing data.
 *  - Provides default values for missing or invalid categories and models.
 *  - Ensures safe string operations and bounds checking when copying names and parameter data.
 *
 * Dependencies:
 *  - MetadataRepository.h for class and method declarations.
 *  - rapidjson/document.h for JSON parsing.
 *  - Standard C++ libraries for string and map operations.
 *  - ModelCategory, CompleteModelInformation, ModelInformation, ExtraParam, LayerInfo, and ErrorCode types.
 *
 * See MetadataRepository.h for class and method declarations.
 */

MetadataRepository::SingleContainerRepository::SingleContainerRepository(const rapidjson::Value &json)
{
	const char *containerName = json["containerName"].GetString();
	_containerName = wstring(containerName, containerName + strlen(containerName));  // Convert to wstring

	const rapidjson::Value &categories = json["modelCategories"];
	for (auto it = categories.Begin(); it != categories.End(); it++)
		ParseModelCategory(*it);
	const rapidjson::Value &models = json["models"];
	for (auto it = models.Begin(); it != models.End(); it++)
		ParseModel(*it);
}

void MetadataRepository::SingleContainerRepository::ParseModelCategory(const rapidjson::Value &json)
{
	ModelCategory cat;

	strncpy(cat.name, json["name"].GetString(), sizeof(cat.name) - 1);
	cat.name[sizeof(cat.name) - 1] = '\0';

	cat.type = (ModelType)json["type"].GetInt();

	const rapidjson::Value &models = json["models"];
	int i = 0;
	for (auto it = models.Begin(); it != models.End(); it++)
		cat.models[i++] = it->GetInt();
	cat.models[i] = -1;

	_categories[json["index"].GetInt()] = cat;
}

void MetadataRepository::SingleContainerRepository::ParseModel(const rapidjson::Value &json)
{
	CompleteModelInformation cmi;

	strncpy(cmi.modelInformation.name, json["name"].GetString(), sizeof(cmi.modelInformation.name) - 1);
	cmi.modelInformation.name[sizeof(cmi.modelInformation.name) - 1] = '\0';

	cmi.modelInformation.category = json["category"].GetInt();
	cmi.modelInformation.modelIndex = json["index"].GetInt();
	cmi.modelInformation.isGPUCompatible = json["gpuCompatible"].GetBool();
	cmi.modelInformation.isSlow = json["slow"].GetBool();
	cmi.modelInformation.ffImplemented = json["ffImplemented"].GetBool();
	cmi.modelInformation.isLayerBased = json["isLayerBased"].GetBool();

	if (json.HasMember("extraParams"))
		ParseExtraParams(cmi, json["extraParams"]);
	cmi.modelInformation.nExtraParams = int(cmi.extraParams.size());

	if (json.HasMember("layers"))
		ParseLayers(cmi, json["layers"]);
	else {
		//If no layers, init min and max to zero, because ModelInformation has no default constructor
		cmi.modelInformation.minLayers = 0;
		cmi.modelInformation.maxLayers = 0;
	}

	cmi.modelInformation.nlp = int(cmi.layerParamNames.size());

	_models[cmi.modelInformation.modelIndex] = cmi;
}

void MetadataRepository::SingleContainerRepository::ParseExtraParams(CompleteModelInformation &cmi, const rapidjson::Value &json)
{
	int i = 0;
	for (auto it = json.Begin(); it != json.End(); it++)
	{
		const rapidjson::Value &epj = *it;
		ExtraParam ep;

		strncpy(ep.name, epj["name"].GetString(), sizeof(ep.name) - 1);
		ep.name[sizeof(ep.name) - 1] = '\0';
		ep.defaultVal = epj["defaultValue"].GetDouble();
		ep.isIntegral = epj["isIntegral"].GetBool();
		ep.decimalPoints = epj["decimalPoints"].GetInt();
		ep.isAbsolute = epj["isAbsolute"].GetBool();
		ep.canBeInfinite = epj["canBeInfinite"].GetBool();

		ep.isRanged = epj.HasMember("range");
		if (ep.isRanged)
		{
			ep.rangeMin = epj["range"]["min"].GetDouble();
			ep.rangeMax = epj["range"]["max"].GetDouble();
		}

		cmi.extraParams.push_back(ep);
	}
}

void MetadataRepository::SingleContainerRepository::ParseLayers(CompleteModelInformation &cmi, const rapidjson::Value &json)
{
	cmi.modelInformation.minLayers = json["min"].GetInt();
	cmi.modelInformation.maxLayers = json["max"].GetInt();

	for (auto it = json["params"].Begin(); it != json["params"].End(); it++)
		cmi.layerParamNames.push_back(it->GetString());

	for (auto it = json["layerInfo"].Begin(); it != json["layerInfo"].End(); it++)
	{
		LayerInfo li;
		const rapidjson::Value &liJson = *it;

		li.Index = liJson["index"].GetInt();
		li.Name = liJson["name"].GetString();
		for (auto jt = liJson["applicability"].Begin(); jt != liJson["applicability"].End(); jt++)
			li.Applicability.push_back(jt->GetInt());

		for (auto jt = liJson["defaultValues"].Begin(); jt != liJson["defaultValues"].End(); jt++)
			li.DefaultValues.push_back(jt->GetDouble());

		cmi.layers[li.Index] = li;
	}
}

int MetadataRepository::SingleContainerRepository::QueryCategoryCount() const
{
	return int(_categories.size());
}

ModelCategory MetadataRepository::SingleContainerRepository::QueryCategory(int ind) const
{
	if (_categories.find(ind) != _categories.end())
		return _categories.at(ind);

	ModelCategory mc = { "N/A", MT_FORMFACTOR, { -1 } };
	return mc;
}

int MetadataRepository::SingleContainerRepository::QueryModelCount() const
{
	return int(_models.size());
}

ModelInformation MetadataRepository::SingleContainerRepository::QueryModel(int ind) const
{
	if (_models.find(ind) != _models.end())
		return _models.at(ind).modelInformation;

	return ModelInformation("N/A");
}

ErrorCode MetadataRepository::SingleContainerRepository::GetLayerParamNames(int index, char **lpNames, int nlp) const
{
	if (_models.find(index) == _models.end())
		return ERROR_MODELNOTFOUND;

	const CompleteModelInformation &cmi = _models.at(index);
	if (nlp > cmi.modelInformation.nlp)
		return ERROR_INVALIDARGS;

	for (int i = 0; i < nlp; i++)
		strncpy(lpNames[i], cmi.layerParamNames[i].c_str(), 256);

	return OK;
}

ErrorCode MetadataRepository::SingleContainerRepository::GetExtraParamInfo(int index, ExtraParam *ep, int nEP) const
{
	if (_models.find(index) == _models.end())
		return ERROR_MODELNOTFOUND;

	const CompleteModelInformation &cmi = _models.at(index);
	if (nEP > cmi.modelInformation.nExtraParams)
		return ERROR_INVALIDARGS;

	for (int i = 0; i < nEP; i++)
		ep[i] = cmi.extraParams[i];

	return OK;
}

ErrorCode MetadataRepository::SingleContainerRepository::GetLayerInfo(int index, int layerIndex,
	char *layerName, int *applicability,
	double *defaultValues, int nlp) const
{
	if (_models.find(index) == _models.end())
		return ERROR_MODELNOTFOUND;

	const CompleteModelInformation &cmi = _models.at(index);
	if (nlp > cmi.modelInformation.nlp)
		return ERROR_INVALIDARGS;

	if (cmi.modelInformation.maxLayers != -1 && layerIndex >= cmi.modelInformation.maxLayers)
		return ERROR_INVALIDARGS;

	int internalLayerIndex = layerIndex;
	if (layerIndex >= cmi.modelInformation.minLayers)  // All layers after min are considered -1
		internalLayerIndex = -1;
	const LayerInfo &li = cmi.layers.at(internalLayerIndex);

	_snprintf(layerName, 256, li.Name.c_str(), layerIndex + 1);
	
	
	for (int i = 0; i < nlp; i++)
	{
		applicability[i] = li.Applicability[i];
		defaultValues[i] = li.DefaultValues[i];
	}

	return OK;
}