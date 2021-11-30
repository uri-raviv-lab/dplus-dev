/*
 * This file contains the implementation of the MetadataRepository::SingleContainerRepository class
 */

#include "MetadataRepository.h"
using namespace std;

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