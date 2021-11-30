/*
 * The MetadataRepository is responsible for parsing the Backend's
 * Metadata JSON and translating it to the data structures required by
 * the UI.
 */
#pragma once

#include <vector>
#include <string>
#include <map>
#include "../../Common/CommProtocol.h"
#include <rapidjson/document.h>

class MetadataRepository
{
private:
	/* A repository for a single container */
	class SingleContainerRepository
	{
	private:
		struct LayerInfo
		{
			int Index;
			std::string Name;
			std::vector<int> Applicability;
			std::vector<double> DefaultValues;
		};

		struct CompleteModelInformation
		{
			/* ModelInformation is incomplete, the JSON contains more data for each model */
			ModelInformation modelInformation;
			std::vector<ExtraParam> extraParams;
			std::map<int, LayerInfo> layers;
			std::vector<std::string> layerParamNames;
		};

		std::map<int, ModelCategory> _categories;
		std::map<int, CompleteModelInformation> _models;
		std::wstring _containerName;


		void ParseModel(const rapidjson::Value &json);
		void ParseModelCategory(const rapidjson::Value &json);
		void ParseExtraParams(CompleteModelInformation &cmi, const rapidjson::Value &json);
		void ParseLayers(CompleteModelInformation &cmi, const rapidjson::Value &json);

	public:
		SingleContainerRepository() { }; // Default constructor to place in map
		SingleContainerRepository(const rapidjson::Value &json);

		std::wstring GetContainerName() const { return _containerName; }
		int QueryCategoryCount() const;
		int QueryModelCount() const;
		ModelCategory QueryCategory(int catInd) const;
		ModelInformation QueryModel(int index) const;
		ErrorCode GetLayerParamNames(int index, char **lpNames, int nlp) const;
		ErrorCode GetExtraParamInfo(int index, ExtraParam *ep, int nEP) const;
		ErrorCode GetLayerInfo(int index, int layerIndex,
			char *layerName, int *applicability,
			double *defaultValues, int nlp) const;
	};

	/* The big repository holds the single container repositories */
	std::map<std::wstring, SingleContainerRepository> _repos;
	const SingleContainerRepository &repo(const wchar_t *container) const;

public:
	MetadataRepository(std::string json);

	int QueryCategoryCount(const wchar_t *container) const;
	int QueryModelCount(const wchar_t *container) const;
	ModelCategory QueryCategory(const wchar_t *container, int catInd) const;
	ModelInformation QueryModel(const wchar_t *container, int index) const;
	ErrorCode GetLayerParamNames(const wchar_t *container, int index, char **lpNames, int nlp) const;
	ErrorCode GetExtraParamInfo(const wchar_t *container, int index, ExtraParam *ep, int nEP) const;
	ErrorCode GetLayerInfo(const wchar_t *container, int index, int layerIndex,
		char *layerName, int *applicability,
		double *defaultValues, int nlp) const;
};

