#include "MetadataRepository.h"

using namespace std;

MetadataRepository::MetadataRepository(std::string json)
{
	rapidjson::Document jsonDoc;
	jsonDoc.Parse(json.c_str());

	assert(jsonDoc.IsArray());
	for (auto it = jsonDoc.Begin(); it != jsonDoc.End(); it++)
	{
		SingleContainerRepository repo(*it);
		_repos[repo.GetContainerName()] = repo;
	}
}

/* All the ModeldataRepository methods simple forward the call to the appropriate SingleContainerRepository */
const MetadataRepository::SingleContainerRepository &MetadataRepository::repo(const wchar_t *container) const
{
	if (container)
		return _repos.at(container);

	return _repos.at(L"xplusmodels"); // Default container
}

int MetadataRepository::QueryCategoryCount(const wchar_t *container) const
{
	// We use 'at' because [] doesn't work on const maps apparantly
	return repo(container).QueryCategoryCount();
}

int MetadataRepository::QueryModelCount(const wchar_t *container) const
{
	return repo(container).QueryModelCount();
}

ModelCategory MetadataRepository::QueryCategory(const wchar_t *container, int catInd) const
{
	return repo(container).QueryCategory(catInd);
}

ModelInformation MetadataRepository::QueryModel(const wchar_t *container, int index) const
{
	return repo(container).QueryModel(index);
}

ErrorCode MetadataRepository::GetLayerParamNames(const wchar_t *container, int index, char **lpNames, int nlp) const
{
	return repo(container).GetLayerParamNames(index, lpNames, nlp);
}

ErrorCode MetadataRepository::GetExtraParamInfo(const wchar_t *container, int index, ExtraParam *ep, int nEP) const
{
	return repo(container).GetExtraParamInfo(index, ep, nEP);
}

ErrorCode MetadataRepository::GetLayerInfo(const wchar_t *container, int index, int layerIndex,
	char *layerName, int *applicability,
	double *defaultValues, int nlp) const
{	
	return repo(container).GetLayerInfo(index, layerIndex, layerName, applicability, defaultValues, nlp);
}

