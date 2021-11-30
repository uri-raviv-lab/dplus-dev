/*
 * Conversion between different ParameterTree representations
 */

#ifndef __PARAM_TREE_CONVERSIONS_H
#define __PARAM_TREE_CONVERSIONS_H

#include "Common.h"

#include <rapidjson/document.h>
#include "JsonWriter.h"
#include <string>


class EXPORTED_BE ParameterTreeConverter
{
protected:
	virtual ModelPtr CreateCompositeModel(ModelPtr stateModel);
	virtual ModelPtr CreateDomainModel(ModelPtr stateModel); 
	virtual ModelPtr CreateModelFromJSON(const std::string &str, const rapidjson::Value &model, bool bAmp, ModelPtr stateModel);
	//virtual void			 ClearAllModelUsageFlags();
	//virtual void			 CleanUnusedModels();

	std::string GetScript(const rapidjson::Value &model);
	std::string GetFilename(const rapidjson::Value &model);
	std::string GetString(const rapidjson::Value &model, const std::string &propertyName);
	wchar_t	*StringToWideCharT(const std::string &str);

	virtual ModelPtr GetStateModelPtr(ModelPtr stateModel);
	OAMethod_Enum iterationMethod;
public:
	virtual ParameterTree FromStateJSON(const rapidjson::Value &json);  // Convert a state file (in JSON format) to a ParameterTree and create all the models
	void WriteSimpleJSON(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const ParameterTree &pt); // Simple JSON with no model information other than the model pointers
	ParameterTree FromSimpleJSON(const rapidjson::Value &json); // Convert simple JSON back to a ParameterTree (no models are created)

private:
	void AddModelsToParamTree(ParameterTree *domain, const rapidjson::Value::ConstMemberIterator mods, bool bAmplitude);
	paramStruct ModelParamsFromJson(const rapidjson::Value &doc);
	paramStruct DomainPreferencesFromJson(const rapidjson::Value &doc);

	Parameter LoadLocationParams(const rapidjson::Value &model, const char *locp);
	void loadConstraints(const rapidjson::Value::ConstValueIterator constraintItr, double &consMin, double &consMax, int &consMinInd, int &consMaxInd, int &linkInd, bool &bCons);
	void loadConstraints(const rapidjson::Value::ConstMemberIterator constraintItr, double &consMin, double &consMax, int &consMinInd, int &consMaxInd, int &linkInd, bool &bCons);
	double jsonObjectToDouble(const rapidjson::Value &val);

	void WriteParameterTree(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const ParameterTree *pt);
	void checkForInfinityThenWriteDouble(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, double value);
	void WriteParameter(rapidjson::PrettyWriter<rapidjson::StringBuffer> &writer, const Parameter &param);
	Parameter ParameterFromJSON(const rapidjson::Value &json);
	ParameterTree ParameterTreeFromJSON(const rapidjson::Value &json);
};

#endif