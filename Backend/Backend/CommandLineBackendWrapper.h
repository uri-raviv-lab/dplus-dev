#ifndef COMMAND_LINE_BACKEND_WRAPPER_H
#define COMMAND_LINE_BACKEND_WRAPPER_H

#include "BackendWrapper.h"

class EXPORTED_BE CommandLineBackendWrapper : public BackendWrapper
{
public:
	CommandLineBackendWrapper();// std::string directory);
	void initializeCache(std::string directory);
	std::vector<ModelPtr> GetModelPtrs();
	const BackendWrapper::LocalBackendInfo GetBackendInfo(const std::string clientId);
	const BackendWrapper::LocalBackendInfo GetBackendInfo(){ return _info; }
	~CommandLineBackendWrapper();

	//function calls:
	void GetAllModelMetadata(JsonWriter &writer);  // Call the base GetAllModelMetadata with the local BackendInfo (call GetBackendInfo to retreive it, pass any client_id)
	void StartGenerate(const rapidjson::Value &json, const rapidjson::Value &useGPUJson);
	void StartFit(const rapidjson::Value &json, const rapidjson::Value &useGPUJson);
	void GetJobStatus(JsonWriter &writer);
	void GetGenerateResults(JsonWriter &writer);
	void GetFitResults(JsonWriter &writer);
	void SaveAmplitude(ModelPtr modelPtr, std::string filepath);
	void SavePDB(ModelPtr modelPtr, std::string filepath, bool electron=false);
	void CheckCapabilities(bool checkTdr);

	//void WriteGraph(JsonWriter &writer, const LocalBackendInfo &backend);
	//void Stop(const LocalBackendInfo &backend);
	//void GetAmplitude(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend);
	//void GetPDB(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend);

private:
	LocalBackendInfo _info;
};


#endif