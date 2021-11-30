/* Implements the Json-ized BackendComm */

#ifndef BACKEND_WRAPPER_JSON
#define BACKEND_WRAPPER_JSON
#include <rapidjson/document.h>
#include <string>
#include "Common.h"
#include "CommProtocol.h"
#include <fstream>

class LocalBackend;
class LocalBackendParameterTreeConverter;
class JsonWriter;

class EXPORTED_BE BackendWrapper : public BackendComm
{
public: 
	BackendWrapper();
	virtual ~BackendWrapper();

	std::string CallBackend(std::string json);

	// Calls to the local backend
protected:
	struct LocalBackendInfo
	{
		LocalBackend *local_backend;
		JobPtr job;
		LocalBackendParameterTreeConverter *Converter;
	};
	virtual const LocalBackendInfo GetBackendInfo(const std::string clientId) = 0;

protected:
	// Handling with all calls
	void CallBackendFunction(const std::string functionName, const rapidjson::Value &args, JsonWriter &writer, const LocalBackendInfo &backend);

	void WriteResponseClientData(JsonWriter &writer, const rapidjson::Value::ConstMemberIterator &clientDataMember);
	void WriteResponseError(JsonWriter &writer, int errorCode, std::string errorMsg);
	void CheckErrorCode(ErrorCode err);
	void CheckCapabilities(bool checkTdr = true);
	void checkTdrLevel();
	void checkAVX();
	void checkGPU();
	void WriteGraph(JsonWriter &writer, const LocalBackendInfo &backend);

	void GetAllModelMetadata(JsonWriter &writer, const LocalBackendInfo &backend);
	void GetJobStatus(JsonWriter &writer, const LocalBackendInfo &backend);
	void Stop(const LocalBackendInfo &backend);
	void StartGenerate(const rapidjson::Value &json, const LocalBackendInfo &backend);
	void GetGenerateResults(JsonWriter &writer, const LocalBackendInfo &backend);
	void StartFit(const rapidjson::Value &json, const LocalBackendInfo &backend);
	void GetFitResults(JsonWriter &writer, const LocalBackendInfo &backend);
	void SetGPUFlag(const rapidjson::Value &json);
	// Retrieve data from backend
	//////////////////////////////////////////////////////////////////////////
	void GetAmplitude(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend);
	void GetPDB(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend);
};


#endif
