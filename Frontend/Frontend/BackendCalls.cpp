
#include "BackendCalls.h"
#include "JsonWriter.h"
#include "Common.h"
#include "LUAConversions.h"
#include "Conversions.h"
#include "ParamTreeConversions.h"

#include <rapidjson/document.h>
#include <string>

#include "base64.h"
#include <windows.h>

using namespace std;

std::string BackendCall::_clientId;

BackendCall::BackendCall(std::string funcName)
{
	_funcName = funcName;
}

std::string BackendCall::GetFuncName(){ return _funcName; }

std::string BackendCall::GetArgs()
{
	JsonWriter writer;
	writer.StartObject();
	WriteArguments(writer);
	writer.EndObject();
	return writer.GetString();
}

std::string BackendCall::GetOptions()
{
	JsonWriter writer;
	writer.StartObject();
	WriteOptions(writer);
	writer.EndObject();
	return writer.GetString();
}

std::string BackendCall::GetCallString()
{
	JsonWriter writer;
	writer.StartObject();

	writer.Key("client-id");
	writer.String(_clientId.c_str());

	writer.Key("client-data");
	writer.StartObject();
	writer.EndObject();

	writer.Key("function");
	writer.String(_funcName.c_str());

	writer.Key("args");
	writer.StartObject();
	WriteArguments(writer);
	writer.EndObject();

	writer.Key("options");
	writer.StartObject();
	WriteOptions(writer);
	writer.EndObject();

	writer.EndObject();

	return writer.GetString();
}

void BackendCall::ParseResults(const std::string results)
{
	rapidjson::Document doc;
	JsonWriter writer;

	doc.Parse(results.c_str());

	_errorCode = ERROR_ILLEGAL_JSON;
	_errorMessage = g_errorStrings[_errorCode];

	if (doc.HasParseError())
		throw invalid_argument("JSON Parsing Error");


	rapidjson::Value &error = doc["error"];
	if (!error.IsObject())/**/
	{
		throw invalid_argument("JSON missing error field");

	}

	else
	{
		const rapidjson::Value &code = error.FindMember("code")->value;
		if (!code.IsInt())
			throw invalid_argument("JSON invalid error code");

		_errorCode = (ErrorCode)code.GetInt();

		const rapidjson::Value &message = error.FindMember("message")->value;
		if (message.IsString())
			_errorMessage = message.GetString();
		else
			_errorMessage = g_errorStrings[_errorCode];

		if (_errorCode != OK)
			return;
	}


	//get here when all is ok
	const rapidjson::Value &result = doc.FindMember("result")->value;

	if (result.IsNull())
	{
		_errorCode = ERROR_ILLEGAL_JSON;
		_errorMessage = g_errorStrings[_errorCode];
		throw invalid_argument("ERROR ILLEGAL JSON");
	}

	ParseResultElement(result);

}

void GetAllModelMetadataCall::ParseResultElement(const rapidjson::Value &result)
{
	JsonWriter writer;
	result.Accept(writer);

	_metadata = writer.GetString();
}

FileContainingCall::FileContainingCall(std::string funcName, const std::map<ModelPtr, std::vector<std::wstring>> &fileMap)
	: BackendCall(funcName)
{
	_filenames.reserve(fileMap.size());

	for (auto &key : fileMap)
		for (auto &filename : key.second)
			_filenames.push_back(filename);
}

StartFitCall::StartFitCall(const wchar_t *luaScript, const std::vector<double>& x,
	const std::vector<double>& y, const std::vector<int>& mask, const std::map<ModelPtr, std::vector<std::wstring>> &fileMap, bool useGPU) :
	FileContainingCall("StartFit", fileMap)
{
	_luaScript = luaScript;
	_x = x;
	_y = y;
	_mask = mask;
	_useGPU = useGPU;
}


void StartFitCall::WriteArguments(JsonWriter &writer)
{
	//convert wstring to string
	int len;
	int slength = (int)_luaScript.length() + 1;
	len = WideCharToMultiByte(CP_ACP, 0, _luaScript.c_str(), slength, 0, 0, 0, 0);
	char* buf = new char[len];
	WideCharToMultiByte(CP_ACP, 0, _luaScript.c_str(), slength, buf, len, 0, 0);
	std::string r(buf);
	delete[] buf;


	writer.Key("state");

	LuaToJSON luaToJsonConverter(r);
	luaToJsonConverter.WriteState(writer);

	// Write the X vector with writer.StartArray,...
	writer.Key("x");
	writer.StartArray();
	for (auto itr = _x.begin(); itr != _x.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();


	writer.Key("y");
	writer.StartArray();
	for (auto itr = _y.begin(); itr != _y.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();

	writer.Key("mask");
	writer.StartArray();
	for (auto itr = _mask.begin(); itr != _mask.end(); ++itr)
	{
		writer.Int(*itr);
	}
	writer.EndArray();
}

void StartFitCall::WriteOptions(JsonWriter & writer)
{
	writer.Key("useGPU");
	writer.Bool(_useGPU);
}

StartGenerateCall::StartGenerateCall(const wchar_t *luaScript, const std::vector<double> &x, const std::map<ModelPtr, std::vector<std::wstring>> &fileMap, bool useGPU )
	: FileContainingCall("StartGenerate", fileMap)
{
	_luaScript = luaScript;
	_x = x;
	_useGPU = useGPU;
}

void StartGenerateCall::WriteArguments(JsonWriter &writer)
{
	//convert wstring to string
	int len;
	int slength = (int)_luaScript.length() + 1;
	len = WideCharToMultiByte(CP_ACP, 0, _luaScript.c_str(), slength, 0, 0, 0, 0);
	char* buf = new char[len];
	WideCharToMultiByte(CP_ACP, 0, _luaScript.c_str(), slength, buf, len, 0, 0);
	std::string r(buf);
	delete[] buf;


	writer.Key("state");

	LuaToJSON luaToJsonConverter(r);
	luaToJsonConverter.WriteState(writer);


	// Write the X vector with writer.StartArray,...
	writer.Key("x");
	writer.StartArray();
	for (auto itr = _x.begin(); itr != _x.end(); ++itr)
	{
		writer.Double(*itr);
	}
	writer.EndArray();
}

void StartGenerateCall::WriteOptions(JsonWriter & writer)
{
	writer.Key("useGPU");
	writer.Bool(_useGPU);
}

void GetJobStatusCall::ParseResultElement(const rapidjson::Value &result)
{
	_jobStatus = JobStatusFromJSON(result);
}

//GetFileCall::GetFileCall(std::string funcName, ModelPtr model) : BackendCall(funcName)
//{
//	_model = model;
//}

GetFileCall::GetFileCall(std::string funcName, ModelPtr model, const wchar_t *filename) : BackendCall(funcName)
{
	_model = model;
	_filename = std::wstring(filename);
}

void GetFileCall::WriteArguments(JsonWriter &writer)
{
	//convert wstring to string
	int len;
	int slength = (int)_filename.length() + 1;
	len = WideCharToMultiByte(CP_ACP, 0, _filename.c_str(), slength, 0, 0, 0, 0);
	char* buf = new char[len];
	WideCharToMultiByte(CP_ACP, 0, _filename.c_str(), slength, buf, len, 0, 0);
	std::string r(buf);
	delete[] buf;

	writer.Key("model");
	writer.Int(_model);
	writer.Key("filepath");
	writer.String(r.c_str());
}

void GetFileCall::ParseResultElement(const rapidjson::Value &result)
{
	// Do nothing, this shouldn't even be called
}



void GetGenerateResultsCall::ParseResultElement(const rapidjson::Value &result)
{
	rapidjson::Value::ConstMemberIterator headerList = result.FindMember("Headers");

	assert(headerList->value.IsArray());
	int numHeaders = headerList->value.Size();
	_domainHeaders.clear();
	for (auto itr = headerList->value.Begin(); itr != headerList->value.End(); ++itr)
	{
		rapidjson::Value::ConstMemberIterator model = itr->FindMember("ModelPtr");
		int stateModelPtr = model->value.GetInt();

		rapidjson::Value::ConstMemberIterator header = itr->FindMember("Header");
		std::string headerString = header->value.GetString();

		_domainHeaders[stateModelPtr] = headerString;
	}

	rapidjson::Value::ConstMemberIterator valueList = result.FindMember("Graph");
	assert(valueList->value.IsArray());
	_graph.clear();
	for (auto itr = valueList->value.Begin(); itr != valueList->value.End(); ++itr)
	{
		_graph.push_back(itr->GetDouble());
	}
}

void GetFitResultsCall::ParseResultElement(const rapidjson::Value &result)
{
	ParameterTreeConverter converter;
	_tree = converter.FromSimpleJSON(result["ParameterTree"]);

	const rapidjson::Value &jsonGraph = result["Graph"];
	_graph.clear();
	for (auto it = jsonGraph.Begin(); it != jsonGraph.End(); it++)
		_graph.push_back(it->GetDouble());
}

CheckCapabilitiesCall::CheckCapabilitiesCall(bool useGPU) :BackendCall("CheckCapabilities")
{
	_useGPU = useGPU;
}

void CheckCapabilitiesCall::WriteOptions(JsonWriter & writer)
{
	writer.Key("useGPU");
	writer.Bool(_useGPU);
}
