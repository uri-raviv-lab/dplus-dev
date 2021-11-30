#ifndef CALLBACKENDPARSER_H
#define CALLBACKENDPARSER_H

#include "LocalComm.h"
#include <string>
#include <vector>
#include "Common.h"
#include <rapidjson/document.h>

// Classes that encapsulate single calls to the backend

class JsonWriter;

class EXPORTED BackendCall
{
protected:
	BackendCall(std::string funcName);

	virtual void WriteArguments(JsonWriter &writer) { }
	virtual void WriteOptions (JsonWriter &writer){}
	virtual void ParseResultElement(const rapidjson::Value &result) { }

public:
	static void SetClientId(std::string clientId) { _clientId = clientId; }
	std::string GetCallString();
	std::string GetFuncName();
	std::string GetArgs();
	std::string GetOptions();
	void ParseResults(const std::string result);

	ErrorCode GetErrorCode() const { return _errorCode; }
	std::string GetErrorMessage() const { return _errorMessage; }

private:
	static std::string _clientId;

	void ParseResultString(std::string resultString);

	std::string _funcName;
	ErrorCode _errorCode;
	std::string _errorMessage;
};


class GetAllModelMetadataCall : public BackendCall
{
public:
	GetAllModelMetadataCall() : BackendCall("GetAllModelMetadata") { }
	std::string GetMetadata() const { return _metadata; }

protected:
	virtual void ParseResultElement(const rapidjson::Value &result);

private:
	std::string _metadata;
};


class FileContainingCall :public BackendCall
{
protected:
	FileContainingCall(std::string funcName, const std::map<ModelPtr, std::vector<std::wstring>> &fileMap);

	std::vector<std::wstring> _filenames;

public:
	const std::vector<std::wstring> &GetFilenames() const { return _filenames; }
};


class StartFitCall : public FileContainingCall
{
public:
	StartFitCall(const wchar_t *luaScript, const std::vector<double>& x,
		const std::vector<double>& y, const std::vector<int>& mask, const std::map<ModelPtr, std::vector<std::wstring>> &fileMap, bool useGPU = true);

protected:
	virtual void WriteArguments(JsonWriter &writer);
	virtual void WriteOptions(JsonWriter &writer);

private:
	std::wstring _luaScript;
	std::vector<double> _x, _y;
	std::vector<int> _mask;
	bool _useGPU;
};


class StartGenerateCall : public FileContainingCall
{
public:
	StartGenerateCall(const wchar_t *luaScript, const std::vector<double> &x, const std::map<ModelPtr, std::vector<std::wstring>> &filemap, bool useGPU = true);
protected:
	virtual void WriteArguments(JsonWriter &writer);
	virtual void WriteOptions(JsonWriter &writer);
private:
	std::wstring _luaScript;
	std::vector<double> _x;
	bool _useGPU;
};

class CheckCapabilitiesCall : public BackendCall
{
public:
	CheckCapabilitiesCall(bool useGPU = true);//:BackendCall("CheckCapabilities") 
protected:
	virtual void WriteOptions(JsonWriter &writer);
private:
	bool _useGPU;
};

class StopCall : public BackendCall
{
public:
	StopCall() : BackendCall("Stop") { }
};


class GetJobStatusCall : public BackendCall
{
public:
	GetJobStatusCall() : BackendCall("GetJobStatus") { }
	JobStatus GetJobStatus() const { return _jobStatus; }

protected:
	virtual void ParseResultElement(const rapidjson::Value &result);

private:
	JobStatus _jobStatus;
};


class GetFileCall : public BackendCall
{
public:
	//GetFileCall(std::string funcName, ModelPtr model);
	GetFileCall(std::string funcName, ModelPtr model, const wchar_t *filename);

public:
	std::wstring GetFilename() const { return _filename; }

protected:
	virtual void WriteArguments(JsonWriter &writer);
	virtual void ParseResultElement(const rapidjson::Value &result);

private:
	ModelPtr _model;
	std::string _file;
	std::wstring _filename;
};


class GetPDBCall : public GetFileCall
{
public:
	//GetPDBCall(ModelPtr model) : GetFileCall("GetPDB", model) { }
	GetPDBCall(ModelPtr model, const wchar_t *filename) : GetFileCall("GetPDB", model, filename) { }
};


class GetAmplitudeCall : public GetFileCall
{
public:
	//GetAmplitudeCall(ModelPtr model) : GetFileCall("GetAmplitude", model) { }
	GetAmplitudeCall(ModelPtr model, const wchar_t *filename) : GetFileCall("GetAmplitude", model, filename) { }
};


class GetGenerateResultsCall : public BackendCall
{
public:
	GetGenerateResultsCall() : BackendCall("GetGenerateResults") { }

	std::map<int, std::string> GetDomainHeaders() const { return _domainHeaders; }
	std::vector<double> GetGraph() const { return _graph; }

protected:
	virtual void ParseResultElement(const rapidjson::Value &result);

private:
	std::map<int, std::string> _domainHeaders;
	std::vector<double> _graph;
};


class GetFitResultsCall : public BackendCall
{
public:
	GetFitResultsCall() : BackendCall("GetFitResults") { }

	ParameterTree GetParameterTree() const { return _tree; }
	std::vector<double> GetGraph() const { return _graph; }

protected:
	virtual void ParseResultElement(const rapidjson::Value &result);

private:
	ParameterTree _tree;
	std::vector<double> _graph;
};

#endif