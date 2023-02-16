#include <string>

#include "LocalBackendParameterTree.h"
#include "../../Conversions/Conversions.h"
#include "MetadataSerializer.h"
#include "../../Conversions/base64.h"

#include "BackendWrapper.h"
#include "LocalBackend.h"

#include <rapidjson/document.h>
#include "../../Conversions/JsonWriter.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include "UseGPU.h"
#include "../backend_version.h"
using namespace rapidjson;
using namespace std;


std::string BackendWrapper::CallBackend(std::string json)
{
    JsonWriter writer;
	rapidjson::Document doc;
	Value::ConstMemberIterator clientDataMember;
	bool clientDataPresent = false;

	try
	{

		doc.Parse(json.c_str());
		
		string	functionName = "";
		string  functionArgs = "";

		//Object will never be empty, since the response will optionally contain client data and either error or result
		writer.StartObject();

		if (doc.HasParseError())
			throw backend_exception(ERROR_ILLEGAL_JSON);

		Value::ConstMemberIterator clientIdMember = doc.FindMember("client-id");
		clientDataMember = doc.FindMember("client-data");
		Value::ConstMemberIterator functionMember = doc.FindMember("function");
		Value::ConstMemberIterator argsMember = doc.FindMember("args");
		Value::ConstMemberIterator optionsMember = doc.FindMember("options");
		//If client data is part of the request, include it in the response
		if (clientDataMember != doc.MemberEnd())
		{
			clientDataPresent = true;
			WriteResponseClientData(writer, clientDataMember);
		}
		
		//ClientId must be included in request
		if (clientIdMember == doc.MemberEnd())
		{		
			throw backend_exception(ERROR_INVALIDARGS, "No client-id specified");
		}
		if (!clientIdMember->value.IsString())
		{
			throw backend_exception(ERROR_INVALIDARGS, "client-id must be a string");
		}

		//Function name must be included in request
		if (functionMember == doc.MemberEnd())
		{
			throw backend_exception(ERROR_INVALIDARGS, "No function name specified");
		}

		//get function name and args (both strings)
		if (!functionMember->value.IsString())
		{
			throw backend_exception(ERROR_INVALIDARGS, "Badly formatted function name (should be a string)");
		}
		else
		{
			functionName = functionMember->value.GetString();
		}

		if (argsMember == doc.MemberEnd())
			throw backend_exception(ERROR_INVALIDARGS, "No args specified");

		SetGPUFlag(optionsMember->value);
		if (argsMember == doc.MemberEnd())
			throw backend_exception(ERROR_INVALIDARGS, "No args specified");


		//Call the backend function
		writer.Key("result");
		const LocalBackendInfo &backend = GetBackendInfo(clientIdMember->value.GetString());
		CallBackendFunction(functionName, argsMember->value, writer, backend);  // Will write the result or throw an exception

		//If we get to here without exception, then all is good.
		WriteResponseError(writer, OK, g_errorStrings[OK]);		
	}
	catch (backend_exception &be)
	{
		JsonWriter writer;   // Return just the error. For this we need a new writer because the old one may be in the middle of writing the result
		writer.StartObject();
		if (clientDataPresent)
		{
			WriteResponseClientData(writer, clientDataMember);
		}
		WriteResponseError(writer, be.GetErrorCode(), be.GetErrorMessage());
		writer.EndObject();

		return writer.GetString();
	}

	writer.EndObject();
	return writer.GetString();
}



void BackendWrapper::WriteResponseClientData(JsonWriter &writer, const Value::ConstMemberIterator &clientDataMember)
{
	writer.Key("client-data");
	if (!clientDataMember->value.Accept(writer))
	{
		CheckErrorCode(ERROR_INVALIDARGS);
	}
}

void BackendWrapper::WriteResponseError(JsonWriter &writer, int errorCode, string errorMsg)
{
	writer.Key("error");
	writer.StartObject();

	writer.Key("code");
	writer.Int(errorCode);

	writer.Key("message");
	writer.String(errorMsg.c_str());

	writer.EndObject();
}

void BackendWrapper::CallBackendFunction(const string functionName, const Value &args, JsonWriter &writer, const BackendWrapper::LocalBackendInfo &backend)
{
	//These functions can throw backend_exception

	if (functionName == "GetAllModelMetadata")
	{
		GetAllModelMetadata(writer, backend);
	}
	else if (functionName == "GetJobStatus")
	{
		GetJobStatus(writer, backend);
	}
	else if (functionName == "Stop")
	{	
		Stop(backend);
		writer.StartObject();   // We must write something as the result, an empty object will do
		writer.EndObject();
	}
	else if (functionName == "StartGenerate")
	{
		StartGenerate(args, backend); //throws an exception if there is an error
		writer.StartObject();   // We must write something as the result, an empty object will do
		writer.EndObject();
	}
	else if (functionName == "GetGenerateResults")
	{
		GetGenerateResults(writer, backend);
	}
	else if (functionName == "GetGenerate2DResults")
	{
		GetGenerate2DResults(writer, backend);
	}
	else if (functionName == "StartFit")
	{
		StartFit(args, backend);
		writer.StartObject();   // We must write something as the result, an empty object will do
		writer.EndObject();
	}
	else if (functionName == "GetFitResults")
	{
		GetFitResults(writer, backend);
	}
	else if (functionName == "GetAmplitude")
	{
		GetAmplitude(args, writer, backend);
	}
	else if (functionName == "GetPDB")
	{
		GetPDB(args, writer, backend);
	}	
	else if (functionName == "CheckCapabilities")
	{
		CheckCapabilities(g_useGPU);
		writer.StartObject();   // We must write something as the result, an empty object will do
		writer.EndObject();
	}
	else
	{
		// throw another backend_exception
		CheckErrorCode(ERROR_UNSUPPORTED);
	}
}


void BackendWrapper::CheckErrorCode(ErrorCode err)
{
	if (OK != err)
	{
		throw backend_exception(err, g_errorStrings[err]);
	}
}

void BackendWrapper::CheckCapabilities(bool checkTdr)
{
	checkAVX();
	
	if (checkTdr) {
		checkGPU();
		checkTdrLevel();
	}
	
}
void BackendWrapper::checkTdrLevel()
{
#ifdef _WIN32
#pragma comment(lib, "advapi32")

	HKEY root = HKEY_LOCAL_MACHINE;
	wstring key = L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers";
	wstring name = L"TdrLevel";
	HKEY hKey;
	if (RegOpenKeyEx(root, key.c_str(), 0, KEY_READ, &hKey) != ERROR_SUCCESS)
		throw backend_exception(ERROR_TDR_LEVEL);
	DWORD cbData = 32;// 32 bytes - the size of TdrLevel
	unsigned int Target = 0;

	if (RegQueryValueEx(hKey, name.c_str(), NULL, NULL, reinterpret_cast<LPBYTE>(&Target), &cbData) != ERROR_SUCCESS)
	{
		RegCloseKey(hKey);
		throw backend_exception(ERROR_TDR_LEVEL);
	}
	RegCloseKey(hKey);
	if (Target != 0)
		throw backend_exception(ERROR_TDR_LEVEL);
#endif
}
void BackendWrapper::checkAVX()
{
#ifdef _WIN32
	bool avxSupported = false;

	// If Visual Studio 2010 SP1 or later
#if (_MSC_FULL_VER >= 160040219)
	// Checking for AVX requires 3 things:
	// 1) CPUID indicates that the OS uses XSAVE and XRSTORE
	//     instructions (allowing saving YMM registers on context
	//     switch)
	// 2) CPUID indicates support for AVX
	// 3) XGETBV indicates the AVX registers will be saved and
	//     restored on context switch
	//
	// Note that XGETBV is only available on 686 or later CPUs, so
	// the instruction needs to be conditionally run.
	int cpuInfo[4];
	__cpuid(cpuInfo, 1);

	bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
	bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

	if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
	{
		// Check if the OS will save the YMM registers
		unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
		avxSupported = (xcrFeatureMask & 0x6) || false;
	}
#endif

	if (!avxSupported)
		throw backend_exception(AVX);
#endif
}
void BackendWrapper::checkGPU()
{
	// there is no need to check the tdr level if there is no GPU in the computer
	int devCount;
	cudaError_t t = cudaGetDeviceCount(&devCount);
	if (devCount <= 0 || t != cudaSuccess)
		throw backend_exception(ERROR_NO_GPU);
}

BackendWrapper::BackendWrapper()
{
}

BackendWrapper::~BackendWrapper()
{
}

void BackendWrapper::GetAllModelMetadata(JsonWriter &writer, const LocalBackendInfo &backend)
{
	MetadataSerializer ser(backend.local_backend);
	ser.Serialize(writer);
}
void BackendWrapper::SetGPUFlag(const rapidjson::Value &json)
{
	if (!json.IsObject() || !json.HasMember("useGPU"))
		return;
	const rapidjson::Value &useGPU = json["useGPU"];
	//if (optionsMember != doc.MemberEnd() && json->value.HasMember("useGPU"))
	g_useGPU = useGPU.GetBool();
}
void BackendWrapper::StartGenerate(const rapidjson::Value &json, const LocalBackendInfo &backend)
{
	std::cout <<"backend version:"<<BACKEND_VERSION<< endl;

	 CheckCapabilities(g_useGPU);
	 string usgG = g_useGPU == true ? "True " : "False ";
	std::cout << "Use GPU flag:" << usgG<< endl;
	const rapidjson::Value &xs = json["x"];
	//Translate xs into a vector<double>
	std::vector<double> x;

	for (auto itr = xs.Begin(); itr != xs.End(); ++itr)
	{
		x.push_back((itr)->GetDouble());
	}

	rapidjson::Value state;
	rapidjson::Document dummy_allocator;
	state.CopyFrom(json["state"], dummy_allocator.GetAllocator()); // json["state"]

	{
		rapidjson::Value& domainPrefs = state["DomainPreferences"];
		domainPrefs["qMax"].SetDouble(x.back());
/*
		rapidjson::StringBuffer buffer;

		buffer.Clear();

		rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
		state.Accept(writer);

		std::cout << buffer.GetString() << "\n\n";
*/
	}

	try
	{
		ParameterTree pt = backend.Converter->FromStateJSON(state);
		//build the fitting parameters from the json string
		FittingProperties fp = FittingPropertiesFromStateJSON(state);
		if (pt.GetNumSubModels() == 0)
		{
			throw backend_exception(ERROR_INVALIDARGS);
		}


		int rc = backend.local_backend->HandleGenerate(backend.job, pt, x, fp);
		if (rc != OK)
			throw backend_exception(rc);
	}
	catch (const std::exception& e)
	{
		throw backend_exception(ERROR_INVALIDPARAMTREE, e.what());
	}
	catch (backend_exception& be)
	{
		throw be;
	}

	

}

void BackendWrapper::StartGenerate2D(const rapidjson::Value& json, const LocalBackendInfo& backend)
{
	std::cout << "backend version:" << BACKEND_VERSION << endl;

	CheckCapabilities(g_useGPU);
	string usgG = g_useGPU == true ? "True " : "False ";
	std::cout << "Use GPU flag:" << usgG << endl;
	const rapidjson::Value& xs = json["x"];
	//Translate xs into a vector<double>
	std::vector<double> x;

	for (auto itr = xs.Begin(); itr != xs.End(); ++itr)
	{
		x.push_back((itr)->GetDouble());
	}

	rapidjson::Value state;
	rapidjson::Document dummy_allocator;
	state.CopyFrom(json["state"], dummy_allocator.GetAllocator()); // json["state"]

	{
		rapidjson::Value& domainPrefs = state["DomainPreferences"];
		domainPrefs["qMax"].SetDouble(x.back());
	}

	try
	{
		ParameterTree pt = backend.Converter->FromStateJSON(state);
		//build the fitting parameters from the json string
		FittingProperties fp = FittingPropertiesFromStateJSON(state);
		if (pt.GetNumSubModels() == 0)
		{
			throw backend_exception(ERROR_INVALIDARGS);
		}


		int rc = backend.local_backend->HandleGenerate2D(backend.job, pt, x, fp);
		if (rc != OK)
			throw backend_exception(rc);
	}
	catch (const std::exception& e)
	{
		throw backend_exception(ERROR_INVALIDPARAMTREE, e.what());
	}
	catch (backend_exception& be)
	{
		throw be;
	}



}

void BackendWrapper::StartFit(const rapidjson::Value &json, const LocalBackendInfo &backend)
{
	std::cout << "backend version:" << BACKEND_VERSION << endl;

	CheckCapabilities(g_useGPU);
	string usgG = g_useGPU == true ? "True " : "False ";
	std::cout << "Use GPU flag:" << usgG << endl;
	//build the parameter tree from the json string
	const rapidjson::Value &state = json["state"];
	ParameterTree pt = backend.Converter->FromStateJSON(state);
	//build the fitting parameters from the json string
	FittingProperties fp = FittingPropertiesFromStateJSON(state);

	if (pt.GetNumSubModels() == 0)
	{
		throw backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]);
	}


	const rapidjson::Value &xs = json["x"];
	//Translate xs into a vector<double>
	std::vector<double> x;

	for (auto itr = xs.Begin(); itr != xs.End(); ++itr)
	{
		x.push_back((itr)->GetDouble());
	}

	const rapidjson::Value &ys = json["y"];
	std::vector<double> y;

	for (auto itr = ys.Begin(); itr != ys.End(); ++itr)
	{
		y.push_back((itr)->GetDouble());
	}

	const rapidjson::Value &masks = json["mask"];
	std::vector<int> mask;

	for (auto itr = masks.Begin(); itr != masks.End(); ++itr)
	{
		mask.push_back((itr)->GetInt());
	}

	// Check that there are no negative intensity values
	{
		Eigen::Map<Eigen::ArrayXd> ymap(y.data(), y.size());
		Eigen::Map<Eigen::ArrayXi> maskmap(mask.data(), mask.size());

		int number_of_not_masked_negative_values = ((ymap < 0).cast<int>() > maskmap).count();
		if (number_of_not_masked_negative_values > 0)
		{
			throw backend_exception(ERROR_UNSUCCESSFULLFITTING, std::string("There are "
				+ std::to_string(number_of_not_masked_negative_values) +
				" negative intensity values. Negative values are not allowed when fitting.").c_str());
		}

	}

	int rc = backend.local_backend->HandleFit(backend.job, pt, x, y, mask, fp);
	if (rc != OK)
		throw backend_exception(rc);
	return;
}

void BackendWrapper::WriteGraph(JsonWriter &writer, const LocalBackendInfo &backend)
{
	writer.StartArray();

	int gSize = backend.local_backend->HandleGetGraphSize(backend.job);
	std::vector<double> graph(gSize);

	if (graph.size() < 1 || !backend.local_backend->HandleGetGraph(backend.job, &graph[0], gSize)) {
		throw backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]);
	}

	for (int i = 0; i < gSize; ++i)
	{
		writer.Double(graph[i]);
	}

	writer.EndArray();

}

void BackendWrapper::Write2DGraph(JsonWriter& writer, const LocalBackendInfo& backend)
{
	writer.StartArray();

	int rows, cols;
	backend.local_backend->HandleGet2DGraphSize(backend.job, rows, cols);
	MatrixXd graph(rows,cols);

	if (rows<1 || cols<1 || !backend.local_backend->HandleGet2DGraph(backend.job, graph, rows, cols)) {
		throw backend_exception(ERROR_INVALIDARGS, g_errorStrings[ERROR_INVALIDARGS]);
	}

	for (int i = 0; i < rows; ++i)
	{
		writer.StartArray();
		for (int j = 0; j < rows; ++j)
		{
			writer.Double(graph(i,j));
		}
		writer.EndArray();
	}

	writer.EndArray();

}

void BackendWrapper::GetGenerateResults(JsonWriter &writer, const LocalBackendInfo &backend)
{
	writer.StartObject();

	//header
	writer.Key("Headers");
	writer.StartArray();

	int len = 999999;		//hardcoded like this in original code ...
	char *grry = new char[len];

	// Going over all the models in the map, create a map in the json of state model ptr to model header string
	auto stateModels = backend.Converter->GetStateModels();
	for (auto itr = stateModels.begin(); itr != stateModels.end(); ++itr)
	{
		int internalModelPtr = backend.Converter->StateToInternal(*itr);

		//If the internal model is not found (this is an existing behaviour), then just skip header for this model
		try
		{
			backend.local_backend->HandleGetDomainHeader(backend.job, internalModelPtr, grry, len);
		}
		catch (backend_exception be)
		{
			if (be.GetErrorCode() == ERROR_MODELNOTFOUND)
				continue;
			throw;
		}
		writer.StartObject();
		writer.Key("ModelPtr");
		writer.Int(*itr);
		writer.Key("Header");
		writer.String(grry);
		writer.EndObject();
	}
	writer.EndArray();

	//results
	writer.Key("Graph");
	WriteGraph(writer, backend);
	writer.EndObject();
}

void BackendWrapper::GetGenerate2DResults(JsonWriter& writer, const LocalBackendInfo& backend)
{
	writer.StartObject();

	//header
	writer.Key("Headers");
	writer.StartArray();

	int len = 999999;		//hardcoded like this in original code ...
	char* grry = new char[len];

	// Going over all the models in the map, create a map in the json of state model ptr to model header string
	auto stateModels = backend.Converter->GetStateModels();
	for (auto itr = stateModels.begin(); itr != stateModels.end(); ++itr)
	{
		int internalModelPtr = backend.Converter->StateToInternal(*itr);

		//If the internal model is not found (this is an existing behaviour), then just skip header for this model
		try
		{
			backend.local_backend->HandleGetDomainHeader(backend.job, internalModelPtr, grry, len);
		}
		catch (backend_exception be)
		{
			if (be.GetErrorCode() == ERROR_MODELNOTFOUND)
				continue;
			throw;
		}
		writer.StartObject();
		writer.Key("ModelPtr");
		writer.Int(*itr);
		writer.Key("Header");
		writer.String(grry);
		writer.EndObject();
	}
	writer.EndArray();

	//results
	writer.Key("2DGraph");
	Write2DGraph(writer, backend);
	writer.EndObject();
}


void BackendWrapper::GetFitResults(JsonWriter &writer, const LocalBackendInfo &backend)
{
	ParameterTree tree;
	ErrorCode err = backend.local_backend->HandleGetResults(backend.job, tree);
	if (OK != err)
	{
		throw(backend_exception(err, g_errorStrings[err]));
	}

	writer.StartObject();
	writer.Key("ParameterTree");
	backend.Converter->WriteSimpleJSON(writer, tree);

	writer.Key("Graph");
	WriteGraph(writer, backend);
	writer.EndObject();
}

void BackendWrapper::GetJobStatus(JsonWriter &writer, const LocalBackendInfo &backend)
{
	JobStatus jobStatus = backend.local_backend->GetJobStatus(backend.job);

	WriteJobStatusJSON(writer, jobStatus);
}

void BackendWrapper::Stop(const LocalBackendInfo &backend) {
	backend.local_backend->HandleStop(backend.job);
}

void BackendWrapper::GetAmplitude(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend) {

	const rapidjson::Value &model = json["model"];
	const rapidjson::Value &path = json["filepath"];

	ModelPtr modelPtr = model.GetInt();
	std::string filepath = path.GetString(); 
	backend.local_backend->HandleGetAmplitude(backend.job, backend.Converter->StateToInternal(modelPtr), filepath);
	writer.String(filepath.c_str());
	
}

void BackendWrapper::GetPDB(const rapidjson::Value &json, JsonWriter &writer, const LocalBackendInfo &backend) {
	const rapidjson::Value &model = json["model"];
	const rapidjson::Value &path = json["filepath"];

	ModelPtr modelPtr = model.GetInt();

	std::string pdb = backend.local_backend->HandleGetPDB(backend.job, backend.Converter->StateToInternal(modelPtr));
	//writer.String(base64_encode(amp).c_str());

	std::string filepath = path.GetString();
	ofstream myfile(filepath, ios::binary);
	if (myfile.is_open()) {
		myfile.write(pdb.c_str(), pdb.size());
		myfile.close();
	}

	writer.String(filepath.c_str());
}
