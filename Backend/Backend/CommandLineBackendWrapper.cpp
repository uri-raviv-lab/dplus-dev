#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"
#include "CommandLineBackendWrapper.h"
#include "AmplitudeCache.h"
namespace fs = boost::filesystem;
using namespace std;

void CommandLineBackendWrapper::GetAllModelMetadata(JsonWriter &writer)
{
	// Call the base GetAllModelMetadata with the local BackendInfo (call GetBackendInfo to retreive it, pass any client_id)
	BackendWrapper::GetAllModelMetadata(writer, _info);
}

void CommandLineBackendWrapper::StartGenerate(const rapidjson::Value &json, const rapidjson::Value &useGPUJson)
{
	BackendWrapper::SetGPUFlag(useGPUJson);
	BackendWrapper::StartGenerate(json, _info);
}

void CommandLineBackendWrapper::GetJobStatus(JsonWriter &writer)
{
	BackendWrapper::GetJobStatus(writer, _info);
}

void CommandLineBackendWrapper::GetGenerateResults(JsonWriter &writer)
{
	BackendWrapper::GetGenerateResults(writer, _info);
}

void CommandLineBackendWrapper::StartFit(const rapidjson::Value &json, const rapidjson::Value &useGPUJson)
{
	BackendWrapper::SetGPUFlag(useGPUJson);
	BackendWrapper::StartFit(json, _info);
}

void CommandLineBackendWrapper::GetFitResults(JsonWriter &writer)
{
	BackendWrapper::GetFitResults(writer, _info);
}

CommandLineBackendWrapper::CommandLineBackendWrapper()
{
	_info.local_backend = new LocalBackend();
	_info.job = _info.local_backend->HandleCreateJob(L"Single job");
	_info.Converter = new LocalBackendParameterTreeConverter(_info.local_backend, _info.job);
}

void CommandLineBackendWrapper::initializeCache(std::string directory)
{
	AmplitudeCache::initializeCache(directory, _info.Converter);
}


std::vector<ModelPtr> CommandLineBackendWrapper::GetModelPtrs()
{
	return _info.Converter->GetStateModels();
}


CommandLineBackendWrapper::~CommandLineBackendWrapper()
{
	if (_info.Converter)
	{
		delete _info.Converter;
		_info.Converter = nullptr;
	}

	if (_info.local_backend)
	{
		_info.local_backend->HandleDestroyJob(_info.job);
		delete _info.local_backend;
		_info.local_backend = nullptr;
	}
}

const BackendWrapper::LocalBackendInfo CommandLineBackendWrapper::GetBackendInfo(const std::string clientId)
{
	return _info;
}

void CommandLineBackendWrapper::SaveAmplitude(ModelPtr modelPtr, std::string folderpath)
{
	char _Dest[50];
	sprintf(_Dest, "%08d.ampj", modelPtr);
	std::string filename(_Dest);
	std::string  filepath = (boost::filesystem::path(folderpath) / boost::filesystem::path(filename)).string();
	std::string amp = "";

	try
	{
		 _info.local_backend->HandleGetAmplitude(_info.job, _info.Converter->StateToInternal(modelPtr), filepath);
	}
	catch (backend_exception &be)
	{
		//we try getting amp even on models that do not have an amplitude. 
		//hence, for these models, we simply continue without saving a file-- hence the empty catch statement
	}



}

void CommandLineBackendWrapper::SavePDB(ModelPtr modelPtr, std::string folderpath)
{
	char _Dest[50];
	sprintf(_Dest, "%08d.pdb", modelPtr);
	std::string filename(_Dest);
	std::string filepath = (boost::filesystem::path(folderpath) / boost::filesystem::path(filename)).string();
	
	std::string pdb = "";

	try
	{
		std::string pdb = _info.local_backend->HandleGetPDB(_info.job, _info.Converter->StateToInternal(modelPtr));
		if (pdb.length() > 0)
		{
			ofstream myfile(filepath, ios::binary);
			if (myfile.is_open())
			{
				myfile.write(pdb.c_str(), pdb.size());
				myfile.close();
			}

		}
	}
	catch (backend_exception &be)
	{
		//we try getting pdb even on models that don't have
		//hence, for these models, we simply continue without saving a file-- hence the empty catch statement
	}
}

void CommandLineBackendWrapper::CheckCapabilities(bool checkTdr)
{
	BackendWrapper::CheckCapabilities(checkTdr);
}