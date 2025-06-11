#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"
#include "CommandLineBackendWrapper.h"
#include "AmplitudeCache.h"
namespace fs = boost::filesystem;
using namespace std;

/**
 * @file CommandLineBackendWrapper.cpp
 * @brief Implements the CommandLineBackendWrapper class, which provides a command-line-oriented backend interface
 *        for model management, job execution, and result export, wrapping a local backend instance.
 *
 * The CommandLineBackendWrapper module is responsible for:
 *  - Managing a local backend and job for command-line or batch processing scenarios.
 *  - Providing methods to start model generation and fitting jobs, retrieve job status, and export results.
 *  - Handling model metadata queries and backend capability checks.
 *  - Exporting amplitude and PDB data for models to the filesystem.
 *  - Managing the lifecycle of the local backend, job, and parameter tree converter.
 *
 * Key Concepts:
 *  - CommandLineBackendWrapper: A wrapper class for backend operations, designed for command-line or scripting use.
 *  - LocalBackend: The underlying backend implementation used for all calculations and data management.
 *  - LocalBackendParameterTreeConverter: Converts between internal and external model representations.
 *  - BackendWrapper::LocalBackendInfo: Structure holding backend, job, and converter pointers for context.
 *  - AmplitudeCache: Used for caching amplitude calculations to disk.
 *
 * Fields:
 *  - BackendWrapper::LocalBackendInfo _info:
 *      Holds pointers to the local backend, job handle, and parameter tree converter.
 *      - LocalBackend* local_backend: Pointer to the local backend instance.
 *      - job_t job: Handle to the current job.
 *      - LocalBackendParameterTreeConverter* Converter: Converts between state and internal model representations.
 *  - (Inherited) BackendWrapper methods and members for job and model management.
 *
 * Main Methods:
 *  - CommandLineBackendWrapper(): Constructor; initializes the local backend, job, and converter.
 *  - ~CommandLineBackendWrapper(): Destructor; cleans up backend, job, and converter resources.
 *  - void GetAllModelMetadata(JsonWriter& writer): Writes all model metadata to a JSON writer.
 *  - void StartGenerate(const rapidjson::Value& json, const rapidjson::Value& useGPUJson): Starts a model generation job.
 *  - void GetJobStatus(JsonWriter& writer): Writes the current job status to a JSON writer.
 *  - void GetGenerateResults(JsonWriter& writer): Writes generation results to a JSON writer.
 *  - void StartFit(const rapidjson::Value& json, const rapidjson::Value& useGPUJson): Starts a model fitting job.
 *  - void GetFitResults(JsonWriter& writer): Writes fitting results to a JSON writer.
 *  - void initializeCache(std::string directory): Initializes the amplitude cache for the backend.
 *  - std::vector<ModelPtr> GetModelPtrs(): Returns pointers to all models in the current state.
 *  - const BackendWrapper::LocalBackendInfo GetBackendInfo(const std::string clientId): Returns backend info/context.
 *  - void SaveAmplitude(ModelPtr modelPtr, std::string folderpath): Saves amplitude data for a model to disk.
 *  - void SavePDB(ModelPtr modelPtr, std::string folderpath, bool electron): Saves PDB data for a model to disk.
 *  - void CheckCapabilities(bool checkTdr): Checks backend capabilities (e.g., GPU support).
 *
 * Threading and Safety:
 *  - Not inherently thread-safe; designed for single-job, command-line or batch use.
 *  - Resource management and cleanup are handled in the destructor.
 *
 * Error Handling:
 *  - Catches and ignores backend exceptions when exporting amplitude or PDB data for models that may not support them.
 *  - Ensures proper cleanup of backend and converter resources.
 *
 * Dependencies:
 *  - LocalBackend and LocalBackendParameterTree for backend operations and parameter conversion.
 *  - BackendWrapper for base backend interface and context management.
 *  - AmplitudeCache for amplitude result caching.
 *  - boost::filesystem for file path manipulation.
 *  - rapidjson and JsonWriter for JSON serialization.
 *
 * See CommandLineBackendWrapper.h for class and method declarations.
 */

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

void CommandLineBackendWrapper::SavePDB(ModelPtr modelPtr, std::string folderpath, bool electron)
{
	char _Dest[50];
	sprintf(_Dest, "%08d.pdb", modelPtr);
	std::string filename(_Dest);
	std::string filepath = (boost::filesystem::path(folderpath) / boost::filesystem::path(filename)).string();
	
	std::string pdb = "";

	try
	{
		std::string pdb = _info.local_backend->HandleGetPDB(_info.job, _info.Converter->StateToInternal(modelPtr), electron);
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