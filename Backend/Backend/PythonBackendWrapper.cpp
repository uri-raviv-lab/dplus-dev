#include "PythonBackendWrapper.h"
#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"
#include "AmplitudeCache.h"
#include "UseGPU.h"

#include <iostream>
using namespace std;

PythonBackendWrapper::PythonBackendWrapper()
{
	InitializeInfo();
	InitializeCache();
}

PythonBackendWrapper::~PythonBackendWrapper()
{
}

void PythonBackendWrapper::CheckCapabilities(bool checkTdr)
{
	try
	{
		BackendWrapper::CheckCapabilities(checkTdr);
	}
	catch (backend_exception& be)
	{
		auto re = ConvertException(be);
		throw re;
	}
}

std::string PythonBackendWrapper::GetAllModelMetadata()
{
	JsonWriter writer;
	try
	{
		BackendWrapper::GetAllModelMetadata(writer, _info);
		return writer.GetString();
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

// Converts a backend_exception into a runtime_errpr that encodes the code and message in a JSON.
// This is done because Cython has a hard time handling C++ exceptions
std::runtime_error PythonBackendWrapper::ConvertException(const backend_exception& be)
{
	JsonWriter writer;
	writer.StartObject();
	writer.Key("code");
	writer.Int(be.GetErrorCode());
	writer.Key("message");
	writer.String(be.GetErrorMessage().c_str());
	writer.EndObject();

	return runtime_error(writer.GetString());
}

bool PythonBackendWrapper::_infoInitialized = false;
BackendWrapper::LocalBackendInfo PythonBackendWrapper::_info;

void PythonBackendWrapper::InitializeInfo()
{
	if (_infoInitialized) {
		return;
	}

	_info.local_backend = new LocalBackend();
	_info.job = _info.local_backend->HandleCreateJob(L"Single job");
	_info.Converter = new LocalBackendParameterTreeConverter(_info.local_backend, _info.job);
}

void PythonBackendWrapper::InitializeCache()
{
	AmplitudeCache::initializeCache(_info.Converter);
}

void PythonBackendWrapper::InitializeCache(std::string cacheDir)
{
	AmplitudeCache::initializeCache(cacheDir, _info.Converter);
}


void PythonBackendWrapper::StartGenerate(const std::string state, bool useGPU)
{
	rapidjson::Document doc;

	try
	{
		doc.Parse(state.c_str());
		if (doc.HasParseError())
		{
			throw backend_exception(ERROR_ILLEGAL_JSON);
		}
		g_useGPU = useGPU;

		BackendWrapper::StartGenerate(doc, _info);
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

std::string PythonBackendWrapper::GetJobStatus()
{
	JsonWriter writer;

	try
	{
		BackendWrapper::GetJobStatus(writer, _info);
		return writer.GetString();
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

std::string PythonBackendWrapper::GetGenerateResults()
{
	JsonWriter writer;

	try
	{
		BackendWrapper::GetGenerateResults(writer, _info);
		return writer.GetString();
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

void PythonBackendWrapper::SaveAmplitude(ModelPtr modelPtr, std::string path)
{
	try
	{
		_info.local_backend->HandleGetAmplitude(_info.job, _info.Converter->StateToInternal(modelPtr), path);
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

std::string PythonBackendWrapper::GetPDB(ModelPtr modelPtr)
{
	try
	{
		std::string pdb_str = _info.local_backend->HandleGetPDB(_info.job, modelPtr);
		return pdb_str;
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

std::vector<ModelPtr> PythonBackendWrapper::GetModelPtrs()
{
	try
	{
		return _info.Converter->GetStateModels();
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}

void PythonBackendWrapper::Stop() 
{
	try
	{
		_info.local_backend->HandleStop(_info.job);
	}
	catch (backend_exception& be)
	{
		throw ConvertException(be);
	}
}
