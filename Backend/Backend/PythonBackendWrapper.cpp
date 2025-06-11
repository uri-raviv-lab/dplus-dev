#include "PythonBackendWrapper.h"
#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"
#include "AmplitudeCache.h"
#include "UseGPU.h"

#include <iostream>
using namespace std;

/**
 * @file PythonBackendWrapper.cpp
 * @brief Implements the PythonBackendWrapper class, which provides a Python-friendly interface
 *        to backend computational logic, enabling seamless integration with Python (via Cython).
 *
 * The PythonBackendWrapper class is responsible for:
 *  - Exposing backend functionality (model generation, fitting, result retrieval, etc.) to Python code.
 *  - Managing a singleton backend job and its associated resources for Python-driven workflows.
 *  - Handling JSON serialization/deserialization for state and result exchange with Python.
 *  - Converting C++ backend exceptions into Python-compatible runtime_error exceptions with JSON payloads.
 *  - Managing backend initialization, including model metadata and amplitude cache setup.
 *  - Providing methods for starting computations, retrieving results, and managing job state from Python.
 *
 * Key Concepts:
 *  - Designed for use with Cython or other Python/C++ interop layers, avoiding direct C++ exception propagation.
 *  - Maintains a static LocalBackendInfo structure for a single job context, simplifying resource management.
 *  - All backend errors are converted to JSON-formatted runtime_error exceptions for easier handling in Python.
 *  - Supports both 1D and 2D model generation, amplitude/PDB retrieval, and job status queries.
 *
 * Fields:
 *  - static bool _infoInitialized:
 *      Indicates whether the static backend info has been initialized (singleton pattern).
 *  - static BackendWrapper::LocalBackendInfo _info:
 *      Holds the singleton backend state for Python integration, including:
 *        - LocalBackend* local_backend: Pointer to the LocalBackend instance used for all backend operations.
 *        - JobPtr job: Handle to the current backend job, used for model and calculation context.
 *        - LocalBackendParameterTreeConverter* Converter: Pointer to the parameter tree converter for mapping frontend and backend models.
 *
 * Main Methods:
 *  - PythonBackendWrapper (constructor/destructor): Initializes backend info and amplitude cache.
 *  - CheckCapabilities: Verifies hardware and OS capabilities, propagating errors as Python exceptions.
 *  - GetAllModelMetadata: Retrieves all model metadata as a JSON string.
 *  - StartGenerate / StartGenerate2D: Initiates model generation using JSON state and GPU flag.
 *  - GetJobStatus / GetGenerateResults / GetGenerate2DResults: Retrieves job status and results as JSON strings.
 *  - SaveAmplitude / GetAmplitude / GetPDB: Handles amplitude and PDB data retrieval and file saving.
 *  - GetModelPtrs: Returns a list of model pointers for the current job.
 *  - Stop: Signals the backend to halt the current job.
 *  - ConvertException: Converts backend_exception to std::runtime_error with JSON-encoded error info.
 *  - InitializeInfo / InitializeCache: Sets up backend job and amplitude cache for Python use.
 *
 * Threading and Safety:
 *  - Not inherently thread-safe; designed for single-job, single-threaded use from Python.
 *  - Backend resource management is handled via static members for simplicity in Python integration.
 *
 * Error Handling:
 *  - All backend_exception errors are caught and rethrown as std::runtime_error with JSON payloads.
 *  - Python code can parse the JSON error for code and message details.
 *
 * Python Integration:
 *  - All methods are designed to be called from Python, with C++ exceptions translated for Python consumption.
 *  - Avoids C++-specific constructs that are difficult for Cython or Python to handle directly.
 *
 * See PythonBackendWrapper.h for class and method declarations.
 */

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

void PythonBackendWrapper::StartGenerate2D(const std::string state, bool useGPU)
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

		BackendWrapper::StartGenerate2D(doc, _info);
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

std::string PythonBackendWrapper::GetGenerate2DResults()
{
	JsonWriter writer;

	try
	{
		BackendWrapper::GetGenerate2DResults(writer, _info);
		const char* str = writer.GetString();
		return str;
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

std::string PythonBackendWrapper::GetAmplitude(ModelPtr modelPtr)
{
	try
	{
		return _info.local_backend->HandleGetAmplitude(_info.job, _info.Converter->StateToInternal(modelPtr));
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
