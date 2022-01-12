#include "PythonBackendWrapper.h"
#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"

#include <iostream>
using namespace std;

PythonBackendWrapper::PythonBackendWrapper()
{
	InitializeInfo();
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
