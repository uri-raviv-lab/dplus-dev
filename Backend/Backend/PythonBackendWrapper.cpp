#include "PythonBackendWrapper.h"
#include "LocalBackendParameterTree.h"
#include "LocalBackend.h"

#include <iostream>
using namespace std;

PythonBackendWrapper::PythonBackendWrapper()
{
	InitializeInfo();
	cout << "PythonBackendWrapper Created" << endl;
}

PythonBackendWrapper::~PythonBackendWrapper()
{
	cout << "PythonBackendWrapper destroyed" << endl;
}

void PythonBackendWrapper::CheckCapabilities(bool checkTdr)
{
	cout << "PythonBackendWrapper:CheckCapabilities called" << endl;
	BackendWrapper::CheckCapabilities(checkTdr);
}

bool PythonBackendWrapper::_infoInitialized = false;
BackendWrapper::LocalBackendInfo PythonBackendWrapper::_info;

void PythonBackendWrapper::InitializeInfo()
{
	if (_infoInitialized) {
		return;
	}

	cout << "Initializing PythonBackendWrapper::Info" << endl;

	_info.local_backend = new LocalBackend();
	_info.job = _info.local_backend->HandleCreateJob(L"Single job");
	_info.Converter = new LocalBackendParameterTreeConverter(_info.local_backend, _info.job);
}
