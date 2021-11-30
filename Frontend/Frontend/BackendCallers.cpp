#include "BackendCallers.h"
#include "BackendCalls.h"
#include "LocalComm.h"
#include "../Backend/Backend/BackendInterface.h"
#include <windows.h>
#include <string>
#include <sstream>

#pragma comment( lib, "wininet" )
using namespace std;

LocalBackendCaller::LocalBackendCaller()
{
	_backend = CreateBackendComm();
}

LocalBackendCaller::~LocalBackendCaller()
{
	delete _backend;
}

void LocalBackendCaller::CallBackend(BackendCall &call, bool runInBackground)
{
	std::string request = call.GetCallString();
	std::string response = _backend->CallBackend(request);
	call.ParseResults(response);
}
ManagedBackendCaller::ManagedBackendCaller(callFunc callHandler)
{
	_handler = callHandler;
	_isPython = false;
}

ManagedBackendCaller::ManagedBackendCaller(callFunc callHandler, bool isPython)
{
	_handler = callHandler;
	_isPython = isPython;
}

bool ManagedBackendCaller::Python()
{
	return _isPython;
}

ManagedBackendCaller::~ManagedBackendCaller()
{

}

void ManagedBackendCaller::CallBackend(BackendCall &call, bool runInBackground)
{
	_handler(call, runInBackground);
}