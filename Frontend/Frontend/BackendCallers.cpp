#include "BackendCallers.h"
#include "BackendCalls.h"
#include "LocalComm.h"
#include "../Backend/Backend/BackendInterface.h"
#include <windows.h>
#include <string>
#include <sstream>

#pragma comment( lib, "wininet" )
using namespace std;

/**
 * @file BackendCallers.cpp
 * @brief Implements backend caller classes that abstract the communication between the frontend and backend systems.
 *
 * This file provides two main classes for backend invocation:
 *  - LocalBackendCaller: Handles direct communication with a local backend instance using a backend communication interface.
 *  - ManagedBackendCaller: Handles backend calls via a user-supplied function pointer, supporting integration with managed or external runtimes (e.g., Python).
 *
 * Responsibilities:
 *  - Encapsulate the details of sending serialized backend requests and receiving responses.
 *  - Parse backend responses and propagate results or errors to the caller.
 *  - Support both synchronous and asynchronous (background) backend calls.
 *  - Allow flexible backend integration, including local C++ backends and managed (e.g., Python) backends.
 *
 * Fields:
 *  - LocalBackendCaller:
 *      - IBackendComm* _backend:
 *          Pointer to the backend communication interface used for sending and receiving backend requests locally.
 *
 *  - ManagedBackendCaller:
 *      - callFunc _handler:
 *          Function pointer or callable object used to invoke backend calls, allowing integration with managed runtimes.
 *      - bool _isPython:
 *          Indicates whether the managed backend is a Python-based implementation (affects integration and behavior).
 *
 * Key Methods:
 *  - CallBackend: Sends a BackendCall to the backend, optionally in the background, and parses the response.
 *  - Python (ManagedBackendCaller): Indicates if the backend is a Python-based implementation.
 *
 * Usage:
 *  - The frontend constructs a BackendCall object, then invokes CallBackend on the appropriate BackendCaller.
 *  - The backend response is parsed and results are made available through the BackendCall interface.
 *
 * See BackendCallers.h for class and method declarations.
 */

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