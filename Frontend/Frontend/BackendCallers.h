#ifndef BACKEND_CALLERS_H
#define BACKEND_CALLERS_H

#include <Windows.h>
#include <WinInet.h>
#include "Common.h"  // For EXPORTED

// Classes that encapsulate calling the backend.

class BackendCall;

class EXPORTED BackendCaller
{
public:
	virtual ~BackendCaller() { }
	virtual void CallBackend(BackendCall &call, bool runInBackground=false) = 0;
};

class BackendComm;
class EXPORTED LocalBackendCaller : public BackendCaller
{
private:
	BackendComm *_backend;

public:
	LocalBackendCaller();
	virtual ~LocalBackendCaller();

	virtual void CallBackend(BackendCall &call, bool runInBackground = false);
};


class EXPORTED ManagedBackendCaller : public BackendCaller
{
public:
	typedef void (STDCALL *callFunc)(BackendCall &call, bool runInBackground);
	ManagedBackendCaller(callFunc callHandler);
	ManagedBackendCaller(callFunc callHandler, bool isPython);
	bool Python();
	virtual ~ManagedBackendCaller();
	virtual void CallBackend(BackendCall &call, bool runInBackground = false);

private:
	callFunc _handler;
	bool _isPython;

};

#endif