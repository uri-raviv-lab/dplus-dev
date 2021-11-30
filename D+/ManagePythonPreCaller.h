#pragma once
class BackendCall;
using namespace PythonBackend;

#include <string>
namespace DPlus {

	ref class ManagedPythonPreCaller
	{
	public:
		ManagedPythonPreCaller(std::string exe_dir);
		ManagedPythonPreCaller(std::string exe_dir, std::string session);
		~ManagedPythonPreCaller() {}
		delegate void CallBackendDelegate(BackendCall &call, bool runInBackground);
		CallBackendDelegate ^GetDelegate() { return _callBackendDelegate; }
	protected:
		CallBackendDelegate ^ _callBackendDelegate;
		PythonBackend::PythonBackendCaller^ pythonCaller;
		void PerformCall(BackendCall &call, bool runInBackground);
	};

}
