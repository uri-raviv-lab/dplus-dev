#include "ManagePythonPreCaller.h"
#include "../../Frontend/Frontend/BackendCalls.h"
#include <msclr\marshal_cppstd.h>

/**
 * @file ManagePythonPreCaller.cpp
 * @brief Implements the ManagedPythonPreCaller class, which bridges C++/CLI and a managed Python backend,
 *        enabling backend calls from the DPlus application to Python via .NET interop.
 *
 * The ManagePythonPreCaller module is responsible for:
 *  - Wrapping and forwarding backend calls from C++/CLI to a managed Python backend using .NET delegates.
 *  - Translating function names and arguments between native and managed representations.
 *  - Handling backend call results and error reporting, including marshaling between std::string and System::String^.
 *  - Managing the lifecycle of the Python backend caller and session.
 *
 * Key Concepts:
 *  - ManagedPythonPreCaller: C++/CLI class that acts as a bridge between native C++ code and the managed Python backend.
 *  - PythonBackend::PythonBackendCaller: Managed class that executes backend calls in Python.
 *  - BackendCall: Encapsulates a backend function call, including function name, arguments, and options.
 *  - CallBackendDelegate: .NET delegate for invoking backend calls asynchronously or synchronously.
 *  - msclr::interop::marshal_context: Used for marshaling strings between managed and native code.
 *
 * Fields:
 *  - PythonBackend::PythonBackendCaller^ pythonCaller:
 *      Managed object responsible for executing backend calls in Python.
 *  - CallBackendDelegate^ _callBackendDelegate:
 *      Delegate for invoking backend calls from C++/CLI.
 *
 * Main Methods:
 *  - ManagedPythonPreCaller(std::string exe_dir):
 *      Constructor; initializes the Python backend caller with the given executable directory.
 *  - ManagedPythonPreCaller(std::string exe_dir, std::string session):
 *      Constructor; initializes the Python backend caller with the given executable directory and session.
 *  - void PerformCall(BackendCall& _call, bool runInBackground):
 *      Marshals call data to managed types, invokes the backend, and parses results or errors.
 *  - functions_code funcHash(String^ inString):
 *      Maps backend function names to enum codes for internal dispatching.
 *
 * Threading and Safety:
 *  - Designed for use in a managed, single-threaded context; backend calls may be run in background if needed.
 *  - Uses .NET delegates for safe invocation of managed code from C++/CLI.
 *
 * Error Handling:
 *  - Catches managed exceptions, logs error messages, and returns error results to the caller.
 *  - Ensures all exceptions are converted to a JSON error result for consistent error handling.
 *
 * Dependencies:
 *  - ManagePythonPreCaller.h for class and method declarations.
 *  - BackendCalls.h for BackendCall structure.
 *  - PythonBackendCaller.cs and CSharpManagedBackendCall.cs for managed backend execution.
 *  - msclr::interop for string marshaling.
 *  - System and PythonBackend .NET namespaces.
 *
 * See ManagePythonPreCaller.h for class and method declarations.
 */

namespace DPlus {
	using namespace System;
	using namespace PythonBackend;

	enum functions_code {
		metadata,
		start_generate,
		get_generate,
		start_fit,
		get_fit,
		job_status,
		stop,
		pdb,
		amplitude
	};
	functions_code funcHash(String ^ inString) {
		if (inString == "GetAllModelMetadata") return metadata;
		if (inString == "StartGenerate") return start_generate;
		if (inString == "GetGenerateResults") return get_generate;
		if (inString == "StartFit") return start_fit;
		if (inString == "GetFitResults") return get_fit;
		if (inString == "GetJobStatus") return job_status;
		if (inString == "GetPDB") return pdb;
		if (inString == "GetAmplitude") return amplitude;
		if (inString == "Stop") return stop;
	}
	ManagedPythonPreCaller::ManagedPythonPreCaller(std::string exe_dir)
	{
		_callBackendDelegate = gcnew CallBackendDelegate(this, &DPlus::ManagedPythonPreCaller::PerformCall);
		String^ exe = gcnew String(exe_dir.c_str());
		pythonCaller = gcnew PythonBackend::PythonBackendCaller(exe);
	}
	ManagedPythonPreCaller::ManagedPythonPreCaller(std::string exe_dir , std::string session)
	{
		_callBackendDelegate = gcnew CallBackendDelegate(this, &DPlus::ManagedPythonPreCaller::PerformCall);
		String^ exe = gcnew String(exe_dir.c_str());
		String^ sess = gcnew String(session.c_str());
		pythonCaller = gcnew PythonBackend::PythonBackendCaller(exe, sess);
	}
	void ManagedPythonPreCaller::PerformCall(BackendCall & _call, bool runInBackground)
	{
		msclr::interop::marshal_context context;
		try {

			PythonBackend::CSharpManagedBackendCall^ csharpBackendCaller = gcnew PythonBackend::CSharpManagedBackendCall();
			std::string fcn = _call.GetFuncName();
			csharpBackendCaller->FuncName = gcnew String(fcn.c_str());
			std::string st = _call.GetArgs();
			csharpBackendCaller->Args = gcnew String(st.c_str());
			std::string opt = _call.GetOptions();
			csharpBackendCaller->Options = gcnew String(opt.c_str());
			std::string callstr = _call.GetCallString();
			csharpBackendCaller->CallString = gcnew String(callstr.c_str());
			pythonCaller->RunCall(csharpBackendCaller);

			std::string result_string = context.marshal_as<std::string>(csharpBackendCaller->Result);
			_call.ParseResults(result_string);

		}
		catch (Exception ^ e)
		{
			System::Console::WriteLine(e->Message);
			String ^ error = System::String::Format("{{\"error\": {{\"code\": 5, \"message\": \"{0}\" }}}} ", e->Message);
			System::Console::WriteLine(error);
			std::string error_string = context.marshal_as<std::string>(error);
			_call.ParseResults(error_string);
		}
		
	}
}
