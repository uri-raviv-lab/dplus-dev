#include "ManagePythonPreCaller.h"
#include "../../Frontend/Frontend/BackendCalls.h"
#include <msclr\marshal_cppstd.h>

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
