using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.IO;
using System.Text.RegularExpressions;
using System.Collections;

namespace PythonBackend
{
    public class PythonBackendCaller
    {
        /*
         * Add CSharpPython as an embedded resource
         * In PythonInstaller, add a /Src subfolder which can hold Python files and is in the path.
         * Store CSharpPythonEntry.py in the /Src subfolder
         * Get list of required wheels (pip install dplus-api in new environment, then pip freeze for list of installed packages)
         * Add all wheels as embedded resources.
         * 
         */
        IntPtr lockPythonPtr;
        string session, exeDir;
        dynamic cSharpPythonEntry;

        public PythonBackendCaller(string _exeDir)
        {
            string tmpPath = Path.GetTempPath();
            Guid g;
            // Create and display the value of two GUIDs.
            g = Guid.NewGuid();
            session = tmpPath + @"dplus\" + g;
            Directory.CreateDirectory(session);
            Console.WriteLine("session folder: " + session);
            exeDir = _exeDir;
            InitPython();
        }
        public PythonBackendCaller(string _exeDir, string _session)
        { 
            session = _session;
            Directory.CreateDirectory(session);
            Console.WriteLine("session folder: " + session);
            exeDir = _exeDir;
            InitPython();
        }
        public void InitPython()
        {
            Console.WriteLine("Initializing the embedded Python environment");
            PythonInstaller.ApplicationName = "DPlus";
            PythonInstaller.InitializePythonEnvironment();
            PythonEngine.BeginAllowThreads();

            using (Py.GIL())
            {
                dynamic csharpModule = Py.Import("CSharpPythonEntry");
                cSharpPythonEntry = csharpModule.CSharpPython(exeDir, session, PythonInstaller.ActualInstallationFolder);
            }

            Console.WriteLine("Python Environment Initialized");
        }

        public void RunCall(CSharpManagedBackendCall call)
        {
            lockPythonPtr = PythonEngine.AcquireLock();
            dynamic result = cSharpPythonEntry.perform_call(call.CallString);
            call.Result = result;
            PythonEngine.ReleaseLock(lockPythonPtr);
        }
    }
}
