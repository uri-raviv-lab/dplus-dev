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

/**
 * @file PythonBackendCaller.cs
 * @brief Implements the PythonBackendCaller class, which bridges .NET and Python for backend computation in DPlus.
 *
 * The PythonBackendCaller module is responsible for:
 *  - Initializing and managing an embedded Python environment for backend operations.
 *  - Creating and managing a session directory for Python execution context.
 *  - Loading and interfacing with the main Python entry point (CSharpPythonEntry.py) for backend calls.
 *  - Executing backend calls by forwarding serialized call strings to Python and retrieving results.
 *  - Managing thread safety and the Python Global Interpreter Lock (GIL) during backend calls.
 *
 * Key Concepts:
 *  - Embedded Python: Uses Python.NET to host a Python interpreter within the .NET process.
 *  - Session Management: Each instance can use a unique session directory for isolation.
 *  - CSharpPythonEntry: Python module that acts as the entry point for backend operations.
 *  - CSharpManagedBackendCall: Encapsulates a backend call, including arguments and result.
 *  - Thread Safety: Acquires and releases the Python GIL for each backend call to ensure safe multi-threaded operation.
 *
 * Fields:
 *  - IntPtr lockPythonPtr:
 *      Pointer to the Python GIL state for thread safety.
 *  - string session:
 *      Path to the session directory for the current backend caller instance.
 *  - string exeDir:
 *      Path to the executable directory, used for locating resources.
 *  - dynamic cSharpPythonEntry:
 *      Reference to the loaded Python entry point object for backend calls.
 *
 * Main Methods:
 *  - PythonBackendCaller(string _exeDir):
 *      Constructor; creates a new session directory and initializes the Python environment.
 *  - PythonBackendCaller(string _exeDir, string _session):
 *      Constructor; uses an existing session directory and initializes the Python environment.
 *  - void InitPython():
 *      Initializes the embedded Python environment and loads the entry point.
 *  - void RunCall(CSharpManagedBackendCall call):
 *      Executes a backend call by passing the call string to Python and retrieving the result.
 *
 * Threading and Safety:
 *  - Acquires and releases the Python GIL for each backend call to ensure thread safety.
 *  - Designed for use in multi-threaded .NET applications.
 *
 * Error Handling:
 *  - Prints status and error messages to the console for initialization and session management.
 *  - Assumes that exceptions in Python or .NET will be handled by the caller.
 *
 * Dependencies:
 *  - Python.Runtime (Python.NET) for embedding Python.
 *  - Newtonsoft.Json for JSON serialization.
 *  - CSharpManagedBackendCall for backend call encapsulation.
 *  - CSharpPythonEntry.py (Python) as the backend entry point.
 *  - System.IO for file and directory management.
 *
 * See CSharpManagedBackendCall.cs and CSharpPythonEntry.py for related implementation details.
 */

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
                cSharpPythonEntry = csharpModule.get_csharp_python_entry(exeDir, session);
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
