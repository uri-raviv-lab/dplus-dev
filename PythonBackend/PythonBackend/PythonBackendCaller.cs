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
        IntPtr lockPythonPtr;
        string pythonPath;
        string session;
        string exeDir;
        string csharpPythonPath;
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
            pythonPath = "";
            //check what is the location of python
            if (exeDir.Contains("D+")) 
            {
                //program
                pythonPath = exeDir + @"\Python35";
                csharpPythonPath = exeDir + @"\Resources\CSharpPython";
            }
            else
            {
                //debug or release
                string [] dirsArray = exeDir.Split('\\');
                int lastIndex = exeDir.LastIndexOf("\\");
                string parent_path = Directory.GetParent(exeDir.Substring(0, lastIndex)).ToString();

                pythonPath = parent_path + @"\Python35";
                csharpPythonPath = parent_path + @"\PythonBackend\PythonBackend\CSharpPython";
            }
            
            Environment.SetEnvironmentVariable("PATH", $@"{pythonPath};" + Environment.GetEnvironmentVariable("PATH"));
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
            }

            UpdatePath();

            lockPythonPtr = PythonEngine.AcquireLock();
            dynamic csharpModule = Py.Import("CSharpPythonEntry");
            cSharpPythonEntry = csharpModule.CSharpPython(exeDir, session, pythonPath);
            PythonEngine.ReleaseLock(lockPythonPtr);
        }

        public void UpdatePath()
        {
            lockPythonPtr = PythonEngine.AcquireLock();

            dynamic sys = Py.Import("sys");
            dynamic sys_path = sys.path;
            int length = sys_path.__len__();
            ArrayList path2remove = new ArrayList();
            for (int i = 0; i < length; i++)
            {
                string cur_path = sys_path[i];
                if (cur_path.Contains("C:\\Python35"))
                {
                    path2remove.Add(cur_path);
                }
            }
            sys.path.append($@"{pythonPath}\Lib");
            sys.path.append(pythonPath);
            sys.path.append(csharpPythonPath);
            // remove C:\\Python35 from the path
            foreach (string item_path in path2remove)
            {
                sys.path.remove(item_path);
            }

            PythonEngine.ReleaseLock(lockPythonPtr);
        }

        ~PythonBackendCaller()
        {
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
