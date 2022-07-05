using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using Python.Runtime;

// This class installs Python and sets up PythonEngine so that it uses it.
// It is heavily influenced by Python.Included

namespace PythonBackend
{
    static class PythonInstaller
    {
        const string PYTHON_VERSION = "3.9.9"; // Make sure this is the same python version as the embedded python zip file

        private static bool IsPythonInstalled() => File.Exists(PythonPath);

        public static string ApplicationName { get; set; } = null;
        public static string InstallationFolder { get; set; } = null;

        public static string ActualInstallationFolder => InstallationFolder ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), ApplicationName, $"python-{PYTHON_VERSION}");
        private static string PythonPath => Path.Combine(ActualInstallationFolder, "python.exe");
        private static string LibFolder => Path.Combine(ActualInstallationFolder, "Lib");
        private static string SrcFolder => Path.Combine(ActualInstallationFolder, "Src");

        public static void InitializePythonEnvironment()
        {
            if (ApplicationName == null && InstallationFolder == null)
            {
                throw new InvalidOperationException("You must set PythonInstaller.ApplicationName or PyInstaller.InstallationFolder before continuing");
            }

            if (IsPythonInstalled())
                Directory.Delete(ActualInstallationFolder, true);

            Directory.CreateDirectory(ActualInstallationFolder);
            using (var embeddedZip = GetEmbeddedResourceStream(Assembly.GetExecutingAssembly(), $"python-{PYTHON_VERSION}-embed-amd64.zip"))
            {
                UnzipStream(embeddedZip, ActualInstallationFolder);
            }
            Directory.CreateDirectory(LibFolder);
            Directory.CreateDirectory(SrcFolder);

            AddExtraFoldersToPath();
            InstallEmbeddedWheels();
            InstallEmbeddedSources();
            InitializeEngine();
        }

        public static void InstallEmbeddedWheels(Assembly assembly = null)
        {
            if (assembly == null)
            {
                assembly = Assembly.GetExecutingAssembly();
            }

            var wheelNames = assembly.GetManifestResourceNames().Where(x => x.EndsWith(".whl"));
            foreach (var wheelName in wheelNames)
            {
                // We need to get the proper dplus_api wheel - debug for debug, release for release
                if (wheelName.Contains("dplus_api"))
                {
#if DEBUGWITHRELEASE
                    // Debug mode
                    if (!wheelName.Contains("debug"))
                        continue;
#else
                    // Release mode
                    if (!wheelName.contains("release"))
                        continue;
#endif
                }

                var wheelStream = GetEmbeddedResourceStream(assembly, wheelName);
                UnzipStream(wheelStream, LibFolder);
            }
        }

        public static void InstallEmbeddedSources(string sourcePrefix = "PythonSources", Assembly assembly = null)
        {
            // Install all Sources in embedded resources, that have "PythonSources" in the resource name, and end with *.py, in the /Src directory
            // so they are importable from Python
            if (assembly == null)
            {
                assembly = Assembly.GetExecutingAssembly();
            }


            if (!sourcePrefix.EndsWith("."))
            {
                sourcePrefix += ".";
            }

            var sourceNames = assembly.GetManifestResourceNames().Where(x => x.Contains(sourcePrefix) && x.EndsWith(".py"));
            foreach (var sourceName in sourceNames)
            {
                // The resource stream is stored in afile whose name is taken after the sourcePrefix in the resource name
                // So if the prefix is PythonSources, and the resource is Resources.PythonSources.package.file.py, the
                // file is going to be SrcFolder/package/file.py
                //
                // We're using some heuristics for this - after the prefix, we break the resource name into parts on '.', combine the last two
                // (because the last . is part of the filename), and combine the rest into a path.
                //
                // This will not work if a resource folder has a . in its name - it will turn into a /.
                var prefixIndex = sourceName.LastIndexOf(sourcePrefix);
                var afterPrefix = sourceName.Substring(prefixIndex + sourcePrefix.Length);
                var parts = new List<string>(afterPrefix.Split('.'));  // Last part is 'py'
                parts[parts.Count - 2] += ".py";
                parts.RemoveAt(parts.Count - 1);

                var targetFilename = Path.Combine(SrcFolder, Path.Combine(parts.ToArray()));

                Stream sourceStream = assembly.GetManifestResourceStream(sourceName);

                Directory.CreateDirectory(Path.GetDirectoryName(targetFilename));
                using (var fileStream = File.Create(targetFilename))
                {
                    sourceStream.CopyTo(fileStream);
                }
            }
        }

        public static void InitializeEngine()
        {
            Environment.SetEnvironmentVariable("PATH", ActualInstallationFolder + ";" + Environment.GetEnvironmentVariable("PATH"));
            PythonEngine.Initialize();
            AddToSysPath(LibFolder);
            AddToSysPath(SrcFolder);
        }

        public static void AddToSysPath(string folder)
        {
            var setpath = $@"
import sys
if not r'{folder}' in sys.path:
    sys.path.append(r'{folder}')
";
            using (Py.GIL())
            {
                PythonEngine.Exec(setpath);
            }
        }

        private static void AddExtraFoldersToPath()
        {
            // Add LibFolder and SrcFolder to the embedded Python path, which is in the python<ver>._pth file in the installation folder
            // It's not clear why this is done - this does not seem to make any difference, and the same folders have to be added to sys.path.
            // However, Python.Included does this, so we do, too.
            var candidates = Directory.GetFiles(ActualInstallationFolder, "python*._pth");
            if (candidates.Length != 1)
            {
                throw new InvalidOperationException($"Expected just one python*._pth files in {ActualInstallationFolder}, instead found {candidates.Length}");
            }

            var allLines = File.ReadAllLines(candidates[0]);
            if (!allLines.Contains("./Lib"))
            {
                File.AppendAllLines(candidates[0], new[] { "./Lib" });
            }
            if (!allLines.Contains("./Src"))
            {
                File.AppendAllLines(candidates[0], new[] { "./Src" });
            }
        }

        private static Stream GetEmbeddedResourceStream(Assembly assembly, string resourceName)
        {
            var key = assembly.GetManifestResourceNames().FirstOrDefault(x => x.Contains(resourceName));
            if (key == null)
                throw new ArgumentException($"Error: Resource name '{resourceName}' not found in assembly {assembly.FullName}!");
            return assembly.GetManifestResourceStream(key);
        }

        private static void UnzipStream(Stream stream, string dest)
        {
            var archive = new ZipArchive(stream);
            foreach (var entry in archive.Entries)
            {
                var filename = Path.Combine(dest, entry.FullName);
                var dirname = Path.GetDirectoryName(filename);

                Directory.CreateDirectory(dirname);
                using (var fileStream = File.Create(filename))
                using (var zipStream = entry.Open())
                {
                    zipStream.CopyTo(fileStream);
                }
            }
        }
    }
}
