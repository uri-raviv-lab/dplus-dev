from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

# Decalre the class with cdef
# cdef extern from r"../../..Backend/Backend/Grid.h" :
cdef extern from r"Backend/Backend/PythonBackendWrapper.h" :
    cdef cppclass PythonBackendWrapper:
        PythonBackendWrapper()
        void CheckCapabilities(bool checkTdr) except +
        string GetAllModelMetadata() except +
        void InitializeCache() except +
        void InitializeCache(string cacheDir) except +

        string GetJobStatus() except +
        string GetGenerateResults() except +
        void StartGenerate(string state, bool useGPU) except +

        void SaveAmplitude(unsigned int modelPtr, string path) except +
        string GetPDB(unsigned int modelPtr) except +
        vector[unsigned int] GetModelPtrs() except +
        void Stop() except +