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
        void InitializeCache(string cacheDir) except +

        string GetJobStatus() except +
        string GetGenerateResults() except +
        string StartGenerate(string state, bool useGPU) except +

        void SaveAmplitude(unsigned int ModelPtr, string path) except +
        vector[unsigned in] GetModelPtrs() except +