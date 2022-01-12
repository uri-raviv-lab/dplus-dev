from libcpp cimport bool
from libcpp.string cimport string

# Decalre the class with cdef
# cdef extern from r"../../..Backend/Backend/Grid.h" :
cdef extern from r"Backend/Backend/PythonBackendWrapper.h" :
    cdef cppclass PythonBackendWrapper:
        PythonBackendWrapper()
        void CheckCapabilities(bool checkTdr) except +
        string GetAllModelMetadata() except +
