from libcpp cimport bool

# Decalre the class with cdef
# cdef extern from r"../../..Backend/Backend/Grid.h" :
cdef extern from r"Backend/Backend/PythonBackendWrapper.h" :
    cdef cppclass PythonBackendWrapper:
        PythonBackendWrapper()
        void CheckCapabilities(bool checkTdr) except +
