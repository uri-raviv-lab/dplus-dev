from libcpp cimport bool
from libcpp.string cimport string

# cdef extern from r"../../../Backend/Backend/backend_exception.h" :
cdef extern from r"Backend/Backend/backend_exception.h" :
    cdef cppclass backend_exception:
        backend_exception(int error_code, const char *error_message = "") except +
        int GetErrorCode()
        string GetErrorMessage()


# Decalre the class with cdef
# cdef extern from r"../../..Backend/Backend/Grid.h" :
cdef extern from r"Backend/Backend/CommandLineBackendWrapper.h" :
    cdef cppclass CommandLineBackendWrapper:
        CommandLineBackendWrapper()
        void CheckCapabilities(bool checkTdr) except +
