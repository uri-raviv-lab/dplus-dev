from libcpp.string cimport string
from libcpp.complex cimport complex
from libcpp.memory cimport shared_ptr


cdef extern from r"../../build/include/Backend/Backend/backend_exception.cpp":
    pass


# cdef extern from r"../../../Backend/Backend/backend_exception.h" :
cdef extern from r"../../build/include/Backend/Backend/backend_exception.h" :
    cdef cppclass backend_exception:
        backend_exception(int error_code, const char *error_message = "") except +
        int GetErrorCode()
        string GetErrorMessage()


# cdef extern from r"../../../Backend/Backend/Grid.cpp":
cdef extern from r"../../build/include/Backend/Backend/Grid.cpp":
    pass

# Decalre the class with cdef
# cdef extern from r"../../..Backend/Backend/Grid.h" :
cdef extern from r"../../build/include/Backend/Backend/Grid.h" :
    cdef cppclass JacobianSphereGrid:
        JacobianSphereGrid()except +
        JacobianSphereGrid(unsigned short gridSize, double qMax)except +
        long long IndexFromIndices( int qi, long long ti, long long pi ) except +
        void IndicesFromIndex( long long index, int &qi, long long &ti, long long &pi ) except +
        void CalculateSplines() except +
        unsigned long long GetRealSize() except +
        double* GetDataPointer() except +
        double* GetInterpolantPointer()except +
        string GetParamJsonString()except +
        double GetQMax() except +
        unsigned short GetSize() except +
        unsigned short GetActualSize() except +
        unsigned short GetExtras() except +
        complex[double] InterpolateThetaPhiPlane( const unsigned short ri, const double theta, const double phi ) except +
        char thetaDivisions
        char phiDivisions
        double stepSize
        unsigned short GetDimX() except +
        unsigned short GetDimY(unsigned short x) except +
        unsigned short GetDimZ(unsigned short x, unsigned short z) except +
        complex[double] GetSphr( double rr, double th, double ph ) except +
        double CalculateIntensity(double q, double epsi, unsigned int seed, unsigned long long iterations) except +
