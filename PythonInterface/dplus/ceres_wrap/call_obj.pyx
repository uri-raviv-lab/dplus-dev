from libc.stdio cimport printf #printf for debugging
from libcpp cimport bool
from libcpp.string cimport string

#although we dont use these imports, without them this file doesnt compile
import numpy as np
cimport numpy as np

#TODO: add credit
cdef public bool call_obj(obj, const double* x,  double * p, double* residual, int numResiduals, int numParams) :
    # for some unknown reason you can't use numpy here ( cdef double[:] view fail, np_x.data fail)
    # however we must convert c types to python types. so we will create python lists that contain the values
    cdef int i
    PyX = []
    for i in range(numResiduals):
        PyX.append(x[i])

    PyParams =[]
    for i in range(numParams):
        PyParams.append(p[i])

    PyResiduals = obj(PyParams, numResiduals)

    for i in range(numResiduals):
        residual[i] = PyResiduals[i]

    return True