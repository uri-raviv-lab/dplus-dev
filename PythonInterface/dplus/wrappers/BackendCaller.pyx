# This file calls the backend using the PythonBackendWrapper

from PythonBackendWrapper cimport PythonBackendWrapper

cdef class BackendWrapper:
    cdef PythonBackendWrapper _wrapper

    def check_capabilities(self, tdrLevel):
        self._wrapper.CheckCapabilities(tdrLevel)
