# This file calls the backend using the CommandLineBackendWrapper

from CommandLineBackendWrapper cimport CommandLineBackendWrapper

cdef class BackendWrapper:
    cdef CommandLineBackendWrapper _wrapper

    def __cinit__(self):
        self._wrapper = CommandLineBackendWrapper()

    def check_capabilities(self, tdrLevel):
        self._wrapper.CheckCapabilities(tdrLevel)
