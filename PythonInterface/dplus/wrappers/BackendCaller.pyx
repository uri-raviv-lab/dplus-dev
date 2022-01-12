# This file calls the backend using the PythonBackendWrapper

from PythonBackendWrapper cimport PythonBackendWrapper

cdef class BackendWrapper:
    cdef PythonBackendWrapper _wrapper

    def check_capabilities(self, tdrLevel):
        self._wrapper.CheckCapabilities(tdrLevel)

    def get_all_model_metadata(self):
        return self._wrapper.GetAllModelMetadata()
