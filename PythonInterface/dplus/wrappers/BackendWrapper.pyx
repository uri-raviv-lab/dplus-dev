# This file calls the backend using the PythonBackendWrapper
from PythonBackendWrapper cimport PythonBackendWrapper

cdef class BackendWrapper:
    cdef PythonBackendWrapper _wrapper

    def check_capabilities(self, tdrLevel):
        self._wrapper.CheckCapabilities(tdrLevel)

    def get_all_model_metadata(self):
        return self._wrapper.GetAllModelMetadata()

    def initialize_cache(self, cache_dir):
        self._wrapper.InitializeCache(cache_dir)

    def start_generate(self, state, useGPU):
        self._wrapper.StartGenerate(state, useGPU)

    def get_job_status(self):
        return self._wrapper.GetJobStatus();

    def get_generate_results(self):
        return self._wrapper.GetGenerateResults();

    def save_amp(self, modelptr, path):
        self._wrapper.SaveAmplitude(modelptr, path)

    def get_pdb(self, modelptr): 
        bytes_pdb = self._wrapper.GetPDB(modelptr)
        str_pdb = bytes_pdb.decode('utf-8')
        return str_pdb

    def get_model_ptrs(self):
        return list(self._wrapper.GetModelPtrs())

