import json
from dplus.wrappers import BackendWrapper

class BackendError(RuntimeError):
    def __init__(self, re):
        re_dict = json.loads(str(re))
        self.error_code = re_dict['code']
        super().__init__(re_dict['message'])

class Backend:
    def __init__(self):
        self._wrapper = BackendWrapper()

    def check_capabilities(self, check_tdr=True):
        try:
            return self._wrapper.check_capabilities(check_tdr)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def get_all_model_metadata(self):
        try:
            metadata = self._wrapper.get_all_model_metadata()
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def initialize_cache(self, cache_dir):
        try:
            metadata = self._wrapper.initialize_cache(cache_dir)
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def start_generate(self, state, useGPU):
        try:
            metadata = self._wrapper.start_generate(state, useGPU)
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be
    
    def get_job_status(self):
        try:
            metadata = self._wrapper.get_job_status()
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def get_generate_results(self):
        try:
            metadata = self._wrapper.get_generate_results()
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def save_amp(self, modelptr, path):
        try:
            metadata = self._wrapper.save_amp(modelptr, path)
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def get_pdb(self, modelptr):
        try:
            metadata = self._wrapper.get_pdb(modelptr)
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be

    def get_model_ptrs(self):
        try:
            metadata = self._wrapper.get_model_ptrs()
            return json.loads(metadata)
        except RuntimeError as re:
            be = BackendError(re)
        raise be
