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

    # TODO: Add the rest of the functions from BackendWrapper - wrap each one with try/except
