import json
from dplus.wrappers import BackendWrapper

class BackendError(RuntimeError):
    def __init__(self, re):
        re_dict = json.loads(str(re))
        self.error_code = re_dict['code']
        super().__init__(re_dict['message'])

def check_capabilities(check_tdr=True):
    wrapper = BackendWrapper()
    try:
        return wrapper.check_capabilities(check_tdr)
    except RuntimeError as re:
        be = BackendError(re)
    raise be
