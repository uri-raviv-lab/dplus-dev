import json
import math

import numpy as np
from collections import OrderedDict


def _handle_infinity_for_json(obj):
    if isinstance(obj, float):
        if obj == math.inf:
            return ("inf")
        if obj == -math.inf:
            return ("-inf")
    elif isinstance(obj, dict):
        return dict((k, _handle_infinity_for_json(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return [_handle_infinity_for_json(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple([_handle_infinity_for_json(x) for x in obj])
    return obj


class NumpyHandlingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyHandlingEncoder, self).default(obj)
