import os
import numpy as np
import time

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner

from tests.test_settings import USE_GPU

root_path = os.path.dirname(os.path.abspath(__file__))

def test_generate_1():
    state_file_path = os.path.join(root_path, "files_for_tests", "sphere.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU)
    runner = EmbeddedLocalRunner()
    runner.generate_async(calc_input)

    status = runner.get_job_status()

    while status and status['isRunning'] and status['code']==-1:
        status = runner.get_job_status()
        time.sleep(0.1)
    print("end", status)
    if status['code'] == 0:
        result = runner.get_generate_results(calc_input)
        print(result.processed_result)
        if result.error["code"] != 0:
            print("Result returned error:", result.error)
    else:
        print("error", status)

    model_ptrs = runner.get_model_ptrs()
    print('model_ptrs', model_ptrs)
    for ptr in model_ptrs:
        print('ptr:', ptr)
    ptr = model_ptrs[-1]
    runner.save_amp(ptr, "amp-{}.ampj".format(ptr))
    print("the Amp was saved")


def test_generate_2():
    state_file_path = os.path.join(root_path, "files_for_tests", "sphere.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU)
    runner = EmbeddedLocalRunner()
    res = runner.generate(calc_input)
    print(res)

