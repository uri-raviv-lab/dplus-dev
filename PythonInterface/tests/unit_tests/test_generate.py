import json
import math
import os
import struct
import datetime
import time
import pytest
import sys
exe_directory=r"C:\Users\chana\Source\DPlus\dplus\x64\ReleaseWithDebugInfo"
sys.path.append(r"C:\Users\chana\Source\DPlus\dplus\PythonInterface")
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner, EmbeddedLocalRunner
from dplus.FitRunner import FitRunner

import numpy as np

def _test_1():
    state_file = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\sphere.state"
    input = CalculationInput.load_from_state_file(state_file)
    runner = LocalRunner(exe_directory)
    result = runner.generate(input)
    print('result', result)
    if result.error["code"] != 0:
        print("Result returned error:", result.error)


def test_2():
    state_file_path = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\sphere.state"
    calc_input = CalculationInput.load_from_state_file(state_file_path)
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
    # pdb = runner.get_pdb(ptr)
    # print(pdb)


def test_3():
    state_file_path = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\sphere.state"
    calc_input = CalculationInput.load_from_state_file(state_file_path)
    # calc_input.use_gpu = False
    runner = EmbeddedLocalRunner()
    res = runner.generate(calc_input)
    print(res)

    
if __name__ == "__main__":
    # _test_1()
    # test_2()
	test_3()
