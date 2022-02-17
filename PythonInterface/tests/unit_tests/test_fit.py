import json
import math
import os
import struct
import datetime
import time
import pytest
import sys
exe_directory=r"C:\Users\chana\Source\DPlus\dplus\x64\ReleaseWithDebugInfo"

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner, EmbeddedLocalRunner

import numpy as np

from dplus.FitRunner import FitRunner

def test_fit():
    input=CalculationInput.load_from_state_file(r'C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\sphere_fixed.state')
    # C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\uhc.state')
    runner = FitRunner()
    result = runner.fit(input)
    print(result)


def test_fit_async():
    input=CalculationInput.load_from_state_file(r'C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\sphere_fixed.state')
    # C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\uhc.state')
    runner = FitRunner()
    runner.fit_async(input)
    status = runner.get_status()
    while status.get('isRunning'):
        status = runner.get_status()
        print("status:", status)
        time.sleep(0.5)

    result = runner.get_result()
    print("done")
    # print(result.graph)


def test_stop():
    input=CalculationInput.load_from_state_file(r'C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\sphere_fixed.state')
    # C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\manual_tests\files\uhc.state')
    runner = FitRunner()
    runner.fit_async(input)
    status = runner.get_status()
    t0 = datetime.datetime.now()
    while status.get('isRunning', False):
        status = runner.get_status()
        print("status:", status)
        if datetime.datetime.now() - t0 > datetime.timedelta(seconds=1) :
            runner.stop()
            print("stop")
            # break
        time.sleep(0.3)

    result = runner.get_result()
    print("done")
    print(result.graph)


if __name__ == "__main__":
    test_fit()
    test_fit_async()
    test_stop()
	