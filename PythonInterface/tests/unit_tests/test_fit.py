import os
import datetime
import time

from dplus.CalculationInput import CalculationInput
from dplus.FitRunner import FitRunner

root_path = os.path.dirname(os.path.abspath(__file__))


def test_fit():
    input = CalculationInput.load_from_state_file(
        os.path.join(root_path, "files_for_tests", "sphere.state"))
    runner = FitRunner()
    result = runner.fit(input)
    # print(result)
    assert result


def test_fit_async():
    input = CalculationInput.load_from_state_file(
        os.path.join(root_path, "files_for_tests", "sphere.state"))
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
    assert result.graph


def test_stop():
    input = CalculationInput.load_from_state_file(
        os.path.join(root_path, "files_for_tests", "sphere.state"))
    runner = FitRunner()
    runner.fit_async(input)
    status = runner.get_status()
    t0 = datetime.datetime.now()
    while status.get('isRunning', False):
        status = runner.get_status()
        print("status:", status)
        if datetime.datetime.now() - t0 > datetime.timedelta(seconds=0.5):
            runner.stop()
            print("stop")
            # break
        time.sleep(0.1)

    status = runner.get_status()
    result = runner.get_result()
    print(result.graph)
    assert not len(result.graph)
