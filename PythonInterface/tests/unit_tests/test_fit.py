import os
import datetime
import time

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner
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
        os.path.join(root_path, "files_for_tests", "ampcontaining.state"))
    runner = FitRunner()
    runner.fit_async(input)
    status = runner.get_status()
    t0 = datetime.datetime.now()
    stopped = False
    while status.get('isRunning', False):
        if datetime.datetime.now() - t0 > datetime.timedelta(seconds=0.01):
            runner.stop()
            stopped = True
            break
        time.sleep(0.1)
        status = runner.get_status()
        print("status:", status)

    status = runner.get_status()
    print("stopped:", stopped)
    if stopped:
        print(status)
        assert status == {"isRunning": False, "progress": 0.0, "code": -1, "message": ""}
    else:
        print(status)
        assert status == {"isRunning": False, "progress": 100.0, "code": 0, "message": ""}

def test_example_five_sphere():
    from dplus.DataModels.models import Sphere
    out_file = os.path.join(root_path, "files_for_tests", 'Sph_r4_ed400.out')

    calc_input = CalculationInput()
    # calc_input.use_gpu = False
    calc_input.DomainPreferences.signal_file = out_file
    calc_input.FittingPreferences.fitting_iterations = 6
    calc_input.FittingPreferences.convergence = 1e-8
    sp = Sphere()
    sp.layer_params[1].radius.value = 4.4
    sp.layer_params[1].radius.mutable = True
    sp.layer_params[1].ed.value = 440
    sp.layer_params[1].ed.mutable = True
    calc_input.Domain.populations[0].add_model(sp)

    runner = EmbeddedLocalRunner()
    runner.fit(calc_input)
    assert abs(calc_input.get_mutable_parameter_values()[0] - 4) / 4 < 0.02 and abs(
        calc_input.get_mutable_parameter_values()[1] - 400) / 400 < 0.02

def test_example_six_sphere_cylinder():
    out_file = os.path.join(root_path, "files_for_tests", 'Cyl_Sph_End.out')

    calc_input = CalculationInput.load_from_state_file(os.path.join(root_path, "files_for_tests", 'Cyl_Sph_Start.state'))
    calc_input.DomainPreferences.signal_file = out_file
    # calc_input.use_gpu = False

    runner = EmbeddedLocalRunner()
    runner.fit(calc_input)

    assert abs(calc_input.get_mutable_parameter_values()[0] - 5) / 5 < 0.02 and abs(
        calc_input.get_mutable_parameter_values()[1] - 400) / 400 < 0.02

def test_example_seven_PDB():
    out_file = os.path.join(root_path, "files_for_tests", '1jff_ED_334_probe_0.14_voxel.out')
    calc_input = CalculationInput.load_from_state_file(os.path.join(root_path, "files_for_tests", '1jff_ED_350_probe_0.125_voxel.state'))
    calc_input.DomainPreferences.signal_file = out_file
    # calc_input.use_gpu = False

    runner = EmbeddedLocalRunner()
    try:
        runner.fit(calc_input)
    except Exception as e:
        print(e)
    print(calc_input.get_mutable_parameter_values())
    assert abs(calc_input.get_mutable_parameter_values()[0] - 334) / 334 < 0.01 and abs(
        calc_input.get_mutable_parameter_values()[1] - 0.14) / 0.14 < 0.025
