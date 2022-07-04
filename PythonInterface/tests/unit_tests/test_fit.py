import os, sys
import datetime
import time

sys.path.append(os.getcwd())
from tests.old_stuff.fix_state_files import fix_file


from dplus.CalculationInput import CalculationInput
from dplus.FitRunner import FitRunner
# from dplus.PyCeresOptimizer import PyCeresOptimizer

tests_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.join(tests_folder, "unit_tests")


def test_fit():
    input = CalculationInput.load_from_state_file(
        os.path.join(tests_folder, "reviewer_tests", "files_for_tests", "fit", "gpu", "short", "Sphere_Radius_Fit_low", "Sphere_Radius_Fit_low_fixed.state")
    )
    runner = FitRunner()
    result = runner.fit(input)
    print("result")
    print(result.graph)
    assert result


def test_fit_async():
    input = CalculationInput.load_from_state_file(
        os.path.join(tests_folder, "reviewer_tests", "files_for_tests", "fit", "gpu", "short", "Sphere_Radius_Fit_low", "Sphere_Radius_Fit_low_fixed.state")
    )
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
        os.path.join(tests_folder, "reviewer_tests", "files_for_tests", "fit", "gpu", "short", "Sphere_Radius_Fit_low", "Sphere_Radius_Fit_low_fixed.state")
    )
    runner = FitRunner()
    runner.fit_async(input)
    status = runner.get_status()
    runner.stop()
    status = runner.get_status()
    assert status == {"error": {"code": 22, "message": "job stop run"}}

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

    runner = FitRunner()
    result = runner.fit(calc_input)
    
    assert abs(calc_input.get_mutable_parameter_values()[0] - 4) / 4 < 0.02 and abs(
        calc_input.get_mutable_parameter_values()[1] - 400) / 400 < 0.02

def test_example_six_sphere_cylinder():
    out_file = os.path.join(root_path, "files_for_tests", 'Cyl_Sph_End.out')
    state_file= os.path.join(root_path, "files_for_tests", 'Cyl_Sph_Start.state')
    calc_input = CalculationInput.load_from_state_file(state_file)
    calc_input.DomainPreferences.signal_file = out_file
    # calc_input.use_gpu = False

    runner = FitRunner()
    result = runner.fit(calc_input)

    assert abs(calc_input.get_mutable_parameter_values()[0] - 5) / 5 < 0.02 and abs(
        calc_input.get_mutable_parameter_values()[1] - 400) / 400 < 0.02

def test_example_seven_PDB():
    out_file = os.path.join(root_path, "files_for_tests", '1jff_ED_334_probe_0.14_voxel.out')
    calc_input = CalculationInput.load_from_state_file(os.path.join(root_path, "files_for_tests", '1jff_ED_350_probe_0.125_voxel.state'))
    calc_input.Domain.populations[0].models[0].filename = os.path.join(root_path, "files_for_tests", '1jff.pdb')
    calc_input.DomainPreferences.signal_file = out_file
    # calc_input.use_gpu = False
    print("test_example_seven_PDB")
    try:
        runner = FitRunner()
        result = runner.fit(calc_input)
    except Exception as ex:
        print(ex)

    print(calc_input.get_mutable_parameter_values())
    print("0:", abs(calc_input.get_mutable_parameter_values()[0] - 334) / 334)
    print("1:", abs(calc_input.get_mutable_parameter_values()[1] - 0.14) / 0.14)
    assert abs(calc_input.get_mutable_parameter_values()[0] - 334) / 334 < 0.02 and abs(
        calc_input.get_mutable_parameter_values()[1] - 0.14) / 0.14 < 0.025


if __name__ == '__main__':
    # test_fit_async()
    # test_stop()
    # test_example_six_sphere_cylinder()
    test_example_seven_PDB()