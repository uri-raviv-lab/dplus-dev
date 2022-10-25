from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner, WebRunner
from dplus.DataModels import Constraints, Parameter
from dplus.State import State, DomainPreferences, FittingPreferences
from dplus.Amplitudes import Amplitude
from dplus.DataModels.models import UniformHollowCylinder
import numpy as np

import os
from os.path import abspath
import datetime
import tempfile
import shutil

from tests.old_stuff.fix_state_files import fix_file

root_path=os.path.dirname(abspath(__file__))


def test_overview_generate_empty_local_runner():
    from dplus.CalculationInput import CalculationInput
    from dplus.CalculationRunner import EmbeddedLocalRunner

    calc_data = CalculationInput.load_from_state_file(os.path.join(root_path, "files", "mystate.state"))
    runner = EmbeddedLocalRunner()
    result = runner.generate(calc_data)
    assert len(result.graph) > 0

def test_calculation_runner_2_params_local_runner():
    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file)
    runner = EmbeddedLocalRunner()
    result = runner.generate(calc_data)
    assert len(result.graph)>0

def test_calculation_runner_exe_param_local_runner():
    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file)
    runner = EmbeddedLocalRunner()
    result = runner.generate(calc_data)
    assert len(result.graph)>0

def test_calculation_runner_sess_param_local_runner():
    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file)
    runner = EmbeddedLocalRunner()
    result = runner.generate(calc_data)
    assert len(result.graph)>0

def x_test_calculation_runner_web_runner():
    # TO DO - check that the server and token are correct???
    url = r'http:// localhost :8000/'
    token = '4bb25edc45acd905775443f44eae'
    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file)
    runner = WebRunner(url, token)
    result = runner.generate(calc_data)
    assert len(result.graph)>0

def test_running_generate_async():

    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file)
    runner = EmbeddedLocalRunner()
    runner.generate_async(calc_data)
    start_time = datetime.datetime.now()
    status=True
    while status:
        try:
            status_dict = runner.get_job_status()
            status=status_dict['isRunning']
        except:
            status=True
        run_time = datetime.datetime.now() - start_time
        if run_time > datetime.timedelta(seconds=50):
            runner.stop_generate()
            raise TimeoutError("Job took too long")
    result = runner.get_generate_results(calc_data)
    assert len(result.graph) > 0


def test_calculation_input():
    #TODO: can add asserts about vector lengths
    from dplus.CalculationInput import CalculationInput
    gen_input2 = CalculationInput()
    fixed_state_file = fix_file(os.path.join(root_path, "files", 'sphere.state'))
    gen_input = CalculationInput.load_from_state_file(fixed_state_file)


def test_datamodels_generate_calculate_input_new_state():
    uhc = UniformHollowCylinder()
    gen_input = CalculationInput()
    gen_input.Domain.populations[0].add_model(uhc)
    gen_input.DomainPreferences.q_max = 10

    runner = EmbeddedLocalRunner()
    result = runner.generate(gen_input)
    assert len(result.graph) > 0

def test_datamodels_save_state():
    uhc = UniformHollowCylinder()
    gen_input = CalculationInput()
    gen_input.Domain.populations[0].add_model(uhc)
    dompref = DomainPreferences()
    dompref.q_max = 10
    gen_input.DomainPreferences = dompref

    tmp_directory = tempfile.mkdtemp()
    new_file_path  = os.path.join(tmp_directory, 'test.state')
    gen_input.export_all_parameters(new_file_path)

    calc_input=CalculationInput.load_from_state_file(new_file_path)

    shutil.rmtree(tmp_directory)



def test_datamodels_fit_calculate_input_state():
    # TODO - fit files???
    state_file = os.path.join(root_path, "files", "sphere.state")
    fixed_state_file = fix_file(state_file)
    fit_input = CalculationInput.load_from_state_file(fixed_state_file)
    fit_input.FittingPreferences.convergence=0.5
    runner = EmbeddedLocalRunner()
    result = runner.fit(fit_input)
    assert len(result.graph) > 0

def test_datamodels_generate_model():
    runner = EmbeddedLocalRunner()
    uhc = UniformHollowCylinder()
    caldata = CalculationInput()
    caldata.Domain.populations[0].add_model(uhc)
    result = runner.generate(caldata)
    assert len(result.graph) > 0

def test_constraints_and_parameters():
    c = Constraints(min_val=5)
    p = Parameter(4)


#the extra examples at the end of the manual:

def test_example_one_sphere_fit():
    from dplus.CalculationInput import CalculationInput
    from dplus.CalculationRunner import EmbeddedLocalRunner

    runner = EmbeddedLocalRunner()
    state_file = os.path.join(root_path, "files", "sphere.state")
    fixed_state_file = fix_file(state_file)
    input = CalculationInput.load_from_state_file(fixed_state_file)
    result = runner.fit(input)
    assert result

def test_example_two_generate_model():
    runner = EmbeddedLocalRunner()
    uhc = UniformHollowCylinder()
    uhc.layer_params[1]["radius"].value = 2.0
    uhc.extra_params.height.value = 3.0
    uhc.location_params.x.value = 2

    caldata = CalculationInput()
    caldata.Domain.populations[0].add_model(uhc)
    result = runner.generate(caldata)
    assert len(result.graph) > 0

def test_example_three_generate_pdb():
    pdb_file = os.path.join(root_path, "files", "1JFF.pdb")
    caldata = CalculationInput.load_from_PDB(pdb_file, 5)
    runner = EmbeddedLocalRunner()
    result = runner.generate(caldata)
    assert len(result.graph) > 0

def test_example_three_generate_epdb():
    pdb_file = os.path.join(root_path, "files", "1JFF.pdb")
    caldata = CalculationInput.load_from_EPDB(pdb_file, 5)
    caldata.use_gpu = False
    runner = EmbeddedLocalRunner()
    result = runner.generate(caldata)

    out_dir = os.path.join(root_path, 'out_dir')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, 'epdb.out')
    result.save_to_out_file(out_file)
    
    assert len(result.graph) > 0

def test_example_three_generate_epdb_from_state():
    state_file = os.path.join(root_path, "files", "epdb.state")
    fixed_state_file = fix_file(state_file)
    caldata = CalculationInput.load_from_state_file(fixed_state_file)
    caldata.use_gpu = False
    runner = EmbeddedLocalRunner()
    result = runner.generate(caldata)
    
    out_dir = os.path.join(root_path, 'out_dir')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, 'epdb_from_state.out')
    result.save_to_out_file(out_file)

    assert len(result.graph) > 0

def test_example_four_fit_UniformHollowCylinder():
    API = EmbeddedLocalRunner()
    state_file = os.path.join(root_path, "files", "uhc.state")
    input = CalculationInput.load_from_state_file(state_file)
    cylinder = input.get_model("test_cylinder")
    result = API.generate(input)
    input.signal = result.signal
    cylinder = input.get_model("test_cylinder")
    cylinder.layer_params[1].radius.value = 2
    cylinder.layer_params[1].radius.mutable = True
    input.FittingPreferences.convergence = 0.5
    input.use_gpu = True
    fit_result = API.fit(input)
    assert len(fit_result.graph) > 0

def test_EPDB():
    from dplus.DataModels.models import EPDB

    O_2 = EPDB(os.path.join(root_path, "files", "O_2.pdb"))
    O_2.extra_params.solvent_method.value = 0
    O = EPDB(os.path.join(root_path, "files", "O.pdb"))
    O.extra_params.solvent_method.value = 0
    O_2_state = CalculationInput()
    O_2_state.Domain.populations[0].add_model(O_2)
    O_state = CalculationInput()
    O_state.Domain.populations[0].add_model(O)

    runner = EmbeddedLocalRunner()
    O_2_result = runner.generate(O_2_state)
    O_result = runner.generate(O_state)

    assert not np.isclose(O_result.graph[0.0], O_2_result.graph[0.0])

def test_nonexist_PDB():
    from dplus.DataModels.models import PDB

    O_2 = PDB(os.path.join(root_path, "files", "O_2.pdb"))
    O_2.extra_params.solvent_method.value = 0
    O_2_state = CalculationInput()
    O_2_state.Domain.populations[0].add_model(O_2)

    runner = EmbeddedLocalRunner()
    O_2_result = runner.generate(O_2_state)

    assert O_2_result.graph[0.0] == 0

def test_EPDB_Pu():
    from dplus.DataModels.models import EPDB

    Pu_p3 = EPDB(os.path.join(root_path, "files", "Pu_p3.pdb"))
    Pu_p3.extra_params.solvent_method.value = 0
    Pu_p3_state = CalculationInput()
    Pu_p3_state.Domain.populations[0].add_model(Pu_p3)
    runner = EmbeddedLocalRunner()
    Pu_p3_result = runner.generate(Pu_p3_state)

    assert Pu_p3_result.graph[0.0] == 0

def test_PDB_vs_EPDB():
    from dplus.DataModels.models import EPDB, PDB

    epdb = EPDB(os.path.join(root_path, "files", "O.pdb"))
    pdb = PDB(os.path.join(root_path, "files", "O.pdb"))
    epdb.extra_params.solvent_method.value = 0
    pdb.extra_params.solvent_method.value = 0
    epdb_state = CalculationInput()
    pdb_state = CalculationInput()
    epdb_state.Domain.populations[0].add_model(epdb)
    pdb_state.Domain.populations[0].add_model(pdb)
    runner = EmbeddedLocalRunner()
    epdb_result = runner.generate(epdb_state)
    pdb_result = runner.generate(pdb_state)

    assert not np.isclose(epdb_result.y[0], pdb_result.y[0])

def test_EPDB_in_sym():
    from dplus.DataModels.models import EPDB
    from dplus.DataModels.models import SpacefillingSymmetry

    my_epdb = EPDB(os.path.join(root_path, "files", "1jff.pdb"))
    my_epdb.extra_params.solvent_method.value = 0

    my_sym = SpacefillingSymmetry()
    my_sym.layer_params[0].distance.value = 1.1
    my_sym.layer_params[1].distance.value = 1.2
    my_sym.layer_params[2].distance.value = 1.3
    my_sym.use_grid = False
    my_sym.children.append(my_epdb)

    my_state = CalculationInput()
    my_state.Domain.populations[0].add_model(my_sym)
    my_state.DomainPreferences.signal_file = os.path.join(root_path, "files", "1jff.out")
    my_state.DomainPreferences.grid_size = 250
    my_state.DomainPreferences.orientation_iterations = 1e6
    # my_state.export_all_parameters(os.path.join(root_path, "files", "1jff.state"))

    runner = EmbeddedLocalRunner()
    result = runner.generate(my_state)
    # result.save_to_out_file(os.path.join(root_path, "files", "found.out"))

    assert np.max(np.abs((np.array(my_state.y) - np.array(result.y))) / np.array(my_state.y)) < 0.10


#test_example_three_generate_epdb_from_state()