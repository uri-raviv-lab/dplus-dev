from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner, WebRunner, LocalRunner
from dplus.DataModels import Constraints, Parameter
from dplus.State import State, DomainPreferences, FittingPreferences
from dplus.Amplitudes import Amplitude
from dplus.DataModels.models import UniformHollowCylinder
from tests.test_settings import USE_GPU, session
import pytest

import os
from os.path import abspath
import datetime
import tempfile
import shutil
import numpy as np

from tests.old_stuff.fix_state_files import fix_file

root_path=os.path.dirname(abspath(__file__))


def test_overview_generate_simple_embedded_local_runner():
    from dplus.CalculationInput import CalculationInput
    from dplus.CalculationRunner import EmbeddedLocalRunner

    calc_data = CalculationInput.load_from_state_file(os.path.join(root_path, "files", "mystate.state"), USE_GPU)
    runner = EmbeddedLocalRunner()
    result = runner.generate(calc_data)
    assert len(result.graph) > 0

def x_test_calculation_runner_web_runner():
    # TO DO - check that the server and token are correct???
    url = r'http:// localhost :8000/'
    token = '4bb25edc45acd905775443f44eae'
    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file, USE_GPU)
    runner = WebRunner(url, token)
    result = runner.generate(calc_data)
    assert len(result.graph)>0

def test_running_generate_async():

    state_file=os.path.join(root_path, "files", "mystate.state")
    calc_data = CalculationInput.load_from_state_file(state_file, USE_GPU)
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

# def test_running_generate_async_localrunner():

#     state_file=os.path.join(root_path, "files", "mystate.state")
#     calc_data = CalculationInput.load_from_state_file(state_file, USE_GPU)
#     runner = LocalRunner(exe_directory)
#     job = runner.generate_async(calc_data)
#     start_time = datetime.datetime.now()
#     status = job.get_status()
#     while status['isRunning']:
#         status = job.get_status()
#         run_time = datetime.datetime.now() - start_time
#         if run_time > datetime.timedelta(seconds=50):
#             job.abort()
#             raise TimeoutError("Job took too long")
#     result = job.get_result(calc_data)
#     assert len(result.graph) > 0


def test_calculation_input():
    #TODO: can add asserts about vector lengths
    from dplus.CalculationInput import CalculationInput
    gen_input2 = CalculationInput()
    fixed_state_file = fix_file(os.path.join(root_path, "files", 'sphere.state'))
    gen_input = CalculationInput.load_from_state_file(fixed_state_file, USE_GPU)

def test_build_simple_state():
    state_json = {
        "DomainPreferences": {
            "Convergence": 0.001,
            "DrawDistance": 50,
            "Fitting_UpdateDomain": False,
            "Fitting_UpdateGraph": True,
            "GridSize": 200,
            "LevelOfDetail": 3,
            "OrientationIterations": 100,
            "OrientationMethod": "Monte Carlo (Mersenne Twister)",
            "SignalFile": "",
            "UpdateInterval": 100,
            "UseGrid": False,
            "qMax": 7.5
        },
        "FittingPreferences": {
            "Convergence": 0.1,
            "DerEps": 0.1,
            "DoglegType": "Traditional Dogleg",
            "FittingIterations": 20,
            "LineSearchDirectionType": "",
            "LineSearchType": "Armijo",
            "LossFuncPar1": 0.5,
            "LossFuncPar2": 0.5,
            "LossFunction": "Tolerant Loss",
            "MinimizerType": "Line Search",
            "NonlinearConjugateGradientType": "",
            "StepSize": 0.01,
            "TrustRegionStrategyType": "Levenberg-Marquardt",
            "XRayResidualsType": "Normal Residuals"
        },
        "Viewport": {
            "Axes_at_origin": True,
            "Axes_in_corner": True,
            "Pitch": 35.264385223389,
            "Roll": 0,
            "Yaw": 225.00004577637,
            "Zoom": 8.6602535247803,
            "cPitch": 35.264385223389,
            "cRoll": 0,
            "cpx": -4.9999966621399,
            "cpy": -5.0000033378601,
            "cpz": 4.9999990463257,
            "ctx": 0,
            "cty": 0,
            "ctz": 0
        },
        "Domain": {
            "Geometry": "Domains",
            "ModelPtr": 1,
            "Populations": [
                {
                    "ModelPtr": 2,
                    "Models": [{
                        "Constraints": [[{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                            [{
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }]],
                        "ExtraConstraints": [{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                        "ExtraMutables": [False,
                                          False],
                        "ExtraParameters": [1,
                                            0],
                        "ExtraSigma": [0,
                                       0],
                        "Location": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "LocationConstraints": {
                            "alpha": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "beta": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "gamma": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "x": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "y": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "z": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }
                        },
                        "LocationMutables": {
                            "alpha": False,
                            "beta": False,
                            "gamma": False,
                            "x": False,
                            "y": False,
                            "z": False
                        },
                        "LocationSigma": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "ModelPtr": 4,
                        "Mutables": [[False,
                                      False],
                                     [False,
                                      False]],
                        "Name": "",
                        "Parameters": [[0,
                                        333],
                                       [1,
                                        400]],
                        "Sigma": [[0,
                                   0],
                                  [0,
                                   0]],
                        "Type": ",2",
                        "Use_Grid": True,
                        "nExtraParams": 2,
                        "nLayers": 2,
                        "nlp": 2
                    },
                        {
                        "Constraints": [[{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                            [{
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }]],
                        "ExtraConstraints": [{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                        "ExtraMutables": [False,
                                          False,
                                          False],
                        "ExtraParameters": [1,
                                            0,
                                            10],
                        "ExtraSigma": [0,
                                       0,
                                       0],
                        "Location": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "LocationConstraints": {
                            "alpha": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "beta": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "gamma": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "x": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "y": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "z": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }
                        },
                        "LocationMutables": {
                            "alpha": False,
                            "beta": False,
                            "gamma": False,
                            "x": False,
                            "y": False,
                            "z": False
                        },
                        "LocationSigma": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "ModelPtr": 6,
                        "Mutables": [[False,
                                      False],
                                     [False,
                                      False]],
                        "Name": "",
                        "Parameters": [[0,
                                        333],
                                       [1,
                                        400]],
                        "Sigma": [[0,
                                   0],
                                  [0,
                                   0]],
                        "Type": ",0",
                        "Use_Grid": True,
                        "nExtraParams": 3,
                        "nLayers": 2,
                        "nlp": 2
                    }],
                    "PopulationSize": 1,
                    "PopulationSizeMut": False
                }
            ],
            "Scale": 1,
            "ScaleMut": False
        }
    }
    s = State()
    s.load_from_dictionary(state_json)
    spheres = s.get_models_by_type("Sphere")
    assert len(spheres) == 1
    
    uhc = UniformHollowCylinder()
    s = State()
    s.Domain.populations[0].add_model(uhc)
    uhcs = s.get_models_by_type("Uniform Hollow Cylinder")
    assert len(uhcs) == 1

def test_constraints_and_parameters():
    uhc=UniformHollowCylinder()
    try:
        uhc.layer_params[1]["radius"]=2.0 
        #will raise error "2.0 can only be set to an instance of Parameter"
    except ValueError as e :
        assert str(e) == '2.0 can only be set to an instance of Parameter'
    else:
        assert False 

    uhc=UniformHollowCylinder()
    uhc.layer_params[1]["radius"].value=2.0
    uhc.extra_params["height"].value=3.0
    uhc.location_params["x"].value=2

    '''
    Initialization of Parameter or Constarints
    '''
    p=Parameter()  #creates a parameter with value: '0', sigma: '0', mutable: 'False', and the default constraints.
    p=Parameter(7) #creates a parameter with value: '7', sigma: '0', mutable: 'False', and the default constraints.
    p=Parameter(sigma=2) #creates a parameter with value: '0', sigma: '2', mutable: 'False', and the default constraints.
    p.value= 4  #modifies the value to be 4.
    p.mutable=True #modifies the value of mutable to be 'True'.
    p.sigma=3 #modifies sigma to be 3.
    p.constraints=Constraints(min_val=5) #sets constraints to a 'Constraints' instance whose minimum value (min_val) is 5.

    c=Constraints(min_val=5) #creates a 'Constraints' instance whose minimum value is 5 and whose maximum value is the default ('infinity').

#the extra examples at the end of the manual:

def test_example_one_sphere_fit():
    from dplus.CalculationInput import CalculationInput
    from dplus.CalculationRunner import EmbeddedLocalRunner

    runner = EmbeddedLocalRunner()
    state_file = os.path.join(root_path, "files", "sphere.state")
    fixed_state_file = fix_file(state_file)
    input = CalculationInput.load_from_state_file(fixed_state_file, USE_GPU)
    result = runner.fit(input)
    assert result

def test_example_load_uhc_and_generate():
    runner = EmbeddedLocalRunner()
    uhc = UniformHollowCylinder()
    uhc.layer_params[1]["radius"].value = 2.0
    uhc.extra_params.height.value = 3.0
    uhc.location_params.x.value = 2

    caldata = CalculationInput(USE_GPU)
    caldata.Domain.populations[0].add_model(uhc)
    result = runner.generate(caldata)
    assert len(result.graph) > 0

def test_example_three_generate_pdb():
    pdb_file = os.path.join(root_path, "files", "1JFF.pdb")
    caldata = CalculationInput.load_from_PDB(pdb_file, 5)
    caldata.use_gpu = USE_GPU
    runner = EmbeddedLocalRunner()
    result = runner.generate(caldata)
    assert len(result.graph) > 0

        

def test_example_four_fit_UniformHollowCylinder():
    if not USE_GPU:
        pytest.skip("NO GPU")
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


def test_save_amplitude_and_load():

    from dplus.DataModels.models import AMP
    
    def my_func(q, theta, phi):
        return np.complex64(q + 0.0j)

    a = Amplitude(50, 7.5)
    a.fill(my_func)
    a.description = "example"
    a.save(os.path.join(session, "artificialgrid.ampj"))
    input = CalculationInput(USE_GPU)
    input.DomainPreferences.use_grid = True
    input.DomainPreferences.q_max = 7.5
    input.DomainPreferences.grid_size = 50
    a = AMP()
    a.filename = os.path.join(session, "artificialgrid.ampj")
    a.centered = True
    input.Domain.populations[0].children.append(a)
    runner = EmbeddedLocalRunner()
    result = runner.generate(input)

def test_get_amp_func():

    runner = EmbeddedLocalRunner()

    # without model name, should name file 00000000.ampj
    uhc=UniformHollowCylinder()
    caldata = CalculationInput(USE_GPU)
    caldata.Domain.populations[0].add_model(uhc)

    expectedFileName = os.path.join(session, '%08d' % (int(uhc.model_ptr)) + ".ampj")
    if os.path.isfile(expectedFileName):
        os.remove(expectedFileName)

    result=runner.generate(caldata)
    dest_folder = result.get_amp(uhc.model_ptr, session)

    assert dest_folder == session
    assert os.path.isfile(expectedFileName)

    # with model name, should name file test_hc.ampj
    uhc=UniformHollowCylinder()
    uhc.name="test_hc"
    caldata = CalculationInput(USE_GPU)
    caldata.Domain.populations[0].add_model(uhc)

    expectedFileName = os.path.join(session, "test_hc.ampj")
    if os.path.isfile(expectedFileName):
        os.remove(expectedFileName)

    runner = EmbeddedLocalRunner()
    result=runner.generate(caldata)
    dest_folder = result.get_amp(uhc.model_ptr, session)

    assert dest_folder == session

    assert os.path.isfile(expectedFileName)

def test_load_and_view_amplitude():
    #loading an existing amplitude and viewing the values it contains
    from dplus.Amplitudes import Amplitude
    my_amp = Amplitude.load(os.path.join(root_path, "files", "myamp.ampj"))

    for c in my_amp.complex_amplitude_array:
        print(c)

def test_create_fill_save_calc_amplitude():
    import numpy as np
    from dplus.Amplitudes import Amplitude

    def my_func(q, theta, phi):
        return np.complex64(q+1 + 0.0j)

    a = Amplitude(80, 7.5)
    a.description= "An example amplitude"						 
    a.fill(my_func)
    a.save("myfile.ampj")

    output_intrp = a.get_interpolation(5,3,6)
    output_intrp_arr = a.get_interpolation([1,2,3],3,6)