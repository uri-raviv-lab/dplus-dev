
from dplus.CalculationRunner import EmbeddedLocalRunner
from dplus.DataModels.models import Sphere, SpacefillingSymmetry
from dplus.CalculationInput import CalculationInput
from tests.test_settings import USE_GPU, session
import math
import os

def test_create_state_with_cylinder():
    from dplus.DataModels.models import UniformHollowCylinder
    uhc=UniformHollowCylinder()
    uhc.name="test_hc"



def test_add_layer():
    from dplus.DataModels.models import ManualSymmetry, UniformHollowCylinder
    ms=ManualSymmetry()
    ms.children.append(UniformHollowCylinder())
    ms.add_layer()
    for param in ms.layer_params[0]:
        ms.layer_params[0][param].value=1

def test_del_layer():
    from dplus.DataModels.models import ManualSymmetry
    ms = ManualSymmetry()
    for i in range(10):
        ms.add_layer()
        ms.layer_params[i].x.value = i + 0.1
        ms.layer_params[i].y.value = i + 0.2
        ms.layer_params[i].z.value = i + 0.3
    ms.del_layer(0)
    ms.del_layer(range(2, 5))
    ms.del_layer([4, 3])

    assert ((ms.layer_params[2].x.value == 6.1) & (ms.layer_params[2].y.value == 6.2) &
           (ms.layer_params[2].z.value == 6.3)) & (len(ms.layer_params) == 4)
def _chi_a_squ(result, expected):
        # chi_a^2 = 1/N \sum_i^N [(I^expected_i - I_calculated_i)/\sigam_i]^2
        N = len(expected._calc_data.x)
        sum_i_to_N = 0
        for i in range(N):
            expected_i = expected.y[i]
            calculated_i = result.y[i]
            #sigma_i = expected.sigma[i]
            #if sigma_i < 10e-4:
            sum_i_to_N += math.pow((expected_i - calculated_i), 2)
            #else:
            #    sum_i_to_N += math.pow(((expected_i - calculated_i) / sigma_i), 2)
        chi_a_sq = sum_i_to_N / N
        return chi_a_sq

def test_model_building():

    runner = EmbeddedLocalRunner()

    sphere = Sphere()
    space_filling = SpacefillingSymmetry()
    space_filling.children.append(sphere)
    input = CalculationInput(USE_GPU)
    input.Domain.populations[0].add_model(space_filling)
    input.export_all_parameters(os.path.join(session, "from_bottom_up.state"))

    result = runner.generate(input)

    space_filling1 = SpacefillingSymmetry()
    input1 = CalculationInput(USE_GPU)
    input1.Domain.populations[0].add_model(space_filling1)
    sphere1 = Sphere()
    input1.Domain.populations[0].models[0].children.append(sphere1)
    input1.export_all_parameters(os.path.join(session, "from_top_down.state"))
    

    result1 = runner.generate(input1)

    result.save_to_out_file(os.path.join(session, 'from_bottom_up.out'))
    result1.save_to_out_file(os.path.join(session, 'from_top_down.out'))


    chi_sq = _chi_a_squ(result, result1)

    print(chi_sq) 
