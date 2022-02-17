import numpy as np
import pytest
from dplus.CalculationInput import CalculationInput
from dplus.Amplitudes import Amplitude  # calculate_intensity, get_intensity
from dplus.DataModels.models import Sphere
from dplus.CalculationRunner import LocalRunner
import matplotlib.pyplot as plt


file_dir = r'.\files_for_tests'

def test_easy_1():

    sp = Sphere()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5
    sp.name = 'my_sphere'

    calc_in = CalculationInput()  # .load_from_PDB(r'C:\Users\owner\Desktop\Eytan\For_Daniel\IMCMonomer.pdb', 7.5)
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)

    runner = LocalRunner()
    calc_res = runner.generate(calc_in)
    my_amp = Amplitude.load(file_dir + r'\intensity\my_sphere.ampj')
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points+1)
    result = []
    for i in q:
        res = my_amp.calculate_intensity(i)
        result.append(res)
    print(calc_res.y == pytest.approx(result))
    assert calc_res.y == pytest.approx(result)

def test_easy_2():

    sp = Sphere()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5
    sp.name = 'my_sphere'

    calc_in = CalculationInput()  # .load_from_PDB(r'C:\Users\owner\Desktop\Eytan\For_Daniel\IMCMonomer.pdb', 7.5)
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)

    runner = LocalRunner()
    calc_res = runner.generate(calc_in)
    my_amp = Amplitude.load(file_dir + r'\intensity\my_sphere.ampj')
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = my_amp.get_intensity(q)
    print(calc_res.y == pytest.approx(result))
    assert calc_res.y == pytest.approx(result)

def test_hard_1():
    calc_in = CalculationInput.load_from_PDB(file_dir + r'\1jff.pdb', 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 750

    runner = LocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(file_dir + r'\intensity\1jff.ampj')
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = my_amp.get_intensity(q, epsilon=1e-3)
    plt.semilogy(result, label='result')
    plt.semilogy(calc_res.y, label='calc_res')
    plt.legend()
    plt.show()
    rel = abs(np.array(result) - np.array(list(calc_res.y)))/np.array(result)  # np.array(list(calc_res.y))
    print(max(rel))
    assert result == pytest.approx(calc_res.y, rel=0.08)

def test_hard_2():
    calc_in = CalculationInput.load_from_PDB(file_dir + r'\1jff.pdb', 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 750
    runner = LocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(file_dir + r'\intensity\1jff.ampj')
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = []
    for i in q:
        result.append(my_amp.calculate_intensity(i))
    plt.semilogy(result, label='result')
    plt.semilogy(calc_res.y, label='calc_res')
    plt.legend()
    plt.show()
    rel = abs(np.array(result) - np.array(list(calc_res.y)))/np.array(list(calc_res.y))
    print(max(rel))
    assert result == pytest.approx(calc_res.y, rel=0.08)

test_easy_1()
test_easy_2()
test_hard_1()
test_hard_2()

