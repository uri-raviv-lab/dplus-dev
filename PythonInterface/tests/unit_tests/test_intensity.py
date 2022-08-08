from cProfile import label
import numpy as np
import pytest
from dplus.CalculationInput import CalculationInput
from dplus.Amplitudes import Amplitude  # calculate_intensity, get_intensity
from dplus.DataModels.models import Sphere, UniformHollowCylinder
from dplus.CalculationRunner import EmbeddedLocalRunner

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import math
from os.path import join, dirname
from tests.test_settings import session
from pylab import *

file_dir = join(dirname( __file__ ), "files_for_tests")

def test_easy_1():

    sp = Sphere()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5
    sp.name = 'my_sphere'

    calc_in = CalculationInput()  # .load_from_PDB(r'C:\Users\owner\Desktop\Eytan\For_Daniel\IMCMonomer.pdb', 7.5)
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)

    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)
    my_amp = Amplitude.load(join(file_dir, 'intensity', 'my_sphere.ampj'))
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

    calc_in = CalculationInput() 
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)

    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)


    my_amp = Amplitude.load(join(file_dir, "intensity", "my_sphere.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)

    result = my_amp.get_intensity(q)
    print(calc_res.y == pytest.approx(result))
    assert calc_res.y == pytest.approx(result)

def test_hard_1():
    calc_in = CalculationInput.load_from_PDB(join(file_dir, "1jff.pdb"), 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 750

    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(join(file_dir, "intensity", "1jff.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = my_amp.get_intensity(q, epsilon=1e-3)
    # plt.semilogy(result, label='result')
    # plt.semilogy(calc_res.y, label='calc_res')
    # plt.legend()
    # plt.show()
    rel = abs(np.array(result) - np.array(list(calc_res.y)))/np.array(result)  # np.array(list(calc_res.y))
    print(max(rel))
    assert result == pytest.approx(calc_res.y, rel=0.08)

def test_hard_2():
    calc_in = CalculationInput.load_from_PDB(join(file_dir, "1jff.pdb"), 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 750
    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(join(file_dir, "intensity", "1jff.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = []
    for i in q:
        result.append(my_amp.calculate_intensity(i, epsilon=1e-3 ,max_iter=1000000))
    
    rel = abs(np.array(result) - np.array(list(calc_res.y)))/np.array(list(calc_res.y))
    print(max(rel))
    assert result == pytest.approx(calc_res.y, rel=0.08)

def test_2d_intensity_easy_1():

    sp = Sphere()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5
    sp.name = 'my_sphere'

    calc_in = CalculationInput() 
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)

    my_amp = Amplitude.load(join(file_dir, 'intensity', 'my_sphere.ampj'))
    q_list = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points+1)
    theta_list = np.linspace(0, math.pi, 33)#calc_in.DomainPreferences.generated_points + 1)
    result = [[0 for t in range(len(theta_list))] for q in range(len(q_list))] 

    q_idx = 0
    for q in q_list:
        t_idx = 0
        for t in theta_list:
            result[q_idx][t_idx] = my_amp.calculate_intensity(q, t)
            t_idx = t_idx + 1
        q_idx = q_idx + 1

    aspect = len(theta_list) / len(q_list)
    plt.imshow(result, origin='lower', aspect=aspect, norm=colors.LogNorm(vmin=0, vmax=np.max(np.max(result))))
    plt.show()

def test_2d_intensity_easy_2():
    sp = UniformHollowCylinder()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5

    calc_in = CalculationInput() 
    calc_in.use_gpu = False
    calc_in.Domain.populations[0].add_model(sp)
    calc_in.DomainPreferences.generated_points = 300

    my_amp = Amplitude.load(join(file_dir, "intensity", "my_sphere.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, 301)#calc_in.DomainPreferences.generated_points + 1)
    theta = np.linspace(0, math.pi, 33)#calc_in.DomainPreferences.generated_points + 1)

    result = my_amp.get_intensity(q, theta)

    aspect = len(theta) / len(q)
    plt.imshow(result, origin='lower', aspect=aspect, norm=colors.LogNorm(vmin=0, vmax=np.max(np.max(result))))
    plt.show()

def test_2d_intentsity_hard_1():
    calc_in = CalculationInput.load_from_PDB(join(file_dir, "1jff.pdb"), 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 300

    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(join(file_dir, "intensity", "1jff.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    theta = np.linspace(0, math.pi, 33)#calc_in.DomainPreferences.generated_points + 1)
    
    result = my_amp.get_intensity(q, theta, epsilon=1e-3)
   
    aspect = len(theta) / len(q)
    plt.imshow(result, origin='lower', aspect=aspect, norm=colors.LogNorm(vmin=0, vmax=np.max(np.max(result))))
    plt.show()
