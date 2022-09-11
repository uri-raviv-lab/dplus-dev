from cProfile import label
import os
import numpy as np
import pytest
from dplus.CalculationInput import CalculationInput
from dplus.Amplitudes import Amplitude
from dplus.CalculationResult import CalculationResult
from dplus.DataModels.models import Sphere, UniformHollowCylinder
from dplus.CalculationRunner import EmbeddedLocalRunner

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import math
from os.path import join, dirname
from tests.manual_tests.test_amplitude_manual import UniformSphere
from tests.test_settings import session
from pylab import *
from tests.test_settings import session

file_dir = join(dirname( __file__ ), "files_for_tests")

def test_easy_1():

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

    result = my_amp.get_intensity_q_theta(q)
    print(calc_res.y == pytest.approx(result))
    assert calc_res.y == pytest.approx(result)

def test_easy_without_ampj_file():
    # GENERATE: ------------------------
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

    # INTENSITY: ------------------------
    amp = runner.get_amp(sp.model_ptr)

    q = np.linspace(0, amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = amp.get_intensity_q_theta(q)

    # COMPARE: --------------------------
    assert calc_res.y == pytest.approx(result)

def test_qZ_qPerp():
    q = 0.1
    theta = 0.2

    qz_qPerp = Amplitude.q_theta_to_qZ_qPerp(q, theta)
    qZ = qz_qPerp['qZ']
    qPerp = qz_qPerp['qPerp']

    q_theta = Amplitude.qZ_qPerp_to_q_theta(qZ, qPerp)
    q_res = q_theta['q']
    theta_res = q_theta['theta']

    assert(q ==  pytest.approx(q_res))
    assert(theta ==  pytest.approx(theta_res))

def test_intensity_qZ_qPerp():
    # GENERATE: ------------------------
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

    # INTENSITY: ------------------------
    amp = runner.get_amp(sp.model_ptr)
    
    result = amp.get_intensity(q_min=0)
    

    plt.imshow(result, origin='lower', aspect=1, norm=colors.LogNorm(vmin=0, vmax=np.max(np.max(result))))
    #plt.show()


def test_hard_1():
    calc_in = CalculationInput.load_from_PDB(join(file_dir, "1jff.pdb"), 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 750

    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(join(file_dir, "intensity", "1jff.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    result = my_amp.get_intensity_q_theta(q, epsilon=1e-3)
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
    q = np.linspace(0, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
    theta = np.linspace(0, math.pi, 33)

    result = my_amp.get_intensity_q_theta(q, theta)

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
    
    result = my_amp.get_intensity_q_theta(q, theta, epsilon=1e-3)
   
    aspect = len(theta) / len(q)
    plt.imshow(result, origin='lower', aspect=aspect, norm=colors.LogNorm(vmin=0, vmax=np.max(np.max(result))))
    plt.show()

def test_2d_intensity_hard_2():
    calc_in = CalculationInput.load_from_PDB(join(file_dir, "1jff.pdb"), 7.5)
    calc_in.use_gpu = False
    calc_in.DomainPreferences.grid_size = 50
    calc_in.DomainPreferences.generated_points = 300
    runner = EmbeddedLocalRunner()
    calc_res = runner.generate(calc_in)

    my_amp = Amplitude.load(join(file_dir, "intensity", "1jff.ampj"))
    q_min=0
    q_list = np.linspace(q_min, my_amp.grid.q_max, calc_in.DomainPreferences.generated_points + 1)
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


def test_read_write_2D():
    sp = UniformHollowCylinder()
    sp.layer_params[1]['radius'].value = 3
    sp.layer_params[1]['ed'].value = 356
    sp.location_params['z'].value = 10.5
    my_amp = Amplitude.load(join(file_dir, "intensity", "my_sphere.ampj"))
    q = np.linspace(0, my_amp.grid.q_max, 50)
    theta = np.linspace(0, math.pi, 5)
    result = my_amp.get_intensity_q_theta(q, theta)

    filename1 = join(session, "file1")
    filename1 = CalculationResult.save_to_2D_out_file(q,theta,result,filename1)
    r_q, r_t, r_res = CalculationResult.read_2D_out_file(filename1)

    assert all(q == r_q)
    assert all(r_t == theta)
    assert all(r_res == result)

    filename2 = join(session, "file2")
    filename2 = CalculationResult.save_to_2D_out_file(r_q,r_t,r_res,filename2)

    with open(filename1) as file_1, open(filename2) as file_2:
        file_1_text = file_1.readlines()
        file_2_text = file_2.readlines()
        for line_1, line_2 in zip(file_1_text, file_2_text):
            assert line_1 == line_2

#test_intensity_qZ_qPerp()
#test_easy_2()
#test_easy_without_ampj_file()