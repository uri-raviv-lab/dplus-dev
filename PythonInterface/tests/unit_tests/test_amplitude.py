import json

import pytest
import struct

from dplus.Amplitudes import Amplitude
from dplus.DataModels.models import AMP
from dplus.State import State
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner
from tests.test_settings import exe_directory, session
import os
import numpy as np
from tests.old_stuff.fix_state_files import fix_file
import math
pi=math.pi

test_dir = os.path.join(os.path.dirname(__file__), "files_for_tests")

def test_save_amplitude_matches():
    filename = os.path.join(test_dir, "tinyAmp.amp")
    tmp_file_name = os.path.join(session, "tmpAmp.ampj")
    amp = Amplitude.load(filename)
    Amplitude.save(amp, tmp_file_name)
    amp_to_check = Amplitude.load(tmp_file_name)
    assert np.all(amp_to_check._values == amp._values)


def test_save_amp_correct():
    def _check_sigma(obs, exp, sig):
        lower_limit = exp - sig
        upper_limit = exp + sig
        test1 = (obs >= lower_limit)
        test2 = (obs <= upper_limit)
        return test1 and test2

    def _test_points(result, expected):
        failed_sig = {
            1: [],
            2: [],
            3: []
        }
        for i in range(len(expected.y)):
            for j in range(1, 3):
                passed = _check_sigma(result.y[i], expected.y[i], 0.00001)
                a = struct.pack("<dd", result.y[i], expected.y[i])
                b = struct.unpack("<qq", a)
                test = b[1] - b[0]
                if not passed:
                    if abs(test) > 256:
                        failed_sig[j].append((result.y[i], expected.intensity[i], expected.sigma[i], expected.q[i]))

        percent_one = len(failed_sig[1]) / len(expected.y)
        percent_two = len(failed_sig[2]) / len(expected.y)
        percent_three = len(failed_sig[3]) / len(expected.y)
        test = percent_one < .5 and percent_two < .3 and percent_three < .1
        return test

    runner = LocalRunner(exe_directory, session)
    filename = os.path.join(test_dir, "tinyAmp.ampj")
    tmp_file_name = os.path.join(session, "tmpAmp.ampj")
    state_file = os.path.join(test_dir, "ampcontaining.state")
    fixed_state_file = fix_file(state_file)
    input = CalculationInput.load_from_state_file(fixed_state_file)
    my_amp = input.get_models_by_type('AMP')[0]
    my_amp.filename = filename

    # testfilename=os.path.join(session, "firssState.state")
    # with open(testfilename, 'w') as file:
    #    file.write(json.dumps(input._state.serialize()))
    result1 = runner.generate(input)

    my_amp.filename = tmp_file_name
    result2 = runner.generate(input)
    assert _test_points(result2, result1)


def test_save2_amp_correct():
    input = CalculationInput()
    a = AMP()
    input.Domain.populations[0].children.append(a)
    input.DomainPreferences.use_grid = True
    input.DomainPreferences.q_max = 1e-007
    input.DomainPreferences.grid_size = 20
    a = input.get_models_by_type('AMP')[0]
    a.filename = os.path.join(test_dir, "tinyAmp.ampj")
    a.centered = True
    runner = LocalRunner(exe_directory, session)
    result = runner.generate(input)
    assert len(result.graph)


def test_create_grid_save_correct():
    def my_func(q, theta, phi):
        return np.complex64(q + 0.0j)

    a = Amplitude(50, 7.5)
    a.fill(my_func)
    a.description = "example"
    a.save(os.path.join(session, "artificialgrid.ampj"))
    input = CalculationInput()
    input.DomainPreferences.use_grid = True
    input.DomainPreferences.q_max = 7.5
    input.DomainPreferences.grid_size = 50
    a = AMP()
    a.filename = os.path.join(session, "artificialgrid.ampj")
    a.centered = True
    input.Domain.populations[0].children.append(a)
    runner = LocalRunner(exe_directory, session)
    result = runner.generate(input)


def trying_limits():
    '''
	(-3.316427134992193+0j) pi pi
	(-0.5332666200809292+0j) 3pi pi
	5pi pi  crashes
	(-0.5332666200809292+0j) 3pi 3pi
	(-0.5332666200809416+0j) 3pi 5pi
	(-6.270075238427314+0j) -5pi 83pi
	(-6.270075238427299+0j) -5pi 3pi 
	(-6.270075238427314+0j) -7pi 3pi
	(-6.270075238427299+0j) -100pi 3pi
	'''
    my_amp=Amplitude.load(r"C:\Users\devora\Downloads\amp.amp")
    tmp_file_name = r"C:\Users\devora\Downloads\amp.ampj"
    Amplitude.save(my_amp, tmp_file_name)
    amp_to_check = Amplitude.load(tmp_file_name)

    result=amp_to_check.interpolate_theta_phi_plane(1, 0.3, 0.1)

    q=0.1
    theta=4.5*pi
    phi=0.1
    #my_amp = Amplitude.load(os.path.join(test_dir, "sphere.ampj"))
    #doesnt_work = my_amp.interpolate_theta_phi_plane(7.5, theta, phi * pi)


    #theta = 0.00001, phi= 806pi, it crashes without raising an exception
	#0.5pi, 921
	#pi, 803
	#2pi, 563pi
	#3pi, 323pi
	#almost 4pi, 85pi

#trying_limits()

class ErrorException(Exception):

    def __init__(self, output, expected):
        self.output = output
        self.expected = expected

    def __str__(self):
        return "output: " + str(self.output) + " expected: " + str(self.expected)


def test_interpolation():

    from dplus.Amplitudes import Amplitude, sph2cart
    from math import pi
    import pytest

    amp_file = os.path.join(test_dir, "slab.ampj")
    my_amp = Amplitude.load(amp_file)

    assert my_amp.get_interpolation(6.5, 0.6, 2 * pi) == (-41.162962945846836 + 0j)
    assert my_amp.get_interpolation([0, 1, 2, 3], pi, pi) == [(-10600 + 0j), (-8921.561539944996 + 0j), (-4819.4000059187765 + 0j), (-498.62402847819754 + 0j)]
    assert my_amp.get_interpolation([0, 1, 2.5, 3.8], pi, pi) == [(-10600 + 0j), (-8921.561539944996 + 0j), (-2536.8136711460065 + 0j), (1708.521236744822 + 0j)]
    assert my_amp.get_interpolation(1, pi, pi) == (-8921.561539944996 + 0j)

    def my_func(q, th, ph):
        return np.complex64(q + th + ph + q * 1j)

    my_amp_sph = Amplitude(80, 7.5)
    my_amp_sph.fill(my_func)

    my_amp_cart = Amplitude(80, 7.5)
    my_amp_cart.fill_cart(my_func)

    def check_sph(_input, amp):
        try:
            output = amp.get_interpolation(*_input)
            expected = my_func(*_input)
            assert (output.real, output.imag) == pytest.approx((expected.real, expected.imag), rel=1e-1)
        except:
            raise ErrorException(output, expected)

    def check_cart(_input, amp):
        try:
            output = amp.get_interpolation(*_input)
            expected = my_func(*sph2cart(*_input))
            assert (output.real, output.imag) == pytest.approx((expected.real, expected.imag), rel=1e-1)
        except:
            raise ErrorException(output, expected)


    total = 0
    failed_sph = 0
    failed_cart = 0
    for q in np.linspace(0, 7.5, 20):
        for th in np.linspace(0, pi, 20):
            for ph in np.linspace(0, 2*pi, 20):
                total += 1
                try:
                    _input = [q, th, ph]
                    check_sph(_input, my_amp_sph)
                except ErrorException as e:
                    #print("-"*100, "\n", "sphr:", _input, "\n", e)
                    failed_sph += 1
                try:
                    _input = [q, th, ph]
                    check_cart(_input, my_amp_cart)
                except ErrorException as e:
                    #print("-"*100, "\n", "cart:", _input,"\n", "cart:", sph2cart(*_input), "\n", e)
                    failed_cart += 1
    print("failed_sph/total =", failed_sph/total)
    print("failed_cart/total =", failed_cart/total)
    # input0 = [0, 0, 0]
    # check(input0)
    # input1 = [2, 2, 0.5]
    # check(input1)
    # input2 = [6, 1.4, pi]
    # check(input2)
    # input3 = [7.5, pi, 2*pi-0.2]
    # check(input3)
    # input4 = [1, 1, 1]
    # check(input4)


def test_use_grid():
    state_file = os.path.join(test_dir, "check_use_grid.state")
    input = CalculationInput.load_from_state_file(state_file)
    runner = LocalRunner(exe_directory, session)
    input.use_gpu = False
    input.Domain.children[0].children[0].children[1].children[0].use_grid = False
    with pytest.raises(ValueError):
        result = runner.generate(input)


def save_to_out_file(filename, grid):
    '''
    receives file name, and saves the results to the file.
    :param filename: string of filename/path
    '''
    with open(filename, 'w') as out_file:
        # out_file.write("# Integration parameters:\n")
        # out_file.write("#\tqmax\t{}\n".format(grid.q_max))
        # out_file.write("#\tOrientation Method\t{}\n".format(grid.orientation_method))
        # out_file.write("#\tOrientation Iterations\t{}\n".format(grid.orientation_iterations))
        # out_file.write("#\tConvergence\t{}\n\n".format(grid.convergence))

        # for value in self.headers.values():
        #     out_file.write(value)
        for key, value in grid.graph.items():
            out_file.write('{:.5f}\t{:.20f}\n'.format(key, value))
        out_file.close()


def get_expected_signal(out_filename):
    sigma_chi_square = chi_square = None
    Q = []
    intensity = []
    sigma = []
    with open(out_filename, 'r') as file:
        for line in file:
            if len(line.strip()) < 1:
                continue
            if line[0] == '#':
                if "Average Chi square" in line:
                    chi_square = float(line.split()[-1])
                if "Sigma Chi square" in line:
                    sigma_chi_square = float(line.split()[-1])
            elif line[0] == 'q':
                continue
            else:
                try:
                    q, i, s = line.split()
                except ValueError:  # for curves I generated for testing Fit before they sent files
                    q, i = line.split()
                    s = 0
                Q.append(float(q))
                intensity.append(float(i))
                sigma.append(float(s))
    return {
        'q': Q,
        'intensity': intensity,
        'sigma': sigma,
        'sigma_chi_square': sigma_chi_square,
        'chi_square': chi_square
    }


def _chi_a_squ(result, expected):
    # chi_a^2 = 1/N \sum_i^N [(I^expected_i - I_calculated_i)/\sigam_i]^2
    N = len(expected.get('q'))
    sum_i_to_N = 0
    for i in range(N):
        expected_i = expected.get('intensity')[i]
        calculated_i = result[i]
        sigma_i = expected.get('sigma')[i]
        if sigma_i < 10e-4:
            sum_i_to_N += math.pow((expected_i - calculated_i), 2)
        else:
            sum_i_to_N += math.pow(((expected_i - calculated_i) / sigma_i), 2)
    chi_a_sq = sum_i_to_N / N
    return chi_a_sq


def test_calculate_intensity():
    # compare 2 signals from D+
    # path_signal_1 = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-d-plus.out"
    # path_signal_1 = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-d-plus-generate-again.out"
    # signal1 = get_expected_signal(path_signal_1)
    # signal2 = get_expected_signal(path_signal_1)
    # result = [val for val in signal2.get('intensity')]
    # chi_a_sq_val = _chi_a_squ(result, signal1)

    amp_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\myamp.ampj"
    out_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-python-myamp-size200.out"
    excepted_signal_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-d-plus-size200.out"
    result_1 = send_calculate_intensity(amp_filename, excepted_signal_filename, out_filename)
    #
    # amp_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\myamp-2.ampj"
    # out_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-python-myamp-2.out"
    # excepted_signal_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-d-plus-2.out"
    # result_2 = send_calculate_intensity(amp_filename, excepted_signal_filename, out_filename)

    # amp_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\myamp_helix2layrers.ampj"
    # out_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-python-myamp_helix2layrers.out"
    # excepted_signal_filename = r"C:\Users\chana\Source\DPlus\dplus\PythonInterface\tests\unit_tests\files_for_tests\intensity\signal-d-plus-helix2layrers.out"
    # result_3 = send_calculate_intensity(amp_filename, excepted_signal_filename, out_filename)
    
    print("result 1:", result_1)
    # print("result 2:", result_2)
    # print("result 3:", result_3)



def send_calculate_intensity(amp_filename, excepted_signal_filename, out_filename):
    expected_signal = get_expected_signal(excepted_signal_filename)
    amp = Amplitude.load(amp_filename)
    q_size = len(expected_signal.get('q'))
    result = []
    # seeds = []
    # for q in np.linspace(0, amp.grid.q_max, q_size)
    #     seeds.append()

    with open(out_filename, 'w') as out_file:
        for q in np.linspace(0, amp.grid.q_max, q_size):
            res = amp.calculate_intensity(q)
            result.append(res)
            out_file.write('{:.5f}\t{:.20f}\n'.format(q, res))

    # for q in np.linspace(0, amp.grid.q_max, q_size):
    #     res = amp.calculate_intensity(q)
    #     result.append(res)
    chi_a_sq_val = _chi_a_squ(result, expected_signal)
    # from scipy.spatial import distance
    # from scipy.signal import correlate
    # from scipy import spatial
    from scipy import stats

    f_oneway = stats.f_oneway(expected_signal.get('intensity'), result)
    spearmanr = stats.spearmanr(expected_signal.get('intensity'), result)
    ttest_ind = stats.ttest_ind(expected_signal.get('intensity'), result)
    # chisquare = stats.chisquare(expected_signal.get('intensity'), result)
    
    # print("chisquare", chisquare)
    print("spearmanr", spearmanr)
    print("ttest_ind", ttest_ind)
    print("f_oneway", f_oneway)
    # euclid_distance_a_to_b = distance.euclidean(expected_signal.get('intensity'), result)
    # sqeuclidean = distance.sqeuclidean(expected_signal.get('intensity'), result)
    # # seuclidean = distance.seuclidean(expected_signal.get('intensity'), result)

    # # Find cross-correlation
    # xcorr = correlate(expected_signal.get('intensity'), result)

    # cosine = 1 - spatial.distance.cosine(expected_signal.get('intensity'), result)
    print("chi_a_sq_val", chi_a_sq_val)
    return chi_a_sq_val


if __name__ == "__main__":
    test_calculate_intensity()
