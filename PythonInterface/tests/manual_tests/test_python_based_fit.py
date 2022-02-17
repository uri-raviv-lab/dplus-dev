'''
A python implemention of the fitting algorithm, using scipy's optimization libraries
'''
from os.path import abspath
import numpy as np
import os
from scipy import optimize
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner
from tests.old_stuff.fix_state_files import fix_file

root_path = os.path.dirname(abspath(__file__))


def close_enough(x1, x2):
    if abs(x1 - x2) < 0.01:
        return True
    return False


def run_fit(input):
    generate_runner = EmbeddedLocalRunner()

    def run_generate(xdata, *params):
        '''
        scipy's optimization algorithms require a function that receives an x array and an array of parameters, and
        returns a y array.
        this function will be called repeatedly, until scipy's optimization has completed.
        :param xdata:
        :param params:
        :return:
        '''
        input.set_mutable_parameter_values(
            params)  # we take the parameters given by scipy and place them inside our parameter tree
        generate_results = generate_runner.generate(input)  # call generate
        return np.array(generate_results.y)  # return the results of the generate call

    x_data = input.x
    y_data = input.y
    p0 = input.get_mutable_parameter_values()
    method = 'lm'
    popt, pcov = optimize.curve_fit(run_generate, x_data, y_data, p0=p0, method=method)

    # popt is the optimized set of parameters from those we have indicated as mutable
    # we can insert them into our original input and run generate to get the results of generate with them
    input.set_mutable_parameter_values(popt)
    best_results = generate_runner.generate(input)
    return input, best_results


def test_fit_1():
    state_file = os.path.join(root_path, "files", "2_pops.state")
    fixed_state_file = fix_file(state_file)
    input = CalculationInput.load_from_state_file(fixed_state_file)
    result_input, result = run_fit(input)
    muts = result_input.get_mutable_params()
    assert close_enough(muts[0][0].value, 1)
    assert close_enough(muts[1][0].value, 3)


def test_fit_2():
    from dplus.State import State
    from dplus.DataModels.models import Sphere
    input = CalculationInput()
    s = Sphere()
    s.use_grid = True
    s.layer_params[1]["radius"].value = 2
    s.layer_params[1]["radius"].mutable = True
    input.Domain.populations[0].add_model(s)
    signal_file = os.path.join(root_path, "files", "sphere.out")
    input.DomainPreferences.signal_file = signal_file
    input.DomainPreferences.use_grid = True
    input.DomainPreferences.grid_size = 200
    input.DomainPreferences.orientation_iterations = 10000
    input.use_gpu = True
    result_input, result = run_fit(input)
    assert close_enough(s.layer_params[1]["radius"].value, 1)


def test_fit_manual():
    input = CalculationInput.load_from_state_file(os.path.join(root_path, "files", "2_pops_fixed.state"))
    generate_runner = EmbeddedLocalRunner()

    def run_generate(xdata, *params):
        '''
        scipy's optimization algorithms require a function that receives an x array and an array of parameters, and
        returns a y array.
        this function will be called repeatedly, until scipy's optimization has completed.
        :param xdata:
        :param params:
        :return:
        '''
        input.set_mutable_parameter_values(
            params)  # we take the parameters given by scipy and place them inside our parameter tree
        generate_results = generate_runner.generate(input)  # call generate
        return np.array(generate_results.y)  # return the results of the generate call

    x_data = input.x
    y_data = input.y
    p0 = input.get_mutable_parameter_values()
    method = 'lm'  # lenenberg-marquadt (see scipy documentation)
    popt, pcov = optimize.curve_fit(run_generate, x_data, y_data, p0=p0, method=method)

    # popt is the optimized set of parameters from those we have indicated as mutable
    # we can insert them back into our CalculationInput and create the optmized parameter tree
    input.set_mutable_parameter_values(popt)
    # we can run generate to get the results of generate with them
    best_results = generate_runner.generate(input)
