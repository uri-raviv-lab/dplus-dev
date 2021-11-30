import json
import struct

from dplus.Amplitudes import Amplitude
from dplus.DataModels.models import AMP
from dplus.State import State
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner
from tests.unit_tests.conftest import test_dir
from tests.test_settings import exe_directory, session
import os
import numpy as np
from tests.old_stuff.fix_state_files import fix_file
import math


def test_save_amplitude_matches():
    filename = os.path.join(test_dir, "tinyAmp.amp")
    tmp_file_name = os.path.join(session, "tmpAmp.ampj")
    amp = Amplitude.load(filename)
    Amplitude.save(amp, tmp_file_name)
    amp_to_check = Amplitude.load(tmp_file_name)
    assert np.all(amp_to_check.values == amp.values)


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
    input.Domain.populations[0].Children.append(a)
    input.DomainPreferences.use_grid = True
    input.DomainPreferences.q_max = 1e-007
    input.DomainPreferences.grid_size = 20
    a = input.get_models_by_type('AMP')[0]
    a.filename = os.path.join(test_dir, "tinyAmp.ampj")
    a.centered = True
    runner = LocalRunner(exe_directory, session)
    result = runner.generate(input)


def test_create_grid_save_correct():
    def my_func(q, theta, phi):
        return np.complex64(q + 0.0j)

    a = Amplitude(7.5, 50)
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
    input.Domain.populations[0].Children.append(a)
    runner = LocalRunner(exe_directory, session)
    result = runner.generate(input)
