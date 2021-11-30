import json
import math
import os
import struct

import pytest

from dplus.CalculationInput import CalculationInput
from dplus.PyCeresOptimizer import PyCeresOptimizer
from dplus.CalculationRunner import LocalRunner, WebRunner
from tests.old_stuff.fix_state_files import fix_file
from tests.reviewer_tests.utils import DplusProps
from tests.test_settings import exe_directory
import numpy as np
token="7b6b2130920eaa02cd67d346960e6d4819331400"

web=False

class TestFitRun(DplusProps):
    def get_input(self, test_path):
        test_name = os.path.basename(os.path.normpath(test_path))
        state_file = os.path.join(test_path, test_name + ".state")
        fixed_state_file = fix_file(state_file)
        input = CalculationInput.load_from_state_file(fixed_state_file)
        input.use_grid = True
        input.use_gpu = True
        return input

    def save_for_web_tests(self, result, session_folder):
        os.makedirs(session_folder, exist_ok=True)
        with open(os.path.join(session_folder, "data.json"), 'w') as file:
            json.dump(result._raw_result, file)

    def test_run(self, test_folder_path):
        #first, do basic checks:
        expected_state=self.get_expected_state(test_folder_path)
        input=self.get_input(test_folder_path)
        #then run the program:
        session_folder=self.get_session_folder(test_folder_path)

        if web:
            api = WebRunner("http://192.168.18.100/", token)
        else:
            api=LocalRunner(exe_directory, session_folder)
        
        test_python_ceres = True
        if test_python_ceres:
            python_fit = PyCeresOptimizer(input, api)
            python_fit.solve()
            result = python_fit.save_dplus_arrays(python_fit.best_results, os.path.join(session_folder, "data.json"))

        else:
            result = api.fit(input)

        if web:
            self.save_for_web_tests(result, session_folder)
        #and finally, a sanity check on the results
        try:
            if result.error["code"]!=0:
                print("Result returned error:", result.error)
        except AttributeError:
            pass

class TestFitCorrect(DplusProps):
    def test_parameters_correct(self, test_folder_path):
        #expected params
        expected_state=self.get_expected_state(test_folder_path)
        expected_params=expected_state.get_mutable_params()

        #result params
        result = self.get_result(test_folder_path, fit=True)
        result_state = result.result_state
        result_params=result_state.get_mutable_params()

        total_passed=True
        for expected_model, result_model in zip(expected_params, result_params):
            for expected_param, result_param in zip(expected_model, result_model):
                e_val=expected_param.value
                r_val=result_param.value

                passed=True
                estr_whole, e_str_dec = str(e_val.split("."))
                prec = 1 * pow(10, -len(e_str_dec))
                diff = abs (e_val - r_val)
                if diff > prec:
                    passed = False
                if not passed:
                    print("Failed param check, expected:", e_val, "got:", r_val)
                total_passed=total_passed and passed

        assert total_passed

