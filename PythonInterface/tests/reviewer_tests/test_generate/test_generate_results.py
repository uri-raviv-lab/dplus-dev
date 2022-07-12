import json
import math
import os
import struct

import pytest

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner, WebRunner
from tests.old_stuff.fix_state_files import fix_file
from tests.reviewer_tests.utils import DplusProps
from tests.test_settings import tests_dir
import numpy as np

web=False

class TestGenerateRun(DplusProps):
    def get_expected_signal(self, test_path):
        expected=super().get_expected_signal(test_path)
        assert len(expected.q)>0
        return expected

    def get_input(self, test_path):
        test_name = os.path.basename(os.path.normpath(test_path))
        state_file = os.path.join(test_path, test_name + ".state")
        fixed_state_file = fix_file(state_file)
        input = CalculationInput.load_from_state_file(fixed_state_file)
        input.use_gpu = ("gpu" in test_path)
        return input

    def test_GPU(self):
        test_file=os.path.join(tests_dir, 'unit_tests', 'files_for_tests', 'sphere_GPU.state')
        input = CalculationInput.load_from_state_file(test_file)
        input.use_gpu = True
        api = EmbeddedLocalRunner()
        result = api.generate(input)
        assert len(result.graph)>0

    def save_result_tests(self, result, test_folder_path):
        session_folder = self.get_session_folder(test_folder_path)
        os.makedirs(session_folder, exist_ok=True)
        with open(os.path.join(session_folder, "data.json"), 'w') as file:
            json.dump(result._raw_result, file)

    def run_calc_and_save_result(self, input, test_folder_path):
        if web:
            api = WebRunner("http://192.168.18.100/", "06946fe6c6a3acd625dabbc7bdb905940140d8aa")
        else:
            api = EmbeddedLocalRunner()
        result = api.generate(input)
        self.save_result_tests(result, test_folder_path)
        return result

    #@pytest.mark.timeout(3000)
    def test_run(self, test_folder_path):
        # first, do basic checks:
        expected = self.get_expected_signal(test_folder_path)
        input = self.get_input(test_folder_path)
        input._fix_use_grid()
        #then run the program:
        result = self.run_calc_and_save_result(input, test_folder_path)
        

        #and finally, a sanity check on the results
        if result.error["code"]!=0:
            print("Result returned error:", result.error)
            assert False

        test_correct = self.then_test_correct(result, test_folder_path)
        assert test_correct

    def placeholder(self):
        """
                if not test_correct and "short" in test_folder_path:
            while i<3:
                i+=1
                result = api.generate(input)
                self.save_result_tests(result, test_folder_path)
                test_correct = self.then_test_correct(result, test_folder_path)
                if test_correct:
                    break
        """
        pass


    def then_test_correct(self, result, test_folder_path):
        expected = self.get_expected_signal(test_folder_path)
        assert len(expected.q) == len(result.y)

        try:
            self._save_out_file(result, expected, test_folder_path)
        except Exception as e:  # doesn't matter what, we still want the assert
            pass

        a = self._chi_sq1(result, expected)
        b = self._chi_sq2(result, expected)
        c = self._test_points(result, expected)
        d = self._test_normalized_Rqi(result, expected)
        e = self._test_div(result, expected)
        test_name = self.get_test_name(test_folder_path)
        if not a:
            print(test_name, " failed chi sq 1")
        if not b:
            print(test_name, " failed chi sq 2")
        if not c:
            print(test_name, " failed points test")
        if not d:
            print(test_name, " failed normalized Rqi")
        return a or b or c or d or e  # the useful information is in the captured stdout call

    def _chi_a_squ(self, result, expected):
        # chi_a^2 = 1/N \sum_i^N [(I^expected_i - I_calculated_i)/\ sigma_i]^2
        N = len(expected.q)
        sum_i_to_N = 0
        for i in range(N):
            expected_i = expected.intensity[i]
            calculated_i = result.y[i]
            sigma_i = expected.sigma[i]
            if sigma_i < 10e-4:
                sum_i_to_N += math.pow((expected_i - calculated_i), 2)
            else:
                sum_i_to_N += math.pow(((expected_i - calculated_i) / sigma_i), 2)
        chi_a_sq = sum_i_to_N / N
        return chi_a_sq

    def _chi_sq2(self, result, expected):
        chi_a_sq = self._chi_a_squ(result, expected)
        test = chi_a_sq < expected.chi_square + expected.sigma_chi_square * 2
        return test

    def _chi_sq1(self, result, expected):
        chi_a_sq = self._chi_a_squ(result, expected)
        test = chi_a_sq < expected.chi_square + expected.sigma_chi_square * 1
        return test

    def _test_points(self, result, expected):
        failed_sig = {
            1: [],
            2: [],
            3: []
        }
        for i in range(len(expected.q)):
            for j in range(1, 3):
                passed = self._check_sigma(result.y[i], expected.intensity[i], j * expected.sigma[i])
                a = struct.pack("<dd", result.y[i], expected.intensity[i])
                b = struct.unpack("<qq", a)
                test = b[1] - b[0]
                if not passed:
                    if abs(test) > 256:
                        failed_sig[j].append((result.y[i], expected.intensity[i], expected.sigma[i], expected.q[i]))

        percent_one = len(failed_sig[1]) / len(expected.q)
        percent_two = len(failed_sig[2]) / len(expected.q)
        percent_three = len(failed_sig[3]) / len(expected.q)

        test = percent_one < .5 and percent_two < .3 and percent_three < .1
        return test

    def _check_sigma(self, obs, exp, sig):
        lower_limit = exp - sig
        upper_limit = exp + sig
        test1 = (obs >= lower_limit)
        test2 = (obs <= upper_limit)
        return test1 and test2

    def _test_normalized_Rqi(self, result, expected):
        normalized = []
        for i in range(len(expected.q)):
            R_q_i = result.y[i] - expected.intensity[i]
            Max_q_i = max(result.y[i], expected.intensity[i])
            if Max_q_i != 0:
                Normalized_R_q_i = abs(R_q_i / Max_q_i)
            else:
                Normalized_R_q_i = 0
            normalized.append(Normalized_R_q_i)

        max_n = max(normalized)
        mean_n = np.mean(normalized)
        test = mean_n < 1e-8 and max_n < 1e-8
        return test

    def _test_div(self, result, expected):
        for i in range(len(expected.q)):
            if abs(result.y[i] / expected.intensity[i]) < 0.999999:
                return False
        return True

    def _save_out_file(self, result, expected, test_folder_path):
        session_folder = self.get_session_folder(test_folder_path)
        test_name = os.path.basename(os.path.normpath(test_folder_path))
        result.save_to_out_file(os.path.join(session_folder, test_name + "python_to_out.out"))
        expected.save_to_out_file(os.path.join(session_folder, test_name + "TestStandardCopy.out"))


