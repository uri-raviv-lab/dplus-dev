
import math
import os
import struct

import pytest

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner
from tests.old_stuff.fix_state_files import fix_file
from tests.reviewer_tests.utils import DplusProps
from tests.test_settings import tests_dir
import numpy as np


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

    #@pytest.mark.timeout(3000)
    def test_run(self, test_folder_path):
        #first, do basic checks:
        expected=self.get_expected_signal(test_folder_path)
        input=self.get_input(test_folder_path)
        input._fix_use_grid()
        #then run the program:
        session_folder=self.get_session_folder(test_folder_path)
        api = EmbeddedLocalRunner()
        result = api.generate(input)
        #and finally, a sanity check on the results
        if result.error["code"]!=0:
            print("Result returned error:", result.error)

        assert len(expected.q) == len(result.y)

class TestGenerateCorrect(DplusProps):
    def test_correct(self, test_folder_path):
        a= self._chi_sq1(test_folder_path)
        b= self._chi_sq2(test_folder_path)
        c= self._test_points(test_folder_path)
        d= self._test_normalized_Rqi(test_folder_path)
        assert a or b or c or d #the useful information is in the captured stdout call

    def _chi_a_squ(self, result, expected):
        # chi_a^2 = 1/N \sum_i^N [(I^expected_i - I_calculated_i)/\sigam_i]^2
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

    def _chi_sq2(self, test_folder_path):
        result= self.get_result(test_folder_path)
        expected= self.get_expected_signal(test_folder_path)
        chi_a_sq= self._chi_a_squ(result, expected)
        test= chi_a_sq < expected.chi_square + expected.sigma_chi_square*2
        if not test:
            test_name = self.get_test_name(test_folder_path)
            print(test_name, " failed chi sq 2")
        return test

    def _chi_sq1(self, test_folder_path):
        result= self.get_result(test_folder_path)
        expected= self.get_expected_signal(test_folder_path)
        chi_a_sq= self._chi_a_squ(result, expected)
        test= chi_a_sq < expected.chi_square + expected.sigma_chi_square*1
        if not test:
            test_name = self.get_test_name(test_folder_path)
            print(test_name, " failed chi sq 1")
        return test

    def _test_points(self, test_folder_path):
        result= self.get_result(test_folder_path)
        expected= self.get_expected_signal(test_folder_path)
        failed_sig={
            1:[],
            2:[],
            3:[]
        }
        for i in range(len(expected.q)):
            for j in range(1,3):
                passed= self._check_sigma(result.y[i], expected.intensity[i], j * expected.sigma[i])
                a = struct.pack("<dd", result.y[i], expected.intensity[i])
                b = struct.unpack("<qq", a)
                test= b[1] - b[0]
                if not passed:
                    if abs(test) > 256:
                        failed_sig[j].append((result.y[i], expected.intensity[i], expected.sigma[i], expected.q[i]))

        percent_one= len(failed_sig[1])/len(expected.q)
        percent_two= len(failed_sig[2])/len(expected.q)
        percent_three= len(failed_sig[3])/len(expected.q)

        test= percent_one < .5 and  percent_two < .3 and percent_three < .1
        if not test:
            test_name = self.get_test_name(test_folder_path)
            print(test_name, "failed points test", percent_one, percent_three, percent_three)
        return test

    def _check_sigma(self, obs, exp, sig):
        lower_limit=exp-sig
        upper_limit=exp+sig
        test1 = (obs >= lower_limit)
        test2= (obs <= upper_limit)
        return test1 and test2

    def _test_normalized_Rqi(self, test_folder_path):
        result= self.get_result(test_folder_path)
        expected= self.get_expected_signal(test_folder_path)
        normalized=[]
        for i in range(len(expected.q)):
            R_q_i= result.y[i] - expected.intensity[i]
            Max_q_i= max(result.y[i], expected.intensity[i])
            if Max_q_i!=0:
                Normalized_R_q_i= abs(R_q_i/Max_q_i)
            else:
                Normalized_R_q_i = 0
            normalized.append(Normalized_R_q_i)

        max_n = max(normalized)
        mean_n = np.mean(normalized)
        test = mean_n < 1e-8 and  max_n < 1e-8
        if not test:
            test_name = self.get_test_name(test_folder_path)
            print(test_name, "failed Rqi test test", max_n, "\t", mean_n)



