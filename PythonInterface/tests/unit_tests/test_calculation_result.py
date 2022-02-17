import json
import pytest
import struct
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner
from tests.reviewer_tests.utils import Expected
from tests.unit_tests.conftest import test_dir
from dplus.DataModels.models import UniformHollowCylinder
import os
import tempfile
import shutil
import numpy as np
import math
pi=math.pi


class TestGenerateResolution():
    def test_resolution_function(self):
        state_file = os.path.join(test_dir, r"files_for_resolution_function\Stacked_Slabs.state")
        input = CalculationInput.load_from_state_file(state_file)
        runner = EmbeddedLocalRunner()
        input.DomainPreferences.apply_resolution = True
        input.DomainPreferences.resolution_sigma = 0.02
        result = runner.generate(input)
        assert len(result.graph) > 0

    def load_file_with_resolution(self, file_name):
        state_file = os.path.join(test_dir, file_name+ ".state")
        input = CalculationInput.load_from_state_file(state_file)
        runner = EmbeddedLocalRunner()

        result = runner.generate(input)
        expected = Expected(os.path.join(test_dir, file_name + "TestStandard.out"))

        d = self._test_normalized_Rqi(result, expected)
        assert d  # the useful information is in the captured stdout call

    def test_some_sigma_value(self):
        file_name1 = r"files_for_resolution_function\Stacked_Slabs_s_1"
        self.load_file_with_resolution(file_name1)
        file_name01 = r"files_for_resolution_function\Stacked_Slabs_s_01"
        self.load_file_with_resolution(file_name01)
        file_name02 = r"files_for_resolution_function\Stacked_Slabs_s_02"
        self.load_file_with_resolution(file_name02)

    def _test_normalized_Rqi(self,result, expected):
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
        print("max_n, mean_n:", max_n, mean_n)
        test = mean_n < 1e-2 and max_n < 1e-2
        # if mean_n > 1e-5 or max_n > 1e-5:
        #     test_name = self.get_test_name(test_folder_path)
        #     print(test_name, "failed Rqi test", max_n, "\t", mean_n)
        return test

    def test_load_and_save(self):

        uhc = UniformHollowCylinder()
        input = CalculationInput()
        input.Domain.populations[0].add_model(uhc)
        input.DomainPreferences.apply_resolution = True
        input.DomainPreferences.resolution_sigma = 0.4

        tmp_directory = tempfile.mkdtemp()
        new_file_path = os.path.join(tmp_directory, 'test.state')
        input.export_all_parameters(new_file_path)

        calc_input = CalculationInput.load_from_state_file(new_file_path)
        shutil.rmtree(tmp_directory)

        assert calc_input.DomainPreferences.apply_resolution is True
        assert calc_input.DomainPreferences.resolution_sigma == 0.4

