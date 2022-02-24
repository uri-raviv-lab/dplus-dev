import codecs
import json
import os

from dplus.CalculationInput import CalculationInput
from dplus.CalculationResult import FitResult, GenerateResult
from dplus.CalculationRunner import EmbeddedLocalRunner
from tests.old_stuff.fix_state_files import fix_file
from dplus.Signal import Signal
from tests.test_settings import session


class Expected:
    def __init__(self, outfilename, statefilename=None):
        self.q = []
        self.intensity = []
        self.sigma = []
        with open(outfilename, 'r') as file:
            for line in file:
                if len(line.strip()) < 1:
                    continue
                if line[0] == '#':
                    if "Average Chi square" in line:
                        self.chi_square = float(line.split()[-1])
                    if "Sigma Chi square" in line:
                        self.sigma_chi_square = float(line.split()[-1])
                elif line[0] == 'q':
                    continue
                else:
                    try:
                        q, i, s = line.split()
                    except ValueError:  # for curves I generated for testing Fit before they sent files
                        q, i = line.split()
                        s = 0
                    self.q.append(float(q))
                    self.intensity.append(float(i))
                    self.sigma.append(float(s))


class DplusProps:
    def get_test_name(self, test_path):
        test_name = os.path.basename(os.path.normpath(test_path))
        return test_name

    def get_session_folder(self, test_path, q_min=None):
        test_name = self.get_test_name(test_path)
        session_folder = os.path.join(session, test_name)
        if q_min:
            session_folder = os.path.join(session_folder, str(q_min))
        return session_folder

    def get_expected_signal(self, test_path, q_min=0):
        test_name = self.get_test_name(test_path)
        try:
            expected = Expected(os.path.join(test_path, test_name + "TestStandard.out"))
        except FileNotFoundError:
            expected = Expected(os.path.join(test_path, test_name + "TestStandard.dat"))
        expected = self.change_q_min_expected(q_min, expected)
        return expected

    def get_result(self, test_path, q_min=0, fit=False):
        """
        only valid once calculation has been run once (via test_run)
        a hackish way to circumvent running calculation multiple times
        """
        test_name = os.path.basename(os.path.normpath(test_path))

        session_folder = self.get_session_folder(test_path, q_min)
        filename = os.path.join(session_folder, "data.json")
        assert os.path.isfile(filename)
        with codecs.open(filename, 'r', encoding='utf8') as f:
            result = json.load(f)
        fixed_state_file = os.path.join(test_path, test_name + "_fixed.state")
        calc_data = CalculationInput.load_from_state_file(fixed_state_file)
        if fit:
            raise NotImplementedError()
            calc_result = FitResult(calc_data, result, LocalRunner.RunningJob(session_folder))
        else:
            calc_data = self.change_q_min_generate_input(q_min, calc_data)
            calc_result = GenerateResult(calc_data, result, job=None)
        return calc_result

    def get_expected_state(self, test_path):
        test_name = os.path.basename(os.path.normpath(test_path))
        state_file = os.path.join(test_path, test_name + "_Result.state")
        fixed_state_file = fix_file(state_file)
        input = CalculationInput.load_from_state_file(fixed_state_file)
        return input

    def change_q_min_generate_input(self, q_min, input):
        prev_q_min = input.DomainPreferences.q_min
        q_max = input.DomainPreferences.q_max
        try:
            if q_min < q_max:
                new_q_min = next(x[1] for x in enumerate(input.x) if x[1] >= q_min)
                input.DomainPreferences.q_min = new_q_min
                if input.DomainPreferences.q_min != prev_q_min:
                    # recalculate densicty of Q's
                    input.DomainPreferences.generated_points = int(
                        ((q_max - q_min) / (q_max - prev_q_min)) * input.DomainPreferences.generated_points)
        except:
            pass
        input.DomainPreferences.signal = Signal.create_x_vector(q_max, q_min, input.DomainPreferences.generated_points)
        return input

    def change_q_min_expected(self, q_min, expected):
        q_max = expected.q[-1]
        if q_min < q_max:
            index_q_min = 0
            if q_min < expected.q[-1]:
                for i in range(0, len(expected.q)):
                    if expected.q[i] < q_min:
                        index_q_min += 1
            expected.q = expected.q[index_q_min:]
            expected.intensity = expected.intensity[index_q_min:]
            expected.sigma = expected.sigma[index_q_min:]
        return expected
