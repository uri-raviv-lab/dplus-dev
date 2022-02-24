import time
import numpy as np
import os
from tests.old_stuff.fix_state_files import fix_file
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner, WebRunner
import json
import shutil
import tempfile

class TestsGenerateTimer:

    def __init__(self, session):
        self.session = session

    def get_time(self, iterations, fun, *args):
        if iterations < 1:
            return -1, -1
        duration = np.zeros(iterations)
        for i in range(iterations):
            runtime = fun(*args)
            duration[i] = runtime
        return np.average(duration), np.var(duration)

    def get_input(self, test_path):
        test_name = os.path.basename(os.path.normpath(test_path))
        state_file = os.path.join(test_path, test_name + ".state")
        fixed_state_file = fix_file(state_file)
        input = CalculationInput.load_from_state_file(fixed_state_file)
        input.use_gpu = ("gpu" in test_path)
        return input

    def generate_func(self,test_folder_path):
        test_session = os.path.join(self.session, os.path.basename(test_folder_path))
        if os.path.exists(test_session):
            shutil.rmtree(test_session)
        calc_input = self.get_input(test_folder_path)
        api = LocalRunner(self.exe_directory, test_session)
        cache_folder = os.path.join(test_folder_path, "cache")
        # if there is cache folder in the test folder use it (timing with cache tests)
        if (os.path.exists(cache_folder)):
            dest_cache_path = os.path.join(api.session_directory, "cache")
            for amp_prm_file in os.listdir(cache_folder):
                shutil.copy(os.path.join(cache_folder,amp_prm_file), dest_cache_path)

        time_stime = time.time()
        result = api.generate(calc_input)
        time_etime = time.time()

        return time_etime - time_stime
