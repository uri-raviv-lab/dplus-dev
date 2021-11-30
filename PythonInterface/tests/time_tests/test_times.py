
import os
from tests.old_stuff.fix_state_files import fix_file
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner
from tests.test_settings import session, exe_directory
import numpy as np
import time
import math
from tests.time_tests.utils import TestsGenerateTimer
class TestGenerateTime():

    def test_time(self, iter, test_folder_path, exp_avg, exp_var, max_per_error):
        assert exp_avg >= 0 , "This doesn't have prev timer value"
        timer = TestsGenerateTimer(exe_directory, session)
        calc_avg, calc_var = timer.get_time(iter,timer.generate_func, test_folder_path)
        max_exp_val = exp_avg + math.sqrt(exp_var)
        test_exp = calc_avg < (max_exp_val + max_exp_val*0.01* max_per_error)
        if not test_exp:
            print("calc avg: {}, calc var: {}, expected avg:{}, expected var:{}".
                  format(calc_avg, calc_var, exp_avg, exp_var))
        assert test_exp



