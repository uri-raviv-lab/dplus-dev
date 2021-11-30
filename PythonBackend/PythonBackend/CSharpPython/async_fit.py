import argparse
# from dplus.Fit import CurveFitter
from dplus.CalculationRunner import LocalRunner
from dplus.CalculationInput import CalculationInput
from dplus.PyCeresOptimizer import PyCeresOptimizer

import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run python async fit')
    parser.add_argument("exe_dir", type=str, help="dplus executable dir")
    parser.add_argument('session_dir', type=str, help='current program session dir')
    args = parser.parse_args()
    fit_calc_runner = LocalRunner(args.exe_dir, args.session_dir)
    calc_input = CalculationInput._load_from_args_file(os.path.join(args.session_dir, "args.json"))
    PyCeresOptimizer.fit(calc_input, fit_calc_runner)