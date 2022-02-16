import sys
import os
from dplus.CalculationRunner import LocalRunner
from dplus.CalculationInput import FitInput
from dplus.FitRunner import FitRunner

if __name__ == '__main__':
    args = sys.argv[1:]
    exe_directory=args[0]
    session_directory=args[1]
    print(args)

    runner= LocalRunner(exe_directory)

    filename=os.path.join(session_directory, "args.json")
    calc_input=FitInput._load_from_args_file(filename)

    print("created calc input")

    fitrunner=FitRunner(runner,session_directory)
    fitrunner.run(calc_input)

    print("finished fit")