
import numpy as np
import time
import argparse
import os
from tests.test_settings import session, exe_directory, tests_dir
from tests.time_tests.utils import TestsGenerateTimer
parser = argparse.ArgumentParser(description='create timer configuration file')
parser.add_argument("--slow", action="store_true", default=False,
                 help="run slow tests")
parser.add_argument("--no_gpu", action="store_true", default=False,
                 help="don't run GPU tests")
parser.add_argument("--no_cpu", action="store_true", default=False,
                 help="don't run CPU tests")
parser.add_argument('--exe_dir', type=str, default=exe_directory,
                    help='exe directory ')
parser.add_argument('--session_dir', type=str, default=session,
                    help='sessions directory')
parser.add_argument('--iter', default=10, type=int,
                    help='number of iteration for each generate operation')


args = parser.parse_args()


def get_files(no_cpu, no_gpu, slow):
    #note that the slow directories INCLUDE all the files from the fast directories as well.
    generate_path=os.path.join(tests_dir, 'reviewer_tests', 'files_for_tests', 'generate')
    fast_gpu_files_dir=os.path.join(generate_path, 'gpu', 'short')
    fast_cpu_files_dir = os.path.join(generate_path, 'cpu', 'short')
    slow_gpu_files_dir=os.path.join(generate_path, 'gpu', 'long')
    slow_cpu_files_dir = os.path.join(generate_path, 'cpu', 'long')
    files_with_cache = os.path.join(tests_dir, 'time_tests', 'files_for_tests_with_cache', 'gpu')
    file_dirs = []

    #file_dirs.append(files_with_cache)
    if slow:
        if not no_cpu:
            file_dirs.append(slow_cpu_files_dir)
        if not no_gpu:
            file_dirs.append(slow_gpu_files_dir)
    else:
        if not no_cpu:
            file_dirs.append(fast_cpu_files_dir)
        if not no_gpu:
            file_dirs.append(fast_gpu_files_dir)
    files = []
    for test_dir in file_dirs:
        for (dirpath, dirnames, filenames) in os.walk(test_dir):
            for folder in dirnames:
                files.append(os.path.join(dirpath, folder))
            break
    return files

run_dirs = get_files(args.no_cpu, args.no_gpu, args.slow)

exe_dir = args.exe_dir
session_dir = args.session_dir
outfile = os.path.join(os.path.dirname(os.path.realpath(__file__)   ), "timer_configfile.txt")
timer = TestsGenerateTimer(exe_dir, session_dir)
with open(outfile, "w") as f:
    for item in run_dirs:
        calc_avg, calc_var = timer.get_time(args.iter, timer.generate_func, item)
        data = "{} {} {}".format(item, str(calc_avg), str(calc_var))
        print(data)
        f.write(data+"\n")


