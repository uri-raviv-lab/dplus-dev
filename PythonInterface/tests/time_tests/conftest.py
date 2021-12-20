import pytest
import os
from tests.test_settings import tests_dir

timer_configuration_file=os.path.join(os.path.dirname(os.path.realpath(__file__)   ), "timer_configfile.txt")


def get_files_dirs(no_cpu, no_gpu, slow):
    #note that the slow directories INCLUDE all the files from the fast directories as well.
    generate_path=os.path.join(tests_dir, 'reviewer_tests', 'files_for_tests', 'generate')
    fast_gpu_files_dir=os.path.join(generate_path, 'gpu', 'short')
    fast_cpu_files_dir = os.path.join(generate_path, 'cpu', 'short')
    slow_gpu_files_dir=os.path.join(generate_path, 'gpu', 'long')
    slow_cpu_files_dir = os.path.join(generate_path, 'cpu', 'long')
    files_with_cache = os.path.join(tests_dir, 'time_tests', 'files_for_tests_with_cache','gpu')
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
    return file_dirs

def pytest_generate_tests(metafunc):
    num_iter = 1
    if "test_folder_path" in metafunc.fixturenames:
        if metafunc.config.option.specific:
            result_params, result_ids = folder_test(metafunc)
        else:
            result_params, result_ids = standard_tests(metafunc)
            #result_params, result_ids=specific_tests()
        if metafunc.config.option.percent:
            max_per_error = int(metafunc.config.option.percent)
        else:
            max_per_error = 4
        with open(timer_configuration_file, "r") as file:
            all_lines = file.readlines()
        avg_var_list = [ line.split() for line in all_lines]
        updated_result_params = []
        for result in result_params:
            is_exists = False
            for item in avg_var_list:
                if result == item[0]: # same test
                    updated_result_params.append([num_iter, result, float(item[1]), float(item[2]), max_per_error])
                    is_exists = True
                    break
            if not is_exists:
                updated_result_params.append([num_iter, result, -1, -1, max_per_error])

        metafunc.parametrize("iter, test_folder_path, exp_avg, exp_var, max_per_error", updated_result_params, ids=result_ids)

def specific_tests():
    generate_path=os.path.join(tests_dir, 'reviewer_tests', 'files_for_tests', 'generate')
    fast_gpu_files_dir=os.path.join(generate_path, 'cpu', 'fast')
    failed_tests=[
        "Single_PDB_Impl_No_Charges_With_Hydr_Ga_Kr",
		"Single_PDB_Impl_No_Charges_With_Hydr_VEGAS",
		"Single_PDB_Impl_No_Charges_With_Hydr_MC"
    ]
    file_dirs=[]
    for test in failed_tests:
        file_dirs.append(os.path.join(fast_gpu_files_dir, test))

    return file_dirs, failed_tests

def folder_test(metafunc):
    path = metafunc.config.option.specific
    file_dirs = [path]
    test_name = [os.path.basename(path)]
    return file_dirs, test_name

def standard_tests(metafunc):
    slow = metafunc.config.option.slow
    no_gpu = metafunc.config.option.no_gpu
    no_cpu = metafunc.config.option.no_cpu
    result_params = []
    result_ids=[]
    file_dirs=get_files_dirs(no_cpu, no_gpu, slow)
    for test_dir in file_dirs:
        for (dirpath, dirnames, filenames) in os.walk(test_dir):
            for folder in dirnames:
                result_params.append(os.path.join(dirpath, folder))
                result_ids.append(folder)
            break
        #parametrize: argsnames, argvalues,  ids
    return result_params, result_ids
