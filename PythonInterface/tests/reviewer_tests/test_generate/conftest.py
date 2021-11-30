import pytest
import os
from tests.test_settings import test_files_dir





def get_files_dirs(no_cpu, no_gpu, slow):
    #note that the slow directories INCLUDE all the files from the fast directories as well.
    generate_path=os.path.join(test_files_dir, 'reviewer_tests', 'files_for_tests', 'generate')
    fast_gpu_files_dir=os.path.join(generate_path, 'gpu', 'short')
    fast_cpu_files_dir = os.path.join(generate_path, 'cpu', 'short')
    slow_gpu_files_dir=os.path.join(generate_path, 'gpu', 'long')
    slow_cpu_files_dir = os.path.join(generate_path, 'cpu', 'long')
    file_dirs = []
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
    if "test_folder_path" in metafunc.fixturenames:
        if metafunc.config.option.specific:
            result_params, result_ids = folder_test(metafunc)
        else:
            result_params, result_ids = standard_tests(metafunc)
            #result_params, result_ids=specific_tests()
        metafunc.parametrize("test_folder_path", result_params, ids=result_ids)

def specific_tests():
    generate_path=os.path.join(test_files_dir, 'reviewer_tests', 'files_for_tests', 'generate')
    fast_gpu_files_dir=os.path.join(generate_path, 'gpu', 'short')
    failed_tests=[
        "Man_Symm_Impl_No_Charges_With_Hydr_Ga_Kr_Amp",
        "Man_Symm_Impl_No_Charges_With_Hydr_MC_Amp",
        "Man_Symm_Impl_No_Charges_wo_Hydr_Ga_Kr_Amp",
        "Man_Symm_Impl_No_Charges_wo_Hydr_MC_Amp",
        "Scripted_Symm_Impl_No_Charges_With_Hydr_Ga_Kr_Amp",
        "Scripted_Symm_Impl_No_Charges_With_Hydr_MC_Amp",
        "Scripted_Symm_Impl_No_Charges_wo_Hydr_Ga_Kr_Amp",
        "Scripted_Symm_Impl_No_Charges_wo_Hydr_MC_Amp",
        "Space_fill_Symm_Impl_No_Charges_With_Hydr_Ga_Kr_Amp",
        "Space_fill_Symm_Impl_No_Charges_With_Hydr_MC_Amp",
        "Space_fill_Symm_Impl_No_Charges_wo_Hydr_Ga_Kr_Amp",
        "Space_fill_Symm_Impl_No_Charges_wo_Hydr_MC_Amp",
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
