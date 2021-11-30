import pytest
import os
from tests.test_settings import test_files_dir


def pytest_generate_tests(metafunc):
    if "test_folder_path" in metafunc.fixturenames:
        if metafunc.config.option.specific:
            result_params, result_ids = folder_test(metafunc)
        else:
            result_params, result_ids = standard_tests(metafunc)
            #result_params, result_ids=specific_tests()
        metafunc.parametrize("test_folder_path", result_params, ids=result_ids)

def get_files_dirs(no_cpu, no_gpu):
    #note that the slow directories INCLUDE all the files from the fast directories as well.
    cpu_files_dir=os.path.join(test_files_dir, 'reviewer_tests', 'files_for_tests', 'qmin', 'cpu')
    gpu_files_dir = os.path.join(test_files_dir, 'reviewer_tests', 'files_for_tests', 'qmin', 'gpu')
    #TODO: finish writing this
    file_dirs = []
    if not no_cpu:
        file_dirs.append(cpu_files_dir)
    if not no_gpu:
        file_dirs.append(gpu_files_dir)

    return file_dirs


def folder_test(metafunc):
    path = metafunc.config.option.specific
    file_dirs = [path]
    test_name = [os.path.basename(path)]
    return file_dirs, test_name

def standard_tests(metafunc):
    no_gpu = metafunc.config.option.no_gpu
    no_cpu = metafunc.config.option.no_cpu
    result_params = []
    result_ids=[]
    file_dirs=get_files_dirs(no_cpu, no_gpu)
    for test_dir in file_dirs:
        for (dirpath, dirnames, filenames) in os.walk(test_dir):
            for folder in dirnames:
                result_params.append(os.path.join(dirpath, folder))
                result_ids.append(folder)
            break
        #parametrize: argsnames, argvalues,  ids
    return result_params, result_ids
