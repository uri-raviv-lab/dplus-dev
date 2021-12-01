'''
Configuration for fitting tests. All fitting tests assume there is a gpu.
'''
import pytest
import os
from tests.test_settings import tests_dir

def get_files_dirs(no_gpu=True, slow=False):
    if no_gpu: #if there's no gpu, don't test fit
        return []

    generate_path=os.path.join(tests_dir, 'reviewer_tests', 'files_for_tests', 'fit')
    fast_gpu_files_dir=os.path.join(generate_path, 'gpu', 'short')
    slow_gpu_files_dir=os.path.join(generate_path, 'gpu', 'long')
    file_dirs = []
    if slow:
        file_dirs.append(slow_gpu_files_dir)
    else:
        file_dirs.append(fast_gpu_files_dir)
    return file_dirs

def pytest_generate_tests(metafunc):
    if "test_folder_path" in metafunc.fixturenames:
        result_params, result_ids= get_folder_tests(metafunc)
        metafunc.parametrize("test_folder_path", result_params, ids=result_ids)

def get_specific_tests():
    generate_path = os.path.join(tests_dir, 'reviewer_tests', 'files_for_tests', 'fit')
    fast_gpu_files_dir = os.path.join(generate_path, 'gpu', 'short')
    failed_tests = [
        "DPPC_slabs_fit",
    ]
    file_dirs = []
    for test in failed_tests:
        file_dirs.append(os.path.join(fast_gpu_files_dir, test))
    return file_dirs, failed_tests


def get_folder_tests(metafunc):
    slow = metafunc.config.option.slow
    no_gpu = metafunc.config.option.no_gpu
    file_dirs=get_files_dirs(no_gpu, slow)
    result_params = []
    result_ids = []
    for test_dir in file_dirs:
        for (dirpath, dirnames, filenames) in os.walk(test_dir):
            for folder in dirnames:
                result_params.append(os.path.join(dirpath, folder))
                result_ids.append(folder)
            break
    return result_params, result_ids
    #parametrize: argsnames, argvalues,  ids

