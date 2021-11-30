import pytest
print("added the option as necessary")
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
        help="run slow tests")
    parser.addoption("--no_gpu", action="store_true", default=False,
                     help="don't run GPU tests")
    parser.addoption("--no_cpu", action="store_true", default=False,
                     help="don't run CPU tests")
    parser.addoption("--specific", action="store", default=None,
                        help="path of single test folder")
    parser.addoption("--percent", dest="percent",
                      help="percent of error - time tests")