#  tests the functions called from CSharpPythonEntry
import os
from dplus.CalculationInput import CalculationInput

from tests.test_settings import exe_directory, session_dir, python_dir, tests_dir
test_dir=os.path.join(tests_dir, "CSharpPython_tests", "files")


def test_init():
    from CSharpPythonEntry import EmbeddedCSharpPython

    filename = os.path.join(test_dir, "sphere.state")
    calc_input = CalculationInput.load_from_state_file(filename)
    runner = EmbeddedCSharpPython() # exe_directory,session_dir,python_dir,calc_input
    runner.start_fit(calc_input)