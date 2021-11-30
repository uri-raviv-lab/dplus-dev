import os
# from dplus.PyCeresOptimizer import PyCeresOptimizer
from dplus.CalculationInput import CalculationInput
from tests.unit_tests.conftest import test_dir

# def test_fit_1():
#     filename = os.path.join(test_dir, "sphere.state")
#     calc_input = CalculationInput.load_from_state_file(filename)
#     calc_input.signal = calc_input.signal.read_from_file(os.path.join(test_dir, "sphere.out"))
#     result= PyCeresOptimizer.fit(calc_input)
#     hi=1