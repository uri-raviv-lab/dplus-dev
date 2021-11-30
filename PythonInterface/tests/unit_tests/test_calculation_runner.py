
# import pytest
# from dplus.CalculationInput import CalculationInput
# from dplus.CalculationRunner import LocalRunner
# from dplus.CalculationInput import *
# import os
# from tests.test_settings import exe_directory, session
# from tests.unit_tests.conftest import test_dir
#
# class TestCalculationRunner:
#
#     # removing this test because we change the behavior (now there is a warning and not error)
#     def test_no_results(self):
#         with pytest.raises(Exception, match="When using a GPU for computations, the grid must be enabled. Make sure either the Use Grid checkbox is checked, or Use GPU is unchecked."):
#             from dplus.DataModels.models import Sphere
#             input = CalculationInput()
#             s = Sphere()
#             s.use_grid = True
#             s.layer_params[1]["Radius"].value = 2
#             s.layer_params[1]["Radius"].mutable = True
#             input.Domain.populations[0].add_model(s)
#             signal_file=os.path.join(test_dir, "sphere.out")
#             input.DomainPreferences.signal_file = signal_file
#             input.DomainPreferences.grid_size = 200
#             input.DomainPreferences.orientation_iterations = 10000
#             input.use_gpu = True
#             input.DomainPreferences.use_grid = False
#             runner = LocalRunner(exe_directory, session)
#             result = runner.fit(input)
