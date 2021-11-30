#unit test CalculationInput.py
import os
from json import JSONDecodeError
import pytest
from dplus.CalculationInput import CalculationInput
from tests.unit_tests.conftest import test_dir

class TestLoadGenerateInput:
    #TODO: update filenames appropriately
    def test_load_from_OK_state(self):
        filename=os.path.join(test_dir, "sphere.state")
        data=CalculationInput.load_from_state_file(filename)
        assert isinstance(data, CalculationInput)

    def test_load_from_state_not_exist(self):
        with pytest.raises(FileNotFoundError):
            filename = os.path.join(test_dir, "XX.state")
            data = CalculationInput.load_from_state_file(filename)

    def test_load_from_notJSON(self):
        with pytest.raises(JSONDecodeError):
            filename = os.path.join(test_dir, "sphere.out")
            data = CalculationInput.load_from_state_file(filename)

    def test_load_from_OK_pdb(self):
        filename=os.path.join(test_dir, "CF4.pdb")
        data=CalculationInput.load_from_PDB(filename, 25)
        assert isinstance(data, CalculationInput)

    def test_load_from_pdb_not_exist(self):
        with pytest.raises(FileNotFoundError):
            filename =os.path.join(test_dir, "XX.pdb")
            data = CalculationInput.load_from_PDB(filename, 25)

    def test_load_from_not_pdb(self):
        with pytest.raises(NameError):
            filename = os.path.join(test_dir, "sphere.state")
            data = CalculationInput.load_from_PDB(filename, 25)