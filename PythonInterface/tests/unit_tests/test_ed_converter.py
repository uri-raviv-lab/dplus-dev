import pytest
import os
from dplus.EDConverter import convert, ResultEDConverter

test_dir = os.path.join(os.path.dirname(__file__), "files_for_tests")
def test_1():
    result: ResultEDConverter = convert()
    # print(result)
    assert result
    assert 30.799705 == pytest.approx(result.eED, 1e-2)
    assert 10.811791 == pytest.approx(result.coeff, 1e-2)


def test_2():
    try:
        result: ResultEDConverter = convert(ed=0.0, pdb='', n=0, a=[])
        assert False # should raise an error before
    except:
        assert True

def test_3():
    pdb_path = os.path.join(test_dir, "CF4.pdb")

    result = convert(pdb=pdb_path)
    assert result
    assert 30.799705 != pytest.approx(result.eED, 1e-2)
    assert 10.811791 != pytest.approx(result.coeff, 1e-2)
