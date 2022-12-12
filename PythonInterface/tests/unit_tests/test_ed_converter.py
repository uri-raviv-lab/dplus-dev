import pytest
from dplus.EDConverter import convert, ResultEDConverter


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

    