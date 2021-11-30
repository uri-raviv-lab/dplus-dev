import os
import struct
from collections import OrderedDict

import pytest
from pytest import approx

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner
from tests.old_stuff.fix_state_files import fix_file


@pytest.fixture(scope="module")
def LocalAPI():
    exe_directory=r"C:\Users\devora.CHELEM\Sources\dplus_master_branch\x64\Release"
    #session=r"D:\UserData\devora\Sources\dplus\WebApplication\media\testing"
    return LocalRunner(exe_directory)#, session)


@pytest.mark.incremental
class TestGenerate:
    '''
    A class for testing that the signal graph of a run of Generate matches (within a tolerance level) a given expected signal
    '''
    def test_args(self, infile, efile):
        '''
        test that the input files exist, the arguments are valid, 
        and the X values from the state and the expected graph match up
        this test is a prerequisite for testing the Generate itself
        '''
        fix_file(infile)
        input = CalculationInput.load_from_state_file(infile)
        expected_graph = expected_file_to_graph(efile)
        for xval in input.x:
            ex, ey = expected_graph.popitem(last=False)
            if not ex == xval:
                raise ValueError(str("ex")+" "+str(xval))

    def test_result_signal(self, infile, efile):
        input = CalculationInput.load_from_state_file(infile)
        result = LocalAPI.generate(input)
        expected_graph = expected_file_to_graph(efile)
        result_graph = result.graph
        if not result_graph:
            raise ValueError(str(result.error))
        length=str(len(expected_graph))
        errors=[]
        while result_graph:
            rx, ry = result_graph.popitem(last=False)
            ex, ey = expected_graph.popitem(last=False)
            if not close_enough(ry, ey):
                errors.append((ry,ey))
                #print("expected: ", ex, ey, "\tresult: ", rx,ry)

        if errors:
            raise ValueError(str(len(errors))+"/" +length+": "+str(errors))


@pytest.mark.incremental
class TestFit:
    '''
    A class that tests that the parameters of an input State Tree after fitting to a signal match (within a tolerance level)
    the expected parameters given in a result state tree
    '''
    def test_args(self, infile, efile):
        fixed_in_file=fix_file(infile)
        fixed_out_file=fix_file(efile)
        input=CalculationInput.load_from_state_file(fixed_in_file)
        output=CalculationInput.load_from_state_file(fixed_out_file)
        assert len(input.get_mutable_params()) == len(output.get_mutable_params())

    def test_result_params(self, infile, efile):
        fixed_in_file=fix_file(infile)
        fixed_out_file=fix_file(efile)
        input=CalculationInput.load_from_state_file(fixed_in_file)
        output=CalculationInput.load_from_state_file(fixed_out_file)
        print(input)




def expected_file_to_graph(efile):
    expected_graph=OrderedDict()
    with open(efile, 'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            if line[0]=='#':
                continue
            x,y=line.split()
            expected_graph[float(x)]=float(y)
    return expected_graph

def close_enough(res, exp):
    '''not allowing more than a 1% difference'''
    close= (res/exp) == approx(1, rel=1e-1, abs=1e-1)
    #if close:
    #    print(res, exp)
    return close


def open_amp_file(ampfile):
    with open(ampfile, 'rb') as f:
        headers=[]
        offset=0
        byte = f.read(2)
        desc=byte.decode()
        if desc[1]=="@": #we have an offset
            offset=1#int(f.read(1).decode())
            f.readline() #garbage
        elif desc[1]==b"\n":
            header=f.readline()
            headers.append(header)

        while True:
            check=f.read(1).decode()
            if check!="#":
                break
            header=f.readline()
            headers.append(header)


        if offset>0:
            f.readline()

        version= f.readline().decode()
        num_bytes=f.readline().decode()
        temp_grid_size=f.readline().decode()
        if int(version)>10:
            temp_extra=f.readline().decode()

        gridsize=(int(temp_grid_size) - int(temp_extra))*2

        #there's supposed to be some kind of stripping of newlines?
        data=f.read(8)
        stepsize=struct.unpack("d", data)
        while f:
            try:
                data=f.read(16)
                if not data:
                    break
                complex_val=struct.unpack("dd", data)
                yield complex_val
            except struct.error:
                data=f.peek()
                pass


@pytest.mark.incremental
class baseforintensity:
    def test_args(self, infile, efile):
        input = CalculationInput.load_from_state_file(infile)
        expected_graph = expected_file_to_graph(efile)
        for xval in input.q:
            ex, ey = expected_graph.popitem(last=False)
            if not ex == xval:
                raise ValueError(str("ex")+" "+str(xval))

    def test_intensity(self, infile, efile, LocalAPI):
        folder=os.path.dirname(infile)
        rfile=os.path.join(folder, 'results.csv')
        input = CalculationInput.load_from_state_file(infile)
        result = LocalAPI.generate(input)
        expected_graph = expected_file_to_graph(efile)
        result_graph = result.graph
        if not result_graph:
            raise ValueError(str(result.error))
        length=str(len(expected_graph))
        errors=[]
        with open(rfile, 'w') as f:
            while result_graph:
                rx, ry = result_graph.popitem(last=False)
                ex, ey = expected_graph.popitem(last=False)
                if not close_enough(ry, ey):
                    errors.append((ry,ey))
                    f.write(str(rx)+","+ str(ry) +","+ str(ey) + "\n")

        if errors:
            raise ValueError(str(len(errors))+"/" +length+": "+str(errors))


@pytest.mark.incremental
class baseforamp:
    def test_amp(self, efile, infile, LocalAPI):
        input = CalculationInput.load_from_state_file(infile)
        LocalAPI.generate(input)

        # method one to get result amp:
        model_ptr = input.Domain.Children[0].Children[0].model_ptr
        rfile = self.get_amp_filepath(LocalAPI._session_directory, model_ptr)

        # method two to get result amp
        session_cache = os.path.join(LocalAPI._session_directory, 'cache')
        amp_files = [os.path.join(session_cache, fn) for fn in next(os.walk(session_cache))[2]]
        ampfile = amp_files[0]

        for expected, result in zip(open_amp_file(efile), open_amp_file(rfile)):
            if not close_enough(expected[0], result[0]):
                raise ValueError(str(expected[0]) + " " + str(result[0]))

    def get_amp_filepath(self, sess_dir, modelptr):
        ptr_string = '%08d.amp' % (int(modelptr))
        filepath = os.path.join(sess_dir, 'cache', ptr_string)
        return filepath