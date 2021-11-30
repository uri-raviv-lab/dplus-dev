import os
import math
import tempfile
from os.path import abspath
import shutil
import numpy as np
import pytest

from tests.test_settings import exe_directory


root_path=os.path.dirname(abspath(__file__))

def test_conversion():
    from dplus.Amplitudes import sph2cart, cart2sph

    q, theta, phi = cart2sph(1, 2, 3)
    x, y, z = sph2cart(q, theta, phi)

def test_grid():
    from dplus.Amplitudes import Grid

    g = Grid(100, 5)
    for ind, (q, theta, phi) in g.create_grid():
        g.index_from_angles(q, theta, phi)


def test_amplitude_load():
    from dplus.Amplitudes import Amplitude
    amp_file = os.path.join(root_path, "files", "myamp.amp")
    my_amp = Amplitude.load(amp_file)
    for c in my_amp.complex_amplitude_array:
        pass
        #print(c)

def test_amplitude_headers():
    from dplus.Amplitudes import Amplitude
    from dplus.CalculationInput import CalculationInput
    from dplus.CalculationRunner import LocalRunner

    def my_func(q, theta, phi):
        return np.complex64(q+1 + 0.0j)

    a = Amplitude(80, 7.5)
    a.description= "An example amplitude"
    a.fill(my_func)
    a.save(os.path.join(root_path, "files", "myamp2.ampj"))

    input = CalculationInput()
    amp_model = input.add_amplitude(a)
    amp_model.centered = True
    runner = LocalRunner(exe_directory)
    result = runner.generate(input)
    assert len(result.y)

def test_amplitude_interpolation():
    from dplus.Amplitudes import Amplitude

    def my_func(q, theta, phi):
        return np.complex64(q+1 + 0.0j)

    a = Amplitude(80, 7.5)
    a.fill(my_func)

    output_intrp = a.get_interpolation(5, 3, 6)
    expected = my_func(5, 3, 6)
    assert (output_intrp.real, output_intrp.imag) == pytest.approx((expected.real, expected.imag), abs=1e-2)

    output_intrp_arr = a.get_interpolation([1, 2, 3], 3, 6)


class UniformSphere:
    def __init__(self):
        self.extraParams=[1,0]
        self.ED=[333, 400]
        self.r=[0,1]

    @property
    def nLayers(self):
        return len(self.ED)

    def calculate(self, vq, vtheta, vphi):
        cos = math.cos
        sin = math.sin
        nLayers = self.nLayers
        ED = self.ED
        extraParams = self.extraParams
        r = self.r
        def closeToZero(x):
            return (math.fabs(x) < 100.0 * 2.2204460492503131E-16)

        q = math.sqrt(math.pow(vq,2) + math.pow(vtheta,2) + math.pow(vphi,2))
        if closeToZero(q):
            electrons = 0.0
            for i in range( 1, nLayers):
                electrons += (ED[i] - ED[0]) * (4.0 / 3.0) * math.pi * (r[i] ** 3 - r[i-1] ** 3)
            return np.complex64(electrons  * extraParams[0] + extraParams[1]+ 0.0j)

        res = 0.0

        for i in range(nLayers-1):
            res -= (ED[i] - ED[i + 1]) * (cos(q * r[i]) * q * r[i] - sin(q * r[i]))
        res -= (ED[nLayers - 1] - ED[0]) * (cos(q * r[nLayers - 1]) * q * r[nLayers - 1] - sin(q * r[nLayers - 1]))

        res *= 4.0 * math.pi / (q*q * q)

        res *= extraParams[0] #Multiply by scale
        res += extraParams[1] #Add background
        return np.complex64(res + 0.0j)


def test_dplus_models_sphere():
    from dplus.Amplitudes import Amplitude
    from dplus.State import State
    from dplus.CalculationRunner import LocalRunner
    from dplus.CalculationInput import CalculationInput

    sphere = UniformSphere()
    a = Amplitude(50, 5)
    a.fill(sphere.calculate)
    tmp_directory = tempfile.mkdtemp()
    new_file_path = os.path.join(tmp_directory, 'sphere.ampj')
    try:
        a.save(new_file_path)
        input = CalculationInput()
        amp_model = input.add_amplitude(a)
        amp_model.centered = True
        runner = LocalRunner(exe_directory)
        result = runner.generate(input)
    finally:
        shutil.rmtree(tmp_directory)

class SymmetricSlab:
    def __init__(self):
        self.scale=1
        self.background=0
        self.xDomain=10
        self.yDomain=10
        self.ED=[333, 280]
        self.width=[0,1]
        self.OrganizeParameters()

    @property
    def nLayers(self):
        return len(self.ED)

    def OrganizeParameters(self):
        self.width[0] = 0.0
        self.xDomain *= 0.5
        self.yDomain *= 0.5
        for i in range(2, self.nLayers):
            self.width[i] += self.width[i - 1];

    def calculate(self, q, theta, phi):
        def closeToZero(x):
            return (math.fabs(x) < 100.0 * 2.2204460492503131E-16)
        from dplus.Amplitudes import sph2cart
        from math import sin, cos
        from numpy import sinc
        import numpy as np
        qx, qy, qz = sph2cart(q, theta, phi)
        res= np.complex128(0+0j)
        if(closeToZero(qz)):
            for i in range(self.nLayers):
                res += (self.ED[i] - self.ED[0]) * 2. * (self.width[i] - self.width[i - 1])
            return res * 4. * sinc(qx * self.xDomain) * self.xDomain * sinc(qy * self.yDomain) * self.yDomain

        prevSin = np.float64(0.0)
        currSin=np.float64(0.0)
        for i in range(1, self.nLayers):
            currSin = sin(self.width[i] * qz)
            res += (self.ED[i] - self.ED[0]) * 2. * (currSin - prevSin) / qz
            prevSin = currSin
        res *= 4. * sinc((qx * self.xDomain)/np.pi) * self.xDomain * sinc((qy * self.yDomain)/np.pi) * self.yDomain
        return res * self.scale + self.background #Multiply by scale and add background


def test_dplus_models_slab():
    from dplus.Amplitudes import Amplitude
    from dplus.State import State
    from dplus.CalculationRunner import LocalRunner
    from dplus.CalculationInput import CalculationInput
    symSlab = SymmetricSlab()
    a = Amplitude(80, 7.5)
    a.fill(symSlab.calculate)
    tmp_directory = tempfile.mkdtemp()
    new_file_path  = os.path.join(tmp_directory, 'slab.ampj')
    try:
        a.save(new_file_path)
        input = CalculationInput()
        amp_model = input.add_amplitude(a)
        amp_model.centered = True
        runner = LocalRunner(exe_directory)
        result = runner.generate(input)
    finally:
        shutil.rmtree(tmp_directory)



