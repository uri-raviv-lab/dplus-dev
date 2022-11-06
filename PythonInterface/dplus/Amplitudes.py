import io
import json
import math
import numpy as np
import os
import zipfile
import pathlib


try:
    from dplus.wrappers import CJacobianSphereGrid
except:
    print("could not import from CythonGrid")    
from math import pi

PI = 3.14159265358979323846  # math.pi
encoding = 'ascii'
npDouble = np.float64


def sph2cart(r, theta, phi):
    return [
        r * math.sin(theta) * math.cos(phi),
        r * math.sin(theta) * math.sin(phi),
        r * math.cos(theta)
    ]


def cart2sph(x, y, z):
    # note that a faster vectorized version of this can be found at:
    # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion/4116803#4116803
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)
    theta = math.atan2(math.sqrt(XsqPlusYsq), z)
    phi = math.atan2(y, x)
    if math.fabs(theta) < 1e-10:
        theta = 0
    if math.fabs(phi) < 1e-10:
        phi = 0
    if theta < 0:
        theta += pi
    if phi < 0:
        phi += 2*pi
    print(r, theta,phi)
    return r, theta, phi


class Grid:
    '''
    This class is described in pages 223-224 of the paper (Ginsburg et al., 2019)

    The class Grid is initialized with `q_max` and `grid_size`.
    It is used to create/describe a grid of `q`, `theta`, `phi` angle values.
    These values can be described using two sets of indexing:

    1. The overall index `m`
    2. The individual angle indices `i`, `j`, `k`
    '''

    def __init__(self, grid_size, q_max, q_min=0):
        if grid_size % 2:
            raise ValueError("Grid size must be even")
        self.q_max = q_max
        self.q_min=q_min
        self.grid_size = grid_size
        self.extra_shells = 3

    @property
    def step_size(self):
        '''
        The difference between q's in the grid.

        :return: double q_max/N (in the paper step_size is called eta_q)
        '''
        return npDouble(self.q_max / (self.N))

    @property
    def N(self):
        return int(self.grid_size / 2)

    @property
    def actual_size(self):
        return self.N + self.extra_shells

    @property
    def totalsz(self):
        i=self.actual_size
        return (6 * i * (i + 1) * (3 + 3 + 2 * 3 * i)) / 6 #+1 for origin?

    def _G_i_q(self, i):
        '''based on eq 14 in paper'''
        return 6 * i + 12 * i ** 2 + 6 * i ** 3

    def create_grid(self):
        '''
        a generator that returns q, theta, phi angles in phi-major order
        '''
        for i in range(0, self.N + 4):
            if i == 0:
                yield 0, (0, 0, 0)
                continue
            J_i = (3 * i) + 1
            K_ij = 6 * i
            for j in range(0, J_i):
                for k in range(0, K_ij):
                    yield self.index_from_indices(i,j,k), self.angles_from_indices(i, j, k)

    def angles_from_indices(self, i, j, k):
        '''
         receives angle indices i,j,k and returns their q, theta, and phi angle values.

        :param i: angle indice i
        :param j: angle indice j
        :param k: angle indice k
        :return: q, theta, and phi angle values
        '''
        if i == 0:
            return 0, 0, 0

        q = npDouble(i * self.step_size)
        theta_ij = npDouble((j * PI) / (3 * i))
        phi_ijk = npDouble((k * PI) / (3 * i))
        return q, theta_ij, phi_ijk

    def indices_from_index(self, m):
        '''

        :param m: receives an overall index m.
        :return: individual q, theta, and phi indices: i, j, k
        '''
        if m == 0:
            return 0, 0, 0
        i = math.floor((m / 6) ** (1. / 3.))
        if m > self._G_i_q(i):
            i += 1
        R_i = m - self._G_i_q(i - 1) - 1
        j = math.floor(R_i / (6 * i))
        k = R_i - 6 * i * j
        return i, j, k

    def angles_from_index(self, m):
        '''

        :param m: receives an overall index m
        :return: returns the matching q, theta, and phi angle values
        '''
        i, j, k = self.indices_from_index(m)
        q, theta, phi = self.angles_from_indices(i, j, k)
        return q, theta, phi

    def index_from_indices(self, i, j, k):
        '''
        receives angle indices i,j,k and returns the overall index m that matches them.

        :param i: angle indices i
        :param j: angle indices j
        :param k: angle indices k
        :return: overall index m that matches the given i,j,k
        '''
        if i == 0:
            return 0
        return 6 * (i - 1) + 12 * (i - 1) ** 2 + 6 * (i - 1) ** 3 + 6 * i * j + k + 1

    def indices_from_angles(self, q, theta, phi):
        '''
        receives angles q, theta, phi, ands returns the matching indices i,j,k.

        :param q: q angle
        :param theta: theta angle
        :param phi: phi angle
        :return: return indices i, j, k of given q, thta and phi
        '''

        eps = 0.000001
        i = math.floor(q / self.step_size + eps)

        j,k=self._jk_indices_from_angles_and_index(i, theta, phi)

        return i, j, k

    def _jk_indices_from_angles_and_index(self, i, theta, phi):
        eps = 0.000001

        phiPoints = 6.0 * i
        thePoints = 3.0 * i

        j = math.floor((theta / PI) * thePoints + eps)
        k = math.floor((phi / (PI * 2)) * phiPoints + eps)

        return j, k

    def index_from_angles(self, q, theta, phi):
        '''
        receives angles q, theta, phi and returns the matching overall index m.

        :param q: q angle
        :param theta: theta angle
        :param phi: phi angle
        :return: matching overall index m
        '''
        i, j, k = self.indices_from_angles(q, theta, phi)
        return self.index_from_indices(i, j, k)


class Amplitude():
    '''
    The class `Amplitude`, by contrast, can be used to build an amplitude and then save that amplitude as an amplitude file,
    which can then be opened in D+ (or sent in a class AMP) but it itself cannot be added directly to the Domain parameter tree.
    '''

    def __init__(self, grid_size, q_max, q_min=0):
        self.grid = CJacobianSphereGrid(grid_size, q_max)
        self.helper_grid=Grid(grid_size, q_max, q_min)
        self.external_headers = None
        self.__description = ""
        self.filename = ""
        self.__initialized_splines=False

    @property
    def _values(self):
        '''
        array that contains the grid intensity values as 2 values - real and imaginary

        :return: values array
        '''
        values_array = self.grid.get_data()
        if len(values_array):
            return values_array
        else:
            raise ValueError("Amplitude values empty-- has not been initialized yet")

    @_values.setter
    def _values(self, tmp_values):
        data = self._values
        if len(data) != len(tmp_values):
            raise ValueError("exists grid len doesn't fit to new grid len")
        for i in range(len(tmp_values)):
            data[i] = tmp_values[i]

    @property
    def complex_amplitude_array(self):
        '''
        returns the values array as complex array.

        :return: complex array
        '''
        values = self._values
        complex_arr = np.zeros((int(values.__len__() / 2), 1), dtype=np.complex)
        for index in range(0, complex_arr.__len__()):
            complex_arr[index] = values[2 * index] + 1j * values[2 * index + 1]
        return complex_arr

    @property
    def _default_header(self):
        '''
        Return the default file headers values for amplidute class.

        :return: file headers of amplitude
        '''
        descriptor = "#@".encode(encoding)
        header = "# created from a Python function\n"
        header += "# " + "\\" * 80 + "\n"
        header += "# User description:" + self.description + "\n"
        header += "# " + "\\" * 80 + "\n"
        header += "# Grid was used.\n"
        header += "# N^3; N = " + str(self.grid.grid_size) + "\n"
        header += "# qMax= " + str(self.grid.q_max) + "\n"
        header += "# Grid step size = " + str(self.grid.step_size) + "\n"
        header += "\n"

        headlen = np.uint32(2 * 1 + 4 + 1 + len(header) * 1 + 1)  # descriptor + unsigned int +  \n + header length + \n
        header_list = [descriptor, headlen, "\n".encode(encoding), header.encode(encoding), "\n".encode(encoding)]
        step_size = np.array([self.grid.step_size], dtype=np.float64)
        added_list = [
            (str(13) + "\n").encode(encoding),  # version
            (str(16) + "\n").encode(encoding),  # size of double
            (str(int(self.grid.actual_size)) + "\n").encode(encoding),  # "tmp grid size"
            (str(int(self.grid.extras)) + "\n").encode(encoding),
            step_size.tobytes()  # note that this does not get new line
        ]
        return header_list + added_list

    @property
    def headers(self):
        '''
        Returns the headers - default if amplitude was created nt python API or external if amplitude was created from a file.
        :return:
        '''
        if self.external_headers:
            return self.external_headers
        else:
            return self._default_header

    @property
    def description(self):
        if self.__description:
            return self.__description
        return "None"

    @description.setter
    def description(self, val):
        edit = val.replace("\n", "\n# ")
        self.__description = "\n# " + edit

    def old_save(self, filename):
        '''
         The function will save the information of the Amplitude class to an Amplitude file which can then be
         passed along to D+ to calculate its signal or perform fitting.

        :param filename: new amplitude file name
        '''
        with open(filename, 'wb') as f:
            for header in self.headers:
                f.write(header)
            amps = np.float64(self._values)
            amps.tofile(f)
        self.filename = os.path.abspath(filename)

    def save(self, filename):
        extension = pathlib.Path(filename).suffix
        if extension == ".amp":
            raise ValueError("Please save amplitudes in .ampj format, not .amp")
        ampzip = zipfile.ZipFile(filename, mode='w')
        ampzip.writestr('grid.dat', self._values.tobytes())
        info = self.grid.get_param_json_string()
        ampzip.writestr("criticalinfo.json", info)

        new_header = {"Title": self.description, "qMax": self.grid.q_max, "StepSize": self.grid.step_size,
                      "N": self.grid.grid_size, "Used Grid": True}
        header = json.dumps(
            {"old header": str(self.headers), "Header": new_header}
        )

        ampzip.writestr("header.json", header)

        ampzip.close()
        self.filename = os.path.abspath(filename)

    def fill(self, calcAmplitude):
        self.grid.fill(calcAmplitude)
        self.__initialized_splines = True

    def fill_cart(self, calcAmplitude_cart):
        self.grid.fill_cart(calcAmplitude_cart)
        self.__initialized_splines = True

    def _calculate_splines(self):
        self.grid.calculate_splines()
        self.__initialized_splines = True


    def __interpolate_q_theta_phi(self, q, theta, phi):
        '''
        :param q: must be within [qmin, qmax]
        :param theta:
        :param phi: whatever
        :return:
        '''
        raise NotImplementedError
        if theta <0 or theta > math.pi:
            raise ValueError("theta must be between 0 and pi")
        if phi <0 or phi > 2*math.pi:
            raise ValueError("phi must be between 0 and 2pi")
        if q>self.helper_grid.q_max or q<self.helper_grid.q_min:
            raise ValueError("The q value given is outside of the range qmin:qmax ([{q_min}, {q_max}])".format(q_min=self.helper_grid.q_min, q_max=self.helper_grid.q_max))
        #all the qs must be between the same two is (ie same two shells)
        i,j,k=self.helper_grid.indices_from_angles(q, theta, phi)
        m_index=self.helper_grid.index_from_indices(i,j,k)
        if m_index > self.helper_grid.totalsz:
            raise ValueError("Size overflow: the angle values you have given ({q}, {theta}, {phi}) produce an index ({index}) which exceeds"
                             " the size of the calculated grid ({totalsz}".format(q=q, theta=theta, phi=phi, index = m_index, totalsz=self.helper_grid.totalsz))


    def interpolate_theta_phi_plane(self, index, theta, phi):
        """
        :param index: index for grid layer
        :param theta:
        :param phi:
        :return:
        """
        if type(index) != int and not index.is_integer():
            raise TypeError("The first parameter (index for grid layer) must be of type integer,"
                            " instead it was %s." % type(index))
        if index < 0:
            raise ValueError("The first parameter (index for grid layer) is %s, but it must"
                             " be a positive integer" % index)
        if index > self.helper_grid.N:
            raise ValueError("The first parameter (index for grid layer) is %s, but it must"
                             " be smaller than grid.actual_size %s" % (index, self.grid.actual_size))
        if theta <0 or theta > math.pi:
            raise ValueError("theta must be between 0 and pi")
        if phi <0 or phi > 2*math.pi:
            raise ValueError("phi must be between 0 and 2pi")
        return self.grid.interpolate_theta_phi_plane(index, theta, phi)

    def get_dim(self, x=1, y=1):
        return self.grid.get_dim(x, y)

    def validation_input_interpolate(self, q, theta, phi):
        if not isinstance(q, (int, float)) or q < 0:
            raise ValueError("The first parameter q is %s, but it must"
                             " be a positive integer" % q)
        if q > self.helper_grid.q_max:
            raise ValueError("The first parameter q is %s, but it must"
                             " be smaller than q_max: %s" % (q, self.helper_grid.q_max))
        if not isinstance(theta, (int, float)) or theta < 0 or theta > math.pi :
            raise ValueError("theta must be a valid number between 0 and pi")
        if not isinstance(phi, (int, float)) or phi < 0 or phi > 2*math.pi:
            raise ValueError("phi must be a valid number between 0 and 2pi")
        if theta == math.pi:
            theta = 0.0
        if phi == 2*math.pi:
            phi = 0.0
        return float(q), float(theta), float(phi)

    def __get_interpolation_q1(self, q, theta, phi):
        """
        Receives *one* q value and returns the interpolation for the point
        :param q: a number (int or float) between q_min and q_max
        :param theta: an angle
        :param phi: an angle
        :return: interpolation for the point
        """
        if not self.__initialized_splines:
            raise ValueError("Cannot get interpolated data from uninitalized Amplitude. did you forget to call fill()?")
        q, th, ph = self.validation_input_interpolate(q, theta, phi)
        interp = self.grid.get_sphr(q, th, ph)
        return interp

    def get_interpolation(self, q_list, theta, phi):
        """
        :param q_list: list-double list of q's
        :param theta: double theta angle
        :param phi: double phi angle
        :return: list interpolation of all the q's values
        """
        if isinstance(q_list, (int, float)):
            return self.__get_interpolation_q1(q_list, theta, phi)
        interp_list = []
        try:
            q_list.sort()
            if min(q_list) < self.helper_grid.q_min or max(q_list) > self.helper_grid.q_max:
                raise ValueError("The q values given is outside of the range q_min:q_max [{q_min}, {q_max}]. "
                                 "Your range is [{min_val}, {max_val}]"
                                 .format(q_min=self.helper_grid.q_min, q_max=self.helper_grid.q_max,min_val=min(q_list), max_val=max(q_list)))
        except TypeError:
            raise TypeError("The q values should include only numbers")
        for q in q_list:
            interp_list.append(self.__get_interpolation_q1(q, theta, phi))
        return interp_list

    @staticmethod
    def q_theta_to_qZ_qPerp(q, theta):
        qZ = q * math.cos(theta)
        qPerp = q * math.sin(theta)
        res = {'qZ' : qZ, 'qPerp' : qPerp}
        return res

    def qZ_qPerp_to_q_theta(qZ, qPerp):
        q = math.sqrt( math.pow(qZ,2) + math.pow(qPerp,2) )
        
        # NOTICE THAT Q_PREP SHOULD BE THE FIRST ARGUMENT IN ATAN2 !!!
        theta = np.abs(math.atan2(qPerp, qZ))
        # Theta is abs() according to Eytan's suggestion.

        res = {'q' : q, 'theta' : theta}
        return res

    def get_intensity(
                    self, 
                    q_min,
                    calculated_points=100,
                    q_max=None,
                    phi_min=0, phi_max=2*math.pi,
                    epsilon=1e-3, seed=0, max_iter=1000000):

        if not q_max:
            q_max = q_min
            q_min = 0
        
        if q_max > self.grid.q_max:
            raise ValueError('q_max > grid.q_max !')
        
        if q_min > q_max:
            raise ValueError('q_min > q_max !')

        qZ_list = np.linspace(0-q_max, q_max, calculated_points)
        qPerp_list = np.linspace(0-q_max, q_max, calculated_points)

        qZ_list = np.round(qZ_list, 8)
        qPerp_list = np.round(qPerp_list, 8)

        arr_intensity = [[0 for qPerp in range(calculated_points)] for qZ in range(calculated_points)] 

        qZ_idx = 0
        for qZ in qZ_list:
            qPerp_idx = 0
            for qPerp in qPerp_list:
                q_theta_dict = Amplitude.qZ_qPerp_to_q_theta(qZ, qPerp)
                q = q_theta_dict['q']
                theta = q_theta_dict['theta']
                if q_min <= q <= q_max: # q is always positive
                    res = self.grid.get_intensity(q, theta, epsilon, seed, max_iter, phi_min, phi_max)
                    arr_intensity[qZ_idx][qPerp_idx] = res
                qPerp_idx += 1
            qZ_idx += 1
        return arr_intensity

    def get_crystal_intensity(self, q_min, calculated_points=100, q_max=None, phi=0):
        '''Returns the 2D intensity expected from a given model as if a crystallographic experiment was done.
        The returned array goes from -q_max to q_max. Changing phi will give the scattering from another face (as if
        turned by -phi in real-space)'''
        if not q_max:
            q_max = q_min
            q_min = 0

        if q_max > self.grid.q_max:
            raise ValueError('q_max > grid.q_max !')

        if q_min > q_max:
            raise ValueError('q_min > q_max !')

        qZ_list = np.linspace(-1*q_max, q_max, calculated_points)
        qPerp_list = np.linspace(-1*q_max, q_max, calculated_points)

        amps = np.zeros([calculated_points, calculated_points], dtype=complex)

        phi_1 = phi
        phi_2 = phi + np.pi
        if phi_2 > 2 * np.pi:
            phi_2 -= 2 * np.pi
        for i in range(calculated_points):
            for j in range(calculated_points):
                q = np.sqrt(qPerp_list[i] ** 2 + qZ_list[j] ** 2)
                theta = np.arctan2(qPerp_list[i], qZ_list[j])
                if q_min <= q <= q_max:
                    if theta < 0:
                        amps[j, i] += self.get_interpolation([q], np.abs(theta), phi_2)[0]
                    else:
                        amps[j, i] += self.get_interpolation([q], theta, phi_1)[0]

        I = np.power(np.abs(amps), 2)
        return I


    def get_intensity_q_theta(self, q_list, theta_list=None, epsilon=1e-3, seed=0, max_iter=1000000, phi_min=0, phi_max=2*math.pi):
        """
        replace DomainModel::CalculateIntensityVector
        calculate the 2D intensity for a vector of q's and theta's by send to JacobianSphereGrid::CalculateIntensity
        If theta_list is None, calculate 1D intensity using q only.
        :param q_list: list-double list of q's
        :param theta_list: optional param list-double list of q's
        :param epsilon: the allowed error
        :param seed: start seed point ( for the randomization)
        :param max_iter: max iteration for the optimization, for each q.
        :return: If theta_list is None: a vector of the intensity each q from the list.
                 else: a 2 dimensional matrix of intensity for each q, theta from the lists.
        """
        if theta_list is None:
            arr_intensity = []
            for q in q_list:
                arr_intensity.append(self.grid.get_intensity(q, None, epsilon, seed, max_iter, phi_min, phi_max))
            return arr_intensity

        
        arr_intensity = [[0 for t in range(len(theta_list))] for q in range(len(q_list))] 
        q_idx = 0
        for q in q_list:
            t_idx = 0
            for t in theta_list:
                arr_intensity[q_idx][t_idx] = self.grid.get_intensity(q, t, epsilon, seed, max_iter, phi_min, phi_max)
                t_idx = t_idx + 1
            q_idx = q_idx + 1
            
        return arr_intensity

    def calculate_intensity(self, q, theta=None, epsilon=1e-3, seed=0, max_iter=1000000, phi_min=0, phi_max=2*math.pi): # max_iter = 'iterations' on c++
        """
        calculate the intensity for one q point - DomainModel::DefaultCPUCalculation, JacobianSphereGrid::CalculateIntensity
        :param q: the point (x)
        :param epsilon: the allowed error
        :param seed: start seed point ( for the randomization)
        :param max_iter: max iteration for the optimization
        :return: the intensity for the q point
        """
        return self.grid.get_intensity(q, theta, epsilon, seed, max_iter, phi_min, phi_max)


    @staticmethod
    def _legacy_load_bytes(filestream):
        def _peek(File, length):
            pos = File.tell()
            data = File.read(length)
            File.seek(pos)
            return data
        
        has_headers = False
        headers = []

        if _peek(filestream, 1).decode('ascii') == '#':
            desc = filestream.read(2)
            tempdesc = desc.decode('ascii')
            if (tempdesc[1] == '@'):
                has_headers = True
            else:
                tmphead = filestream.readline()
                headers.append(desc + tmphead)

        if has_headers:
            offset_bytes = filestream.read(np.dtype(np.uint32).itemsize)
            offset = np.fromstring(offset_bytes, dtype=np.uint32)
            del_aka_newline = filestream.readline()  # b"\n"

            while _peek(filestream, 1).decode('ascii') == '#':
                headers.append(filestream.readline())
            if offset > 0:
                filestream.seek(offset[0], 0)

        version_r = filestream.readline().rstrip()
        version = int(version_r.decode('ascii'))
        size_element_r = filestream.readline().rstrip()
        size_element = int(size_element_r.decode('ascii'))

        if size_element != int(2 * np.dtype(np.float64).itemsize):
            raise ValueError("dtype is not float64\n")

        tmpGridsize_r = filestream.readline().rstrip()
        tmpGridsize = int(tmpGridsize_r.decode('ascii'))  # I

        tmpExtras_r = filestream.readline().rstrip()
        extra_shells = int(tmpExtras_r.decode('ascii'))  # extra shells
        grid_size = (tmpGridsize - extra_shells) * 2  # grid_size

        actualGridSize = grid_size / 2 + extra_shells  # I

        i = actualGridSize
        totalsz = int((6 * i * (i + 1) * (3 + 3 + 2 * 3 * i)) / 6)
        totalsz = totalsz + 1
        totalsz = totalsz * 2

        step_size_bytes = filestream.read(np.dtype(np.float64).itemsize)
        step_size = np.fromstring(step_size_bytes, dtype=np.float64)

        q_max = np.float64(step_size * (grid_size / 2.0))

        amp_values_bytes = filestream.read(np.dtype(np.float64).itemsize * totalsz)
        amp_values = np.fromstring(amp_values_bytes, dtype=np.float64)

        header_List = []
        if has_headers:
            pos = 0
            header_List.append(desc)
            pos = pos + len(desc)

            header_List.append(offset[0].tobytes())
            pos = pos + len(offset[0].tobytes())
            header_List.append(del_aka_newline)
            pos = pos + len(del_aka_newline)

            for i in headers:
                header_List.append(i)
                pos = pos + len(i)
            header_List.append(del_aka_newline)
            header_List.append(del_aka_newline)
            pos = pos + 2 * len(del_aka_newline)

            pos = np.int32(pos)
            if pos != offset[0]:
                header_List[1] = pos.tobytes()

            header_List.append(version_r + b"\n")
            header_List.append(size_element_r + b"\n")
            header_List.append(tmpGridsize_r + b"\n")
            header_List.append(tmpExtras_r + b"\n")
            header_List.append(step_size.tobytes())

        amp = Amplitude(grid_size, q_max)
        amp.extra_shells = extra_shells
        amp._values = amp_values
        amp.external_headers = header_List
        return amp

    @staticmethod
    def _legacy_load(filename):
        with open(filename, "rb") as f:
            try:
                f_bytes = f.read()
                f_bytes_io = io.BytesIO(f_bytes)
                amp = Amplitude._legacy_load_bytes(f_bytes_io)
            except ValueError as ve:
                raise ValueError("error in file: " + filename + " " + ve)

        return amp

    @staticmethod
    def load(filename):
        '''
        A static method, `load`,  which receives a filename of an Amplitude file, and returns an Amplitude instance
        with the values from that file already loaded.

        :param filename: filename of an Amplitude file
        :return: instance of Amplitude class.
        '''
        # legacy support
        extension = pathlib.Path(filename).suffix
        if extension == ".amp":
            amp= Amplitude._legacy_load(filename)
            amp._calculate_splines()
            amp.filename = filename
            return amp

        ampzip = zipfile.ZipFile(filename, mode='r')
        b_info = ampzip.read("criticalinfo.json")
        info_str = b_info.decode('ascii')
        info_dict = json.loads(info_str)
        q_max = info_dict["qmax"]
        grid_size = info_dict["gridSize"]
        step_size = info_dict["stepSize"]
        assert math.isclose(q_max, step_size * grid_size / 2.0, abs_tol=1e-12)
        extra_shells = 3
        actualGridSize = grid_size / 2 + extra_shells  # I
        i = actualGridSize
        totalsz = int((6 * i * (i + 1) * (3 + 3 + 2 * 3 * i)) / 6)
        totalsz = totalsz + 1
        totalsz = totalsz * 2
        dat = ampzip.read("grid.dat")
        amp_values = np.frombuffer(dat, dtype=np.float64, count=totalsz)
        try:
            b_headers = ampzip.read("header.json")
            header_str = b_headers.decode('ascii')
            header_dict = json.loads(header_str)
        except KeyError:
            pass  # it's okay to not have headers

        amp = Amplitude(grid_size, q_max)
        amp.extra_shells = extra_shells
        amp._values = amp_values
        amp._calculate_splines()
        amp.external_headers = header_dict
        amp.filename = filename
        return amp


def amp_to_ampj_converter(amp_filename):
    old_a = Amplitude.load(amp_filename)
    new_filename = amp_filename + "j"
    old_a.save(new_filename)
    new_a = Amplitude.load(new_filename)

    for old, new in zip(old_a._values, new_a._values):
        assert old == new

    return new_filename


def ampj_to_amp_converter(ampj_filename):
    ampj = Amplitude.load(ampj_filename)
    new_filename = ampj_filename[:-1]
    ampj.old_save(new_filename)
    amp = Amplitude.load(new_filename)

    for old, new in zip(ampj._values, amp._values):
        assert old == new

    return new_filename


def scrap():
    from pathlib import Path
    dir = r"C:\Users\devora\Sources\dplus\dplus\PythonInterface\tests\time_tests\files_for_tests_with_cache\gpu"
    folders = ["Man_Symm_Impl_No_Charges_With_Hydr_MC",
               "Scripted_Symm_Impl_No_Charges_With_Hydr_Ga_Kr",
               "Scripted_Symm_Impl_No_Charges_wo_Hydr_MC",
               "Single_PDB_Impl_No_Charges_wo_Hydr_Ga_Kr",
               "Space_fill_Symm_Impl_No_Charges_With_Hydr_MC"]

    files = []
    for folder in folders:
        test_dir = os.path.join(dir, folder, 'cache')
        for file in os.listdir(test_dir):
            ext = Path(file).suffix
            if ext == ".amp":
                files.append(os.path.join(test_dir, file))

    return files


if __name__ == "__main__":
    # files = scrap()
    files=[]
    for file in files:
        new = amp_to_ampj_converter(file)
        print(new)
