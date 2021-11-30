import json
import math

import numpy as np
import os
import zipfile
import pathlib
from CythonWrapping import CJacobianSphereGrid

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
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az


class Grid:
    '''
    This class is described in pages 12-15 of the paper

    The class Grid is initialized with `q_max` and `grid_size`.
    It is used to create/describe a grid of `q`, `theta`, `phi` angle values.
    These values can be described using two sets of indexing:

    1. The overall index `m`
    2. The individual angle indices `i`, `j`, `k`
    '''

    def __init__(self, q_max, grid_size):
        if grid_size % 2:
            raise ValueError("Grid size must be even")
        self.q_max = q_max
        self.grid_size = grid_size
        self.extra_shells = 3

    @property
    def step_size(self):
        '''
        The difference between q's in the grid.

        :return: double q_max/N
        '''
        return npDouble(self.q_max / (self.N))

    @property
    def N(self):
        return int(self.grid_size / 2)

    @property
    def actual_size(self):
        return self.N + self.extra_shells

    def _G_i_q(self, i):
        return 6 * i + 12 * i ** 2 + 6 * i ** 3

    def create_grid(self):
        '''
        a generator that returns q, theta, phi angles in phi-major order
        '''
        for i in range(0, self.N + 4):
            if i == 0:
                yield 0, 0, 0
                continue
            J_i = (3 * i) + 1
            K_ij = 6 * i
            for j in range(0, J_i):
                for k in range(0, K_ij):
                    yield self.angles_from_indices(i, j, k)

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

        phiPoints = 6.0 * i
        thePoints = 3.0 * i

        j = math.floor((theta / PI) * thePoints + eps)
        k = math.floor((phi / (PI * 2)) * phiPoints + eps)

        return i, j, k

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

    def __init__(self, q_max, grid_size):
        self.grid = CJacobianSphereGrid(q_max, grid_size)
        self.external_headers = None
        self.__description = ""
        self.filename = ""

    @property
    def values(self):
        '''
        array that contains the grid intensity values as 2 values - real and imaginary

        :return: values array
        '''
        values_array = self.grid.get_data()
        if len(values_array):
            return values_array
        else:
            raise ValueError("Amplitude values empty-- has not been initialized yet")

    @values.setter
    def values(self, tmp_values):
        data = self.values
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
        values = self.values
        complex_arr = np.zeros((int(values.__len__() / 2), 1), dtype=np.complex)
        for index in range(0, complex_arr.__len__()):
            complex_arr[index] = values[2 * index] + 1j * values[2 * index + 1]
        return complex_arr

    @property
    def default_header(self):
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
            return self.default_header

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
            amps = np.float64(self.values)
            amps.tofile(f)
        self.filename = os.path.abspath(filename)

    def save(self, filename):
        extension = pathlib.Path(filename).suffix
        if extension == ".amp":
            raise ValueError("Please save amplitudes in .ampj format, not .amp")
        ampzip = zipfile.ZipFile(filename, mode='w')
        ampzip.writestr('grid.dat', self.values.tobytes())
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

    def calculate_splines(self):
        self.grid.calculate_splines()

    def interpolate_theta_phi_plane(self, ri, theta, phi):
        return self.grid.interpolate_theta_phi_plane(ri, theta, phi)

    @staticmethod
    def _legacy_load(filename):
        def _peek(File, length):
            pos = File.tell()
            data = File.read(length)
            File.seek(pos)
            return data

        has_headers = False
        headers = []
        with open(filename, "rb+") as f:
            if _peek(f, 1).decode('ascii') == '#':
                desc = f.read(2)
                tempdesc = desc.decode('ascii')
                if (tempdesc[1] == '@'):
                    has_headers = True
                else:
                    tmphead = f.readline()
                    headers.append(desc + tmphead)

            if has_headers:
                offset = np.fromfile(f, dtype=np.uint32, count=1, sep="")
                del_aka_newline = f.readline()  # b"\n"

                while _peek(f, 1).decode('ascii') == '#':
                    headers.append(f.readline())
                if offset > 0:
                    f.seek(offset[0], 0)

            version_r = f.readline().rstrip()
            version = int(version_r.decode('ascii'))
            size_element_r = f.readline().rstrip()
            size_element = int(size_element_r.decode('ascii'))

            if size_element != int(2 * np.dtype(np.float64).itemsize):
                raise ValueError("error in file: " + filename + "dtype is not float64\n")

            tmpGridsize_r = f.readline().rstrip()
            tmpGridsize = int(tmpGridsize_r.decode('ascii'))  # I

            tmpExtras_r = f.readline().rstrip()
            extra_shells = int(tmpExtras_r.decode('ascii'))  # extra shells
            grid_size = (tmpGridsize - extra_shells) * 2  # grid_size

            actualGridSize = grid_size / 2 + extra_shells  # I

            i = actualGridSize
            totalsz = int((6 * i * (i + 1) * (3 + 3 + 2 * 3 * i)) / 6)
            totalsz = totalsz + 1
            totalsz = totalsz * 2
            step_size = np.fromfile(f, dtype=np.float64, count=1, sep="")
            q_max = np.float64(step_size * (grid_size / 2.0))

            amp_values = np.fromfile(f, dtype=np.float64, count=totalsz, sep="")

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

            amp = Amplitude(q_max, grid_size)
            amp.extra_shells = extra_shells
            amp.values = amp_values
            amp.external_headers = header_List
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
            return Amplitude._legacy_load(filename)

        ampzip = zipfile.ZipFile(filename, mode='r')
        b_info = ampzip.read("criticalinfo.json")
        info_str = b_info.decode('ascii')
        info_dict = json.loads(info_str)
        q_max = info_dict["qmax"]
        grid_size = info_dict["gridSize"]
        step_size = info_dict["stepSize"]
        assert (q_max == step_size * (grid_size / 2.0))
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

        amp = Amplitude(q_max, grid_size)
        amp.extra_shells = extra_shells
        amp.values = amp_values
        amp.calculate_splines()
        amp.external_headers = header_dict
        return amp


def amp_to_ampj_converter(amp_filename):
    old_a = Amplitude.load(amp_filename)
    new_filename = amp_filename + "j"
    old_a.save(new_filename)
    new_a = Amplitude.load(new_filename)

    for old, new in zip(old_a.values, new_a.values):
        assert old == new

    return new_filename


def ampj_to_amp_converter(ampj_filename):
    ampj = Amplitude.load(ampj_filename)
    new_filename = ampj_filename[:-1]
    ampj.old_save(new_filename)
    amp = Amplitude.load(new_filename)

    for old, new in zip(ampj.values, amp.values):
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
    files = scrap()

    for file in files:
        new = amp_to_ampj_converter(file)
        print(new)
