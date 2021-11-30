
from Grid cimport JacobianSphereGrid
import numpy as np
import json
from libc.string cimport memcpy
import ctypes
cimport numpy as np
from cython cimport view
import pathlib
import zipfile

cdef class PyJacobianSphereGrid:
    cdef JacobianSphereGrid c_grid


    def __cinit__(self, double qMax=0, unsigned short gridSize=0):
        self.c_grid = JacobianSphereGrid(gridSize, qMax)

    def get_q_max(self):
        return self.c_grid.GetQMax()

    def get_size(self):
        return self.c_grid.GetSize()

    def get_dim_x(self):
        return self.c_grid.GetDimX()

    def get_dim_y(self, unsigned short x):
        return self.c_grid.GetDimY(x)

    def get_dim_z(self, unsigned short x, unsigned short y):
        return self.c_grid.GetDimZ(x,y)

    def get_cart(self,  double x, double y, double z):
        return self.c_grid.GetCart(x, y, z)

    def get_sphr(self, double rr, double th, double ph):
        return self.c_grid.GetSphr(rr, th, ph)

    def index_from_indices(self, int qi, long long ti, long long pi):
        return self.c_grid.IndexFromIndices(qi, ti, pi)

    def indices_from_index(self, long long index):
        cdef  int qi = 0;
        cdef  long long ti = 0, pi = 0;
        self.c_grid.IndicesFromIndex(index, qi, ti, pi)
        return qi, ti, pi

    def indices_from_radians(self, const unsigned short ri, const double theta, const double phi):
        cdef  double tTh = 0, tPh = 0;
        cdef  long long ti = 0, pi = 0, base = 0;
        self.c_grid.IndicesFromRadians(ri, theta, phi, ti, pi, base,tTh, tPh)
        return ti, pi, base, tTh, tPh

    def calculate_splines(self):
        self.c_grid.CalculateSplines()

    def get_real_size(self):
        return self.c_grid.GetRealSize()

    def get_actual_size(self):
        return self.c_grid.GetActualSize()

    def get_extras(self):
        return self.c_grid.GetExtras()

    def get_data(self):
        cdef double *data_array = self.c_grid.GetDataPointer()
        real_size_bytes = self.get_real_size()
        real_size_double = int(real_size_bytes/sizeof(double))
        cdef double[:] view = <double[:real_size_double]> data_array

        return np.asarray(view)

    def get_interpolant_coeffs(self):
        cdef double *data_array = self.c_grid.GetInterpolantPointer()
        real_size_bytes = self.get_real_size()
        real_size_double = int(real_size_bytes/sizeof(double))
        cdef double[:] view = <double[:real_size_double]> data_array
        return np.asarray(view)


    def get_param_json_string(self):
        return self.c_grid.GetParamJsonString()

    def save(self, filename, user_desc=""):
        extension = pathlib.Path(filename).suffix
        if extension == ".amp":
            raise ValueError("Please save amplitudes in .ampj format, not .amp")
        amp_zip = zipfile.ZipFile(filename, mode='w')
        amp_zip.writestr('grid.dat', self.get_data().tobytes())
        info = self.get_param_json_string()
        amp_zip.writestr("criticalinfo.json", info)

        header = self.headers(user_desc)

        amp_zip.writestr("header.json", header)

        amp_zip.close()

    def step_size(self):
        return self.c_grid.stepSize

    def theta_divisions(self):
        return self.c_grid.thetaDivisions

    def phi_divisions(self):
        return self.c_grid.phiDivisions

    def fill(self, calcAmplitude):
        data = self.get_data()
        cdef long long dims = (self.get_real_size() / sizeof(double)) / 2;
        cdef long long ind = 0

        M_PI =  np.double(3.14159265358979323846)
        M_2PI = np.double(6.28318530717958647692528676656)
        while ind < dims:
            rs = np.complex64(0)
            qi, thi ,phi = self.indices_from_index(ind)
            dimz = int(self.phi_divisions()) * qi
            qI = np.double(qi) * self.step_size()
            if qi == 0:
                cst = snt = csp = snp = 0.0
            else:
                tI = M_PI  * np.double(thi) / np.double(int(self.theta_divisions()) * qi)
                pI = M_2PI * np.double(phi) / np.double(dimz)
                cst = np.cos(tI)
                snt = np.sin(tI)
                csp = np.cos(pI)
                snp = np.sin(pI)
            rs = calcAmplitude(qI * snt * csp, qI * snt * snp, qI * cst)
            data[2*ind] = rs.real
            data[2*ind+1] = rs.imag
            ind += 1
        self.calculate_splines()

    def interpolate_theta_phi_plane(self, const unsigned short ri, const double theta, const double phi):
        return self.c_grid.InterpolateThetaPhiPlane(ri, theta, phi)




