
from Grid cimport JacobianSphereGrid
import numpy as np
import json
from libc.string cimport memcpy
import ctypes
cimport numpy as np
from cython cimport view
import pathlib
import zipfile

cdef class CJacobianSphereGrid:
    cdef JacobianSphereGrid c_grid


    def __cinit__(self, unsigned short gridSize=0, double qMax=0):
        self.c_grid = JacobianSphereGrid(gridSize, qMax)
    @property
    def q_max(self):
        return self.c_grid.GetQMax()

    @property
    def grid_size(self):
        return self.c_grid.GetSize()

    def index_from_indices(self, int qi, long long ti, long long pi):
        return self.c_grid.IndexFromIndices(qi, ti, pi)

    def indices_from_index(self, long long index):
        cdef  int qi = 0;
        cdef  long long ti = 0, pi = 0;
        self.c_grid.IndicesFromIndex(index, qi, ti, pi)
        return qi, ti, pi

    def calculate_splines(self):
        self.c_grid.CalculateSplines()

    @property
    def real_size(self):
        return self.c_grid.GetRealSize()

    @property
    def actual_size(self):
        return self.c_grid.GetActualSize()

    @property
    def extras(self):
        return self.c_grid.GetExtras()

    def get_data(self):
        cdef double *data_array = self.c_grid.GetDataPointer()
        real_size_bytes = self.real_size
        real_size_double = int(real_size_bytes/sizeof(double))
        cdef double[:] view = <double[:real_size_double]> data_array

        return np.asarray(view)

    def get_interpolant_coeffs(self):
        cdef double *data_array = self.c_grid.GetInterpolantPointer()
        real_size_bytes = self.real_size
        real_size_double = int(real_size_bytes/sizeof(double))
        cdef double[:] view = <double[:real_size_double]> data_array
        return np.asarray(view)


    def get_param_json_string(self):
        return self.c_grid.GetParamJsonString()

    @property
    def step_size(self):
        return self.c_grid.stepSize

    @property
    def theta_divisions(self):
        return self.c_grid.thetaDivisions

    @property
    def phi_divisions(self):
        return self.c_grid.phiDivisions

    def fill(self, calcAmplitude):
        data = self.get_data()
        cdef long long dims = (self.real_size / sizeof(double)) / 2;
        cdef long long ind = 0

        while ind < dims:
            rs = np.complex64(0)
            qi, thi ,phi = self.indices_from_index(ind)
            qI = np.double(qi) * self.step_size
            # The comment below removed because we don't want convert the coordinates to cartesian
            M_PI =  np.double(3.14159265358979323846)
            M_2PI = np.double(6.28318530717958647692528676656)
            dimz = int(self.phi_divisions) * qi
            if qi == 0:
            #     cst = snt = csp = snp = 0.0
                tI = 0
                pI = 0
            else:
                tI = M_PI  * np.double(thi) / np.double(int(self.theta_divisions) * qi)
                pI = M_2PI * np.double(phi) / np.double(dimz)
            #     cst = np.cos(tI)
            #     snt = np.sin(tI)
            #     csp = np.cos(pI)
            #     snp = np.sin(pI)
            # rs = calcAmplitude(qI * snt * csp, qI * snt * snp, qI * cst)
            rs = calcAmplitude(qI, tI, pI)
            data[2*ind] = rs.real
            data[2*ind+1] = rs.imag
            ind += 1
        self.calculate_splines()

    def fill_cart(self, calcAmplitude_cart):
        data = self.get_data()
        cdef long long dims = (self.real_size / sizeof(double)) / 2;
        cdef long long ind = 0

        while ind < dims:
            rs = np.complex64(0)
            qi, thi ,phi = self.indices_from_index(ind)
            qI = np.double(qi) * self.step_size
            M_PI =  np.double(3.14159265358979323846)
            M_2PI = np.double(6.28318530717958647692528676656)
            dimz = int(self.phi_divisions) * qi
            if qi == 0:
                cst = snt = csp = snp = 0.0
            else:
                tI = M_PI  * np.double(thi) / np.double(int(self.theta_divisions) * qi)
                pI = M_2PI * np.double(phi) / np.double(dimz)
                cst = np.cos(tI)
                snt = np.sin(tI)
                csp = np.cos(pI)
                snp = np.sin(pI)
            rs = calcAmplitude_cart(qI * snt * csp, qI * snt * snp, qI * cst)
            data[2*ind] = rs.real
            data[2*ind+1] = rs.imag
            ind += 1
        self.calculate_splines()

    def interpolate_theta_phi_plane(self, const unsigned short ri, const double theta, const double phi):
        return self.c_grid.InterpolateThetaPhiPlane(ri, theta, phi)

    def get_dim(self,  x, y):
        cdef unsigned short dim_x = self.c_grid.GetDimX()
        cdef unsigned short dim_y = self.c_grid.GetDimY(x)
        cdef unsigned short dim_z = self.c_grid.GetDimZ(x, y)
        dim_xyz = [dim_x, dim_y, dim_z]
        return dim_xyz

    def get_sphr(self, double _q, double theta, double phi):
        """
        Receives one q value and returns the interpolation for the point
        :param q:
        :param theta:
        :param phi:
        :return: interpolation for the point by JacobianSphereGrid::GetSphr( double rr, double th, double ph ) const
        """
        cdef double q = _q
        cdef double th = theta
        cdef double ph = phi
        result =  self.c_grid.GetSphr(q, th, ph)
        return result

    
    def get_intensity(self, _q, _theta, _epsi, _seed, _iterations, _phi_min, _phi_max):
        cdef double q = _q
        cdef double epsi = _epsi
        cdef unsigned int seed = _seed
        cdef  long long iterations = _iterations
        cdef double theta
        cdef double phi_min = _phi_min
        cdef double phi_max = _phi_max

        if _theta is None:
            result = self.c_grid.CalculateIntensity(q, epsi, seed, iterations, phi_min, phi_max)
        else:
            theta = _theta
            result = self.c_grid.CalculateIntensity(q, theta, epsi, seed, iterations, phi_min, phi_max)
        
        return result




