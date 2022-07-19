from dplus.Amplitudes import Grid

import numpy as np
import math

q_max = 7.5
grid_size = 50
test_grid = Grid(grid_size, q_max)
M_2PI = 6.28318530717958647692528676656
M_PI = 3.14159265358979323846
phiDivisions = 6
thetaDivisions = 3
double = np.float64
long = math.floor
step_size = double(q_max / double(grid_size / 2.0))
npd = np.float64


def G_i_q(i):
    '''
On the i-th shell there are 6i(3i+1) grid points

Within each shell (q_i), the values are arranged in a \phi-major storage order (i.e. two neighboring \phi values (with the same \theta  - i.e, the same i and j indexes) will be adjacent in memory, where two \theta values with the same \phi will be distanced by 6i other values.
The total number of grid points, G^i_q, inside i spheres is:
    G^i_q=6i+12i^2 + 6i^3
'''
    # almost definitely related to bot:
    # lqi= long((3*index)/(thetaDivisions*phiDivisions))**(1./3.)
    # ((lqi * 6 * (lqi + 1) * (3 + 3 + 2 * 3 * lqi)) / 6)
    return 6 * i + 12 * i ** 2 + 6 * i ** 3


# 1. Gets q_Max, and Grid size and creates a grid (a list of q\theta\phi points in a specific order)
class CreateGridUri:
    # 1. creates a grid (a list of q\theta\phi points in a specific order)
    @staticmethod
    def uri_func():
        '''
        To create the grid:
        we need to get the values of q_i, \theta_{i,j}, \phi_{i,j,k}, and the amplitude at that point (which depends on the model we compute).

        On the i-th shell of the grid:
        q_{i} = (i*q_Max)/N
        where i \in { 0,1,....,N, N+1, N+2, N+3}

        we then define: J_{i} =3 i + 1
        The j - th polar angle on the i - th shell is:
        \theta_{i,j} = (j*\pi)/(J_{i} - 1) =  (j*\pi)/( 3 i)
        where j \in {0,....,(J_{i} - 1)}

        we define: K_{i,j}=6*i
        The k - th azimuthal angle of the j-th polar angle on the i-th shell is then given by
        \phi_{i,j,k}=(2*\pi*k)/K_{i,j} = (\pi*k)/(3*i) ,
        where k\in { 0,....,K_{i,j}}

        '''
        N = int(grid_size / 2)
        for i in range(0, N + 4):
            if i == 0:
                yield 0, 0, 0
                continue
            q = (i * q_max) / N
            J_i = (3 * i) + 1
            for j in range(0, J_i):
                K_ij = 6 * i
                theta_ij = (j * M_PI) / (J_i - 1)
                theta_ij = (j * M_PI) / (3 * i)
                for k in range(0, K_ij):
                    phi_ijk = (2 * M_PI * k) / K_ij
                    phi_ijk = (M_PI * k) / (3 * i)
                    yield q, theta_ij, phi_ijk
                    continue
                continue
            continue

    @staticmethod
    def uri_func2():
        N = int(grid_size / 2)
        for i in range(0, N + 4):
            if i == 0:
                yield 0, 0, 0
                continue
            J_i = (3 * i) + 1
            K_ij = 6 * i
            for j in range(0, J_i):
                for k in range(0, K_ij):
                    yield _AnglesFromIndices.uri_func(i, j, k)

    @staticmethod
    def uri_func3():
        kk = (grid_size / 2) + 3
        totalsz = (6 * kk * (kk + 1) * (3 + 3 + 2 * 3 * kk)) / 6
        totalsz += 1
        totalsz = int(totalsz)
        for i in range(totalsz):
            yield AnglesFromIndex.uri_func(i)


class CreateGridAvi:
    @staticmethod
    def avi_func():
        '''
void JacobianSphereGrid::Fill() {

	const long long dims = totalsz / 2;

	for(long long ind = 0; ind < dims; ind++) {
		std::complex<double> rs(0.0,0.0);
		int qi;
		long long thi, phi;

		// Determine q_vec as a function of the single index
		IndicesFromIndex(ind, qi, thi, phi);

		//const int dimy = int(thetaDivisions) * qi + 1;	// (4*i+1)
		const int dimz = int(phiDivisions) * qi;

		double qI = double(qi) * stepSize;
		double tI = M_PI  * double(thi) / double(int(thetaDivisions) * qi);	// dimy - 1
		double pI = M_2PI * double(phi) / double(dimz);

		double cst = cos(tI),
			   snt = sin(tI);

		double csp = cos(pI),
			   snp = sin(pI);

		if(qi == 0) {
			cst = snt = csp = snp = 0.0;
		}

		/*The points on the sphere with radius qI can be parameterized via:
		x= x0 + qI*cos(phi)*sin(theta)
		y = y0 + qI*sin(phi)*sin(theta)
		z= z0 + qI*cos(theta)
		calcAmplitude - calculate the fourier transform of this vec(x,y,z) - (in q space)
		*/
		rs = amp->calcAmplitude(qI * snt * csp, qI * snt * snp, qI * cst);

		data[2*ind]   = rs.real();
 		data[2*ind+1] = rs.imag();
	}
        '''
        kk = (grid_size / 2) + 3
        totalsz = (6 * kk * (kk + 1) * (3 + 3 + 2 * 3 * kk)) / 6
        totalsz += 1

        for ind in range(int(totalsz)):
            if ind == 0:
                yield 0, 0, 0
                continue
            i, j, k = _IndicesFromIndex.avi_func(ind)
            # yield i, j, k
            dim_z = int(phiDivisions) * i
            q = double(i) * step_size
            theta = M_PI * double(j) / double(int(thetaDivisions) * i)
            phi = M_2PI * double(k) / double(dim_z)
            yield npd(q), npd(theta), npd(phi)

    @staticmethod
    def avi_func2():
        N = int(grid_size / 2)
        for i in range(0, N + 3):
            yield AnglesFromIndex.avi_func(i)

        def polar_to_cartesian(q, theta, phi):
            cst = math.cos(theta)
            snt = math.sin(theta)

            csp = math.cos(phi)
            snp = math.sin(phi)

            i = math.floor(q / step_size)
            if i == 0:
                cst = snt = csp = snp = 0.0

            # The points on the sphere with radius qI can be parameterized via:
            # x= x0 + qI*cos(phi)*sin(theta)
            # y = y0 + qI*sin(phi)*sin(theta)
            # z= z0 + qI*cos(theta)
            # calcAmplitude - calculate the fourier transform of this vec(x,y,z) - (in q space)
            return i * snt * csp, i * snt * snp, i * cst

        def call_amplitude_calc_func(cart1, cart2, cart3):
            pass
            # rs = calc_amplitude()
            # data[2*ind]= rs.real
            # data[2*ind+1]=rs.imag


class _IndicesFromIndex:
    # verified as equal
    @staticmethod
    def uri_func(m):
        '''
        although this was given under "angles from index", it describes "indices from index":

    the relation between the index m and the three indices used to indicate the discrete values of
    q_i, \theta_{i,j}, and \phi_{i,j,k}  (or the i, j, and k, indexes) is:

    i=\floor{(m/6)^(1/3)},
    unless
    m >  G^(i-1)_q =  6(i-1)+12(i-1)^2 + 6(i-1)^3
    in which case
    i= 1 + \floor{(m/6)^(1/3)}

    we define:
    R_i = m - {6(i-1)+12(i-1)^2 + 6(i-1)^3} - 1
    and then
    j= \floor{R_i/(6i)}
    and
    k= R_i - 6*i*j


    also:

    q_{i} = (i *q_Max)/N
    \theta_{i,j}= (j*\pi)/( 3 i)
    \phi_{i,j,k}= (\pi*k)/(3*i)
        '''
        if m == 0:
            return 0, 0, 0
        i = math.floor((m / 6) ** (1. / 3.))

        if m > G_i_q(i):
            i += 1

        R_i = m - G_i_q(i - 1) - 1

        j = math.floor(R_i / (6 * i))
        k = R_i - 6 * i * j

        return i, j, k

    @staticmethod
    def avi_func(index):
        '''
//This function find for each index (cell) in the grid what should be the values of (q,phi,theta) in this cell
void JacobianSphereGrid::IndicesFromIndex( long long index, int &qi, long long &ti, long long &pi ) {
	// Check the origin
	if(index == 0) {
		qi = 0;
		ti = 0;
		pi = 0;
		return;
	}

	long long bot, rem;
	// Find the q-radius
	long long lqi = icbrt64((3*index)/(thetaDivisions*phiDivisions));
	bot = (lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	if(index > bot )
		lqi++;
	lqi--;
	bot =(lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6;	//(lqi*(28 + lqi*(60 + 32*lqi))) / 3;
	lqi++;
	qi = (int)lqi;
	rem = index - bot - 1;
	// Find the theta and phi radii
	ti = rem / (phiDivisions*qi);
	pi = rem % (phiDivisions*qi);

}
        '''
        if index == 0:
            return 0, 0, 0

        lqi = math.floor(((3 * index) / (thetaDivisions * phiDivisions)) ** (1. / 3.))
        bot = math.floor((lqi * phiDivisions * (lqi + 1) * (
                    3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6)  # (lqi * (28 + lqi * (60 + 32 * lqi))) / 3
        if index <= bot:
            lqi -= 1
            bot = long((lqi * phiDivisions * (lqi + 1) * (3 + thetaDivisions + 2 * thetaDivisions * lqi)) / 6)

        lqi += 1
        i = int(lqi)
        rem = index - bot - 1
        # Find the theta and phi radii
        j = math.floor(rem / (phiDivisions * i))
        k = long(rem % (phiDivisions * i))

        return i, j, k


class _AnglesFromIndices:
    @staticmethod
    def uri_func(i, j, k):
        '''
        derived from the fill grid code above
        '''
        if i == 0:
            return 0, 0, 0
        N = int(grid_size / 2)
        q = (i * q_max) / N
        theta_ij = (j * M_PI) / (3 * i)
        phi_ijk = (M_PI * k) / (3 * i)
        return npd(q), npd(theta_ij), npd(phi_ijk)

    @staticmethod
    def avi_func(i, j, k):
        '''
        taken from fill grid
        '''
        if i == 0:
            return 0, 0, 0
        dim_z = int(phiDivisions) * i
        q = double(i) * step_size
        theta = M_PI * double(j) / double(int(thetaDivisions) * i)
        phi = M_2PI * double(k) / double(dim_z)
        return q, theta, phi


# 2. Gets an index m, q Max, and Grid size, and returns q, \theta, \phi (which is the physical location on the grid)
class AnglesFromIndex:
    @staticmethod
    def uri_func(m):
        '''
        although uri sent the description in a mail for this function, it really is
        angles from indices ( indices from index ())
        and hence has been copied to indices_from_index instead
        '''
        i, j, k = _IndicesFromIndex.uri_func(m)
        return _AnglesFromIndices.uri_func(i, j, k)

    @staticmethod
    def avi_func(m):
        i, j, k = _IndicesFromIndex.avi_func(m)
        return _AnglesFromIndices.avi_func(i, j, k)


class _IndexFromIndices:
    @staticmethod
    def uri_func(i, j, k):
        '''
        m=  6(i-1)+12(i-1)^2 + 6(i-1)^3 + 6*i* j +k+1.
        '''
        if i == 0:
            return 0
        return 6 * (i - 1) + 12 * (i - 1) ** 2 + 6 * (i - 1) ** 3 + 6 * i * j + k + 1

    @staticmethod
    def avi_func(i, j, k):
        '''
            long long JacobianSphereGrid::IndexFromIndices( int qi, long long ti, long long pi ) const {
            if(qi == 0)
                return 0;
            qi--;
            long long base = ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;
            return (base + ti * phiDivisions * (qi+1) + pi + 1);	// The +1 is for the origin
        }

            '''
        if i == 0:
            return 0
        i -= 1
        base = long(((i + 1) * (phiDivisions * i) * (2 * thetaDivisions * i + thetaDivisions + 3)) / 6)
        thingy = (base + j * phiDivisions * (i + 1) + k + 1)  # the +1 is for the origin
        return thingy


# 3. Gets q, \theta. \phi, q Max and Grid size and returns its indexes (i,j,k, and m).
class IndicesFromAngles:
    @staticmethod
    def uri_func(q, theta, phi):
        '''

i=\floor{(q*N)/q_Max},
j= [ 3*i +1+\floor{(3*i*\theta)/\pi} ] \mod {3i +1},
k= [6*i +1 + \floor{(3*i*\phi)/(\pi)}] \mod {6i}.
        '''
        pass

    @staticmethod
    def avi_func(qi, theta, phi):
        '''
void JacobianSphereGrid::IndicesFromRadians( const u16 ri, const double theta, const double phi,
											long long &tI, long long &pI, long long &base, double &tTh, double &tPh ) const {
	// Determine the first cell using ri
	int qi = ri - 1;
	base = 1 + ((qi+1)*(phiDivisions * qi)*(2 * thetaDivisions * qi + thetaDivisions + 3)) / 6;

	// Determine the lowest neighbors coordinates within the plane
	int phiPoints = phiDivisions * ri;
	int thePoints = thetaDivisions * ri;
	double edge = M_2PI / double(phiPoints);

	tI = (theta / M_PI) * double(thePoints);
	pI = (phi  / M_2PI) * double(phiPoints);

	// The value [0, 1] representing the location ratio between the two points
	tTh = (theta / edge) - tI; //fmod(theta, edge) / edge;
	tPh = (phi   / edge) - pI; //fmod(phi, edge) / edge;

	if(fabs(tTh) < 1.0e-10)
		tTh = 0.0;
	if(fabs(tPh) < 1.0e-10)
		tPh = 0.0;
	assert(tTh >= 0.0);
	assert(tPh >= 0.0);
	assert(tTh <= 1.0000000001);
	assert(tPh <= 1.0000000001);

	//pI = (pI == phiPoints) ? 0 : pI;
	if(pI == phiPoints) {
		assert(tPh <= 0.000001);
		pI = 0;
	}
}
        '''
        ri = qi + 1
        base = 1 + ((qi + 1) * (phiDivisions * qi) * (2 * thetaDivisions * qi + thetaDivisions + 3)) / 6

        # Determine the lowest neighbors coordinates within the plane
        phiPoints = phiDivisions * ri
        thePoints = thetaDivisions * ri
        edge = M_2PI / double(phiPoints)

        tI = (theta / M_PI) * double(thePoints)
        pI = (phi / M_2PI) * double(phiPoints)

        return int(base), int(tI), int(pI)


class __IndexFromAngles:
    # fully implemented via combinations of the above
    pass


class Test_For_M:
    def test_indices_from_index(self):
        for m in range(500):
            u1, u2, u3 = test_grid.indices_from_index(m)
            a1, a2, a3 = _IndicesFromIndex.avi_func(m)
            assert (u1, u2, u3) == (a1, a2, a3)

    def test_index_from_indices(self):
        for m in range(500):
            u1, u2, u3 = test_grid.indices_from_index(m)
            a1, a2, a3 = _IndicesFromIndex.avi_func(m)
            mu = test_grid.index_from_indices(u1, u2, u3)
            ma = _IndexFromIndices.avi_func(a1, a2, a3)
            assert (m == mu == ma)

    def test_angles_from_indices(self):
        for m in range(500):
            u1, u2, u3 = test_grid.indices_from_index(m)
            a1, a2, a3 = _IndicesFromIndex.avi_func(m)
            uri = test_grid.angles_from_indices(u1, u2, u3)
            avi = _AnglesFromIndices.avi_func(a1, a2, a3)
            assert (uri == avi)

    def test_angles_from_index(self):
        for m in range(500):
            (u1, u2, u3) = test_grid.angles_from_index(m)
            (a1, a2, a3) = AnglesFromIndex.avi_func(m)
            assert (u1, u2, u3) == (a1, a2, a3)

    def test_indices_from_angles(self):
        for m in range(500):
            (u1, u2, u3) = test_grid.angles_from_index(m)
            (a1, a2, a3) = AnglesFromIndex.avi_func(m)
            try:
                urii = test_grid.indices_from_angles(u1, u2, u3)
            except ZeroDivisionError:
                pass
            avii = IndicesFromAngles.avi_func(a1, a2, a3)
            hi = 1
            # assert (urii == avii)


class TestCreateGrid:
    def test_length(self):
        len_uri = 0
        for i in test_grid.create_grid():
            len_uri += 1
        len_avi = 0
        for avi in CreateGridAvi.avi_func():
            len_avi += 1
        assert len_uri == len_avi

    def test_values(self):
        for (ind, uri), avi in zip(test_grid.create_grid(), CreateGridAvi.avi_func()):
            assert uri == avi


def test_index_from_angles():
    g = Grid(8, 50)
    for m_index, (index, (q, theta, phi)) in enumerate(g.create_grid()):
        m = g.index_from_angles(q, theta, phi)
        # if not index==m:
        #    print(index, "(",m, "):\t",  q, theta, phi, ":\t", g.indices_from_angles(q, theta, phi))
        assert m_index == m
