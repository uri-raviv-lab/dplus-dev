import os.path
import fileinput
import numpy as np
from numpy.random import default_rng, randint, random, normal
from scipy.integrate import simpson
from scipy.fft import dst, fftfreq
from scipy.constants import k as Kb
import scipy.stats as stats
from scipy.special import gamma, sici, erfcx
from scipy.spatial import distance_matrix
import csv
import math
import dace as dc
from dace import dtypes#, device


V = dc.symbol('V', dc.int64)
W = dc.symbol('W')
Y = dc.symbol('Y')
M = dc.symbol('M')
TV = dc.symbol('TV')
TF = dc.symbol('TF')
Q = dc.symbol('Q')
L = dc.symbol('L', dc.int64)
U = dc.symbol('U')
S = dc.symbol('S')
NFC = dc.symbol('NFC')
RO = dc.symbol('RO', dc.float64)

egam = np.exp(np.euler_gamma)
W_S = np.sqrt(6) / (32 * np.pi**3) * gamma(1/24) * gamma(5/24) * gamma(7/24) * gamma(11/24)
R_INF = W_S / 3


def MoveToGC(Array):
    """Given a certain matrix of (Nx3) with Cartesian coordinates, returns the same array but moved to its geometric
    center."""
    r = np.zeros(3)
    len_array = Array.shape[0]
    # print(len_array, Array.shape)
    for i in range(len_array):
        r[:] += Array[i][:]
    r[:] /= len_array
    for i in range(len_array):
        Array[i, :] -= r[:]

    return Array


def rad_balls(r, g_r):
    n = g_r.shape[0]
    rad = np.array([])
    for i in range(n):
        if g_r[i] != 0.0:
            rad = np.append(rad, [r[i], g_r[i]])
    rad_n = rad.reshape([dc.int64(rad.shape[0] / 2), 2])
    return rad_n


def build_crystal(lattice, rep_a, rep_b, rep_c, dol_out='', ran=0, move_to_GC=1):
    """Given lattice vectors a, b, c and the number of repetitions of each vector, builds a .dol file at location
     dol_out (dol_out must contain both the location and the file name). If lattice constants are used,
     then the angles must be given in radians."""

    m = lattice.shape
    if m[0] == 6:
        ## Lattice constants to vectors
        a1, b1, c1, alpha, beta, gamma = lattice
        t = np.cos(beta) - np.cos(alpha) * np.cos(gamma)
        B = np.sqrt(np.sin(gamma) ** 2 - np.sin(gamma) ** 2 * np.cos(alpha) ** 2 - t ** 2)

        a = np.array([a1, 0, 0])
        b = np.array([b1 * np.cos(gamma), b1 * np.sin(gamma), 0])
        c = np.array([c1 * np.cos(alpha), c1 * t / np.sin(gamma), c1 * B / np.sin(gamma)])
    elif m[0] == 3:
        ## Lattice vectors
        a, b, c = lattice
    else:
        raise ValueError('Your lattice has to be either 3X3 or 1X6 i.e. X, Y, Z vectors, or a, b, c, alpha, beta, '
                         'gamma.')
    places = np.zeros([rep_a * rep_b * rep_c, 3])

    l = 0
    if ran == 0:
        ## No randomization
        for i in range(rep_a):
            for j in range(rep_b):
                for k in range(rep_c):
                    places[l] = i * a + j * b + k * c
                    l += 1
    else:
        ## Randomization
        change = 2 * ran * np.random.rand(rep_a * rep_b * rep_c, 3) - ran
        for i in range(rep_a):
            for j in range(rep_b):
                for k in range(rep_c):
                    places[l] = (i * a + a * change[l][0]) + (j * b + b * change[l][1]) + (k * c + c * change[l][2])
                    l += 1

    if move_to_GC:
        ## Move to geometric center
        places = MoveToGC(places)

    if dol_out:
        ## Write to .dol file
        if dol_out[-4:] != '.dol':
            dol_out += '.dol'

        with open(dol_out, 'w', newline='', encoding='utf-8') as file:
            dolfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            # dolfile.writerow(['## vec a = ', a, 'rep a = ', rep_a, 'vec b = ', b, 'rep b = ', rep_b, 'vec c = ', c, 'rep c = ', rep_c])
            for i in range(0, np.shape(places)[0]):
                dolfile.writerow([i, *places[i], 0, 0, 0])

    return places


def write_to_out(out_file, q, I):
    if out_file[-4:] != '.out' and out_file[-4] != '.':
        out_file += '.out'
    elif out_file[-4:] != '.out' and out_file[-4] == '.':
        out_file = out_file[:-3] + 'out'
        print('The function write_to_out only accepts .out file extensions, it has been changed')
    else:
        pass

    with open(out_file, 'w', newline='', encoding='utf-8') as file:
        outfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        # dolfile.writerow(['## q', 'I(q)'])
        for i in range(q.shape[0]):
            outfile.writerow([q[i], I[i]])
    return


def write_to_dol(dol_file, xyz):
    m, n = xyz.shape
    with open(dol_file, 'w+', newline='', encoding='utf-8') as file:
        dolfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        if n == 3:
            for i in range(m):
                dolfile.writerow([i, *xyz[i], 0, 0, 0])
        elif n == 4:
            for i in range(m):
                dolfile.writerow([i, *xyz[i][:-1], 0, 0, 0])
        elif n == 6:
            for i in range(m):
                dolfile.writerow([i, *xyz[i]])
        else:
            print('The size of the dol (xyz) is (%i, %i) but should be (%i, 3) or (%i, 6) instead' % (m, n, m, m))
            pass


def find_atm_rad(atm_type):

    atm_type = atm_type.replace(' ', '')
    rad_dic = {
    'H': 53, 'He': 31, 'Li': 167, 'Be': 112, 'B': 87, 'C': 67, 'N': 56, 'O': 48, 'F': 42, 'Ne': 38, 'Na': 190,
    'Mg': 145, 'Al': 118, 'Si': 111, 'P': 98, 'S': 88, 'Cl': 79, 'K': 243, 'Ca': 194, 'Sc': 184, 'Ti': 176,
    'V': 171, 'Cr': 166, 'Mn': 161, 'Fe': 156, 'Ni': 149, 'Cu': 145, 'Zn': 142, 'Ga': 136, 'Ge': 125, 'As': 114,
    'Se': 103, 'Br': 94, 'Kr': 88, 'Rb': 265, 'Sr': 219, 'Y': 212, 'Zr': 206, 'Nb': 198, 'Mo': 190, 'Tc': 183,
    'Ru': 178, 'Rh': 173, 'Pd': 169, 'Ag': 165, 'Cd': 161, 'In': 156, 'Sn': 145, 'Sb': 133, 'Te': 123, 'I': 115,
    'Xe': 108, 'Cs': 298, 'Ba': 253, 'La': 195, 'Ce': 186, 'Pr': 185, 'Nd': 184, 'Pm': 183, 'Sm': 181, 'Eu': 199,
    'Gd': 196, 'Tb': 194, 'Dy': 192, 'Ho': 191, 'Er': 189, 'Tm': 188, 'Yb': 187, 'Lu': 175, 'Hf': 167, 'Ta': 149,
    'W': 141, 'Re': 137, 'Os': 135, 'Ir': 136, 'Pt': 139, 'Au': 144, 'Hg': 150, 'Tl': 170, 'Pb': 146, 'Bi': 148,
    'Th': 180, 'Pa': 180, 'U': 175, 'Np': 175, 'Pu': 175, 'Am': 175, 'Cm': 174, 'Bk': 170, 'Cf': 170
}
    atm_rad = rad_dic[atm_type]
    return atm_rad / 1e3


def read_from_file(filename, r=-1):
    """Given a .dol or .pdb file, reads the file and returns a (Nx4) matrix with data [x, y, z, radius] of each atom.
    If the file is a .dol then the radius is as given in the function, if a .pdb then as given from function
    find_atm_rad."""

    if filename[-3:] == 'dol':
        try:
            with open(filename, encoding='utf-8') as file:
                try:
                    dol = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                    vec = np.array([])
                    for line in dol:
                        if type(line[0]) == str:
                            continue
                        vec = np.append(vec, line[1:4])
                        vec = np.append(vec, r)
                except:  ## Needed for dol files created with PDB units
                    dol = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                    vec = np.array([])
                    for line in dol:
                        if type(line[0]) == str:
                            continue
                        vec = np.append(vec, line[1:4])
                        vec = np.append(vec, r)
        except:
            with open(filename, encoding='utf-16') as file:
                dol = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                vec = np.array([])
                for line in dol:
                    if type(line[0]) == str:
                        continue
                    vec = np.append(vec, line[1:4])
                    vec = np.append(vec, r)

    elif filename[-3:] == 'pdb':
        with open(filename, encoding='utf-8') as pdb:
            vec = np.array([])
            for line in pdb:
                if (line[:6] == 'ATOM  ') | (line[:6] == 'HETATM'):
                    if r == -1:
                        atm_rad = find_atm_rad(line[76:78])
                    else:
                        atm_rad = r
                    # atm_rad = find_atm_rad(line[76:78])
                    vec = np.append(vec, [float(line[30:38]) / 10, float(line[38:46]) / 10, float(line[46:54]) / 10,
                                          atm_rad])
                else:
                    continue
    n = dc.int64(vec.shape[0] / 4)
    vec = np.reshape(vec, [n, 4])
    # print('Done reading file')
    return vec, n


def draw_symmetry(filepath):
    ## Not working yet
    import matplotlib.pyplot as plt

    lattice, _ = read_from_file(filepath)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(lattice[:, 0], lattice[:, 1], lattice[:, 2], 'k')
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_zlabel('z [nm]')
    plt.grid()
    plt.show()

    return


def to_1d(mat):
    sq = mat ** 2
    vec = np.sqrt(np.sum(sq, axis=1))
    return vec


def Lens_Vol(R, r, d):
    """
    Calculates the volume of the lens that is produced from the intersection of the atom with radius r and the search
    radius R of g(r). The atom is at distance d from the center of the search radius.
    """

    if d == 0.0:
        return 0.0
    Vol = (np.pi / (12 * d)) * (R + r - d) ** 2 * (d ** 2 + 2 * d * r - 3 * r ** 2 + 2 * d * R + 6 * r * R - 3 * R ** 2)
    return Vol


def N_R(r, dR):
    N_R = np.ceil(2 * r / dR) + 1
    return N_R


def balls_in_spheres(vec_triple, x_0, y_0, z_0, R_max):
    """Given a matrix vec_triple, a point of reference (x_0, y_0, z_0), and a radius R_max, returns the number of points
    that are inside the sphere."""

    m = np.shape(vec_triple)[0]
    dist = np.zeros(m)
    for i in range(m):
        dist[i] = np.sqrt(np.sum((vec_triple[i][:3] - np.array([x_0, y_0, z_0])) ** 2))
    num = np.sum((dist < R_max) & (dist > 1e-10))

    return num


def find_N(r, g_r, rho, r_min=0, r_max=0.56402):
    """Find the number of atoms inside the range (r_min, r_max), given r, g_r and density rho."""

    r_range = (r > r_min) & (r < r_max)
    N = simpson(rho * g_r[r_range] * 4 * np.pi * r[r_range] ** 2, r[r_range])

    return N

# @dc.program()
# def triple(mat_single: dc.float64[V, 4], Lx: dc.float64, Ly: dc.float64[1], Lz: dc.float64[1], file_triple: str = ''):
def triple(mat_single, size_or_reps, file_triple: str = '', cube=True, lattice_vecs=np.zeros(6), thermal: np.bool_ = False,
                   u: dc.float64[4] = np.array([0., 0., 0., 0.])):
    """Given a matrix of atoms, returns a matrix of all the atoms copied in the 27 cells around the central cell.
    If cube is True, the lattice is assumed to be cubic, and the size of the cell is given by size_or_reps. If cube is
    False, the lattice is assumed to be triclinic, size_or_reps is then the number of repetitions, and the lattice
    vectors are given by lattice_vecs."""

    # mat_single = mat_single[:, :3]
    if cube:
        ## If the lattice is cubic, we use the following algorithm
        ms = mat_single.shape[0]  #: dc.int64
        mat_triple = np.zeros([27 * ms, 4])
        Lx, Ly, Lz = size_or_reps[0], size_or_reps[1], size_or_reps[2]
        lx_t: dc.float64[3] = np.array([-Lx, 0., Lx])
        ly_t: dc.float64[3] = np.array([-Ly, 0., Ly])
        lz_t: dc.float64[3] = np.array([-Lz, 0., Lz])
        for dim_1 in range(3):
            for dim_2 in range(3):
                for dim_3 in range(3):
                    it = dc.int32((9 * dim_1 + 3 * dim_2 + dim_3) * ms)
                    itpn = dc.int32((9 * dim_1 + 3 * dim_2 + dim_3 + 1) * ms)
                    mat_triple[it:itpn] += mat_single + np.array([lx_t[dim_1], ly_t[dim_2], lz_t[dim_3], 0.0])
        if file_triple:
            write_to_dol(file_triple, mat_triple)
    else:
        new_repx, new_repy, new_repz = size_or_reps[0] * 3, size_or_reps[1] * 3, size_or_reps[2] * 3
        mat_triple = build_crystal(lattice_vecs, new_repx, new_repy, new_repz, file_triple)
        # mat_triple = np.append(mat_triple, np.zeros([mat_triple.shape[0], 1], dtype=int), axis=1)
    # for i in range(3):

    #     for j in range(3):
    #         for k in range(3):
    #             # if (trans[i] != 0) & (trans[j] != 0) & (trans[k] != 0):
    #             # new_loc[:] = np.add(mat_single, np.array([trans[i] * Lx, trans[j] * Ly, trans[k] * Lz, .0]))
    #             # mat_triple = np.append(mat_triple, new_loc, axis=0)
    #             # it = dc.int64((9 * i + 3 * j + k)*V)
    #             # itpv = dc.int64((9 * i + 3 * j + k + 1)*V)
    #             it = int((9 * i + 3 * j + k)*ms)
    #             itpn = int((9 * i + 3 * j + k + 1)*ms)
    #             mat_triple[it:itpn] = mat_single + np.array([trans[i] * Lx, trans[j] * Ly, trans[k] * Lz, .0])
    #             # mat_triple[it:itpv, :] = new_loc
    #             # mat_triple[it:it+n] = mat_single + np.array([trans[i] * Lx, trans[j] * Ly, trans[k] * Lz, 0.])
    #             # it += V
    #             # it = it + n
    if thermal:
        mat_triple = thermalize(np.copy(mat_triple), u)

    return mat_triple


@dc.program
def thermalize_dace(vec: dc.float64[V, U], u: dc.float64[U]):
    # TODO::Make this actually work well...
    # u_new: dc.float64[4]
    # if (np.size(u) != 1) & (np.size(u) != 4):
    #     u_new = np.array([u[0], u[1], u[2], 0])
    #     new_vec = np.random.normal(vec, u_new)  # Radius is still inside
    # elif U == 1:
    #     u_new = np.array([u, u, u, 0])
    #     new_vec = np.random.normal(vec, u_new)  # Radius is still inside
    # else:
    #   new_vec: dc.float64[V, 4] = np.zeros([V, U])
    new_vec = np.random.normal(vec, u)  # Radius is still inside
    return new_vec


def thermalize(vec, u):
    u_len = np.size(u)
    vec_len = np.size(vec, axis=1)
    u_new = np.array(u)
    if not (vec_len % u_len) & (vec_len != u_len):
        while np.size(u_new) < vec_len:
            u_new = np.append(u_new, u)
    else:
        u_new = np.append(u, 0)
    if (vec_len == 4) & (u_new[-1] != 0):
        u_new[-1] = 0
    new_vec = np.random.normal(vec, u_new)

    return new_vec


def different_atoms(file):
    with open(file, encoding='utf-8') as pdb:
        atom_list = np.array(['Fake'])
        atom_reps = np.array([0])
        for line in pdb:
            # print(line[:6], 'HETATM')
            if (line[:6] == 'ATOM  ') | (line[:6] == 'HETATM'):
                atm_type = line[76:78].replace(' ', '')
                if any(atm_type == atom_list):
                    atom_reps[atm_type == atom_list] += 1
                    # continue
                else:
                    atom_list = np.append(atom_list, atm_type)
                    atom_reps = np.append(atom_reps, 1)
                    with open(atm_type + r'.pdb', 'w', encoding='utf-8') as pdb:
                        changed_line = line[:30] + '   0.000   0.000   0.000' + line[54:]
                        pdb.write(changed_line)

    atom_list = atom_list[1:]
    atom_reps = atom_reps[1:]
    return atom_list, atom_reps


def fill_SF(q, theta, phi, dol_lat):
    from dplus.Amplitudes import sph2cart
    qs = sph2cart(q, theta, phi)
    N = len(dol_lat)
    some = 0
    for i in range(len(dol_lat)):
        w = float(np.dot(dol_lat[i], qs))
        some += np.exp(complex(0, w))
    return some/np.sqrt(N)


def Amp_of_SF(dol_filename, grid_size, q_max = 1, q_min = 0, ampj_filename=''):
    from dplus.Amplitudes import Amplitude
    a = Amplitude(grid_size, q_max, q_min)
    dol_f = read_from_file(dol_filename)[0][:, :3]
    a.fill(fill_SF, dol_f)
    
    if ampj_filename:
        if not ampj_filename[-5:] == '.ampj':
            ampj_filename += '.ampj'
        a.save(ampj_filename)
    return a


def fillmultigrid(q, theta, phi, SF, FF, N):
    if q > FF.helper_grid.q_max:
        ind = [SF.helper_grid.index_from_angles(q, theta, phi), FF.helper_grid.index_from_angles(q, theta, phi)]
        amps = SF._values
        am_s = complex(amps[ind[0]*2], amps[ind[0]*2 + 1])
        ampf = FF._values
        am_f = complex(ampf[ind[1]*2], ampf[ind[1]*2 + 1])
        return am_s * am_f * np.sqrt(N)
    try:
        return SF.get_interpolation(q, theta, phi) * FF.get_interpolation(q, theta, phi) * np.sqrt(N)
    except:
        theta = np.pi - 10e-15
        return SF.get_interpolation(q, theta, phi) * FF.get_interpolation(q, theta, phi) * np.sqrt(N)


def Amp_multi(SF_Ampj, FF_Ampj, filename='', N = 1):
    from dplus.Amplitudes import Amplitude
    # if not SF_Ampj[-5:] == '.ampj':
        # SF_Ampj += '.ampj'
    # if not FF_Ampj[-5:] == '.ampj':
        # FF_Ampj += '.ampj'

    # SF = Amplitude.load(SF_Ampj)
    # FF = Amplitude.load(FF_Ampj)
    grid_min_size = np.max([SF_Ampj.helper_grid.grid_size, FF_Ampj.helper_grid.grid_size])
    q_min_size = np.max([SF_Ampj.helper_grid.q_min, FF_Ampj.helper_grid.q_min])
    q_max_size = np.min([SF_Ampj.helper_grid.q_max, FF_Ampj.helper_grid.q_max])
    multi_amp = Amplitude(grid_min_size, q_max_size, q_min_size)
    multi_amp.fill(fillmultigrid, SF_Ampj, FF_Ampj, N)
    if filename:
        if not filename[-5:] == '.ampj':
            filename += '.ampj'
        multi_amp.save(filename)
    return multi_amp


def fillsumgrid(q, theta, phi, SF, FF):
    if q > FF.helper_grid.q_max:
        ind = [SF.helper_grid.index_from_angles(q, theta, phi), FF.helper_grid.index_from_angles(q, theta, phi)]
        amps = SF._values
        am_s = complex(amps[ind[0]*2], amps[ind[0]*2 + 1])
        ampf = FF._values
        am_f = complex(ampf[ind[1]*2], ampf[ind[1]*2 + 1])
        return am_s + am_f
    try:
        return SF.get_interpolation(q, theta, phi) + FF.get_interpolation(q, theta, phi) 
    except:
        theta = np.pi - 10e-15
        return SF.get_interpolation(q, theta, phi) + FF.get_interpolation(q, theta, phi) 


def Amp_sum(amp1, amp2, filename=''):
    from dplus.Amplitudes import Amplitude

    grid_min_size = np.max([amp1.helper_grid.grid_size, amp2.helper_grid.grid_size])
    q_min_size = np.max([amp1.helper_grid.q_min, amp2.helper_grid.q_min])
    q_max_size = np.min([amp1.helper_grid.q_max, amp2.helper_grid.q_max])
    addition_amp = Amplitude(grid_min_size, q_max_size, q_min_size)
    addition_amp.fill(fillsumgrid, amp1, amp2)

    if filename:
        if not filename[-5:] == '.ampj':
            filename += '.ampj'
        addition_amp.save(filename)

    return addition_amp

def fillmultigrid(q, theta, phi, SF, FF, N):
    if q > FF.helper_grid.q_max:
        ind = [SF.helper_grid.index_from_angles(q, theta, phi), FF.helper_grid.index_from_angles(q, theta, phi)]
        amps = SF._values
        am_s = complex(amps[ind[0]*2], amps[ind[0]*2 + 1])
        ampf = FF._values
        am_f = complex(ampf[ind[1]*2], ampf[ind[1]*2 + 1])
        return am_s * am_f * np.sqrt(N)
    try:
        return SF.get_interpolation(q, theta, phi) * FF.get_interpolation(q, theta, phi) * np.sqrt(N)
    except:
        theta = np.pi - 10e-15
        return SF.get_interpolation(q, theta, phi) * FF.get_interpolation(q, theta, phi) * np.sqrt(N)


def S_Q_from_I(I_q, f_q, N):
    """Given the intensity I_q, the subunit form-factor f_q, and number of subunits N, returns the structure factor.
    I_q and f_q have to be np.arrays. From CalculationResult, this can be attained through
     np.array(list(calc_result.graph.values())) or if from signal then np.array(calc_input.y).
     If one inputs either a CalculationResult or a signal, the function will convert it into an np.array."""

    if (type(I_q) != np.ndarray) & (type(I_q) == tuple):
        # print('Switched tuple to array')
        I_q = np.array(I_q)
    elif (type(I_q) != np.ndarray) & (type(I_q) != tuple):
        I_q = np.array(list(I_q))
        # print('Switched dict to array')
    if (type(f_q) != np.ndarray) & (type(f_q) == tuple):
        # print('Switched tuple to array')
        f_q = np.array(f_q)
    elif (type(f_q) != np.ndarray) & (type(f_q) != tuple):
        f_q = np.array(list(f_q))
        # print('Switched dict to array')

    Nf2 = N * f_q
    S_q = I_q / Nf2

    return S_q


def _model_coordinates(filename, r=-1):
    """Accepts either a .dol/.pdb filename or a lattice matrix (the content of a .dol, an Nx3 or Nx4 array)
    and returns an (Nx3) coordinate matrix together with the number of points."""

    if isinstance(filename, (str, bytes, os.PathLike)):
        r_mat, n = read_from_file(os.fspath(filename), r)
        return np.ascontiguousarray(r_mat[:, :3], dtype=np.float64), n

    r_mat = np.atleast_2d(np.asarray(filename, dtype=np.float64))
    if (r_mat.ndim != 2) | (r_mat.shape[1] < 3):
        raise ValueError('A lattice matrix must be an (Nx3) or (Nx4) array of [x, y, z(, radius)] rows, got shape '
                         + str(np.shape(filename)) + '.')

    return np.ascontiguousarray(r_mat[:, :3], dtype=np.float64), dc.int64(r_mat.shape[0])


def S_Q_from_model_slow(filename: str, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                        , thermal: np.bool_ = False, Number_for_average_conf: dc.int64 = 1, u=np.array([0, 0, 0]),
                        conv_eps: dc.float64 = 1e-6, min_iter: dc.int64 = 10, check_step: dc.int64 = 5):
    """Given a .dol or .pdb filename and a q-range, returns the orientation averaged structure factor."""

    r_mat, n = read_from_file(filename)
    r_mat = r_mat[:, :3]
    q = np.arange(q_min, q_max + dq, dq)
    S_Q = np.zeros([Number_for_average_conf, len(q)])
    S_Q += n
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    if thermal:
        r_mat_old = np.copy(r_mat)
        check_matrix = np.zeros([4, len(q)])

    while it < Number_for_average_conf:
        # print('Finished iteration ' + str(it + 1) + ' of ' + str(Number_for_average_conf) + ' iterations.')
        if thermal:
            r_mat[:] = thermalize(np.copy(r_mat_old), u)

        for i in range(n - 1):
            r_i = r_mat[i]
            if i == 0:
                r = np.sqrt(np.sum(r_i ** 2))
                if r > R[it]:
                    R[it] = r
            for j in range(i + 1, n):
                r_j = r_mat[j]
                if i == 0:
                    r = np.sqrt(np.sum(r_j ** 2))
                    if r > R[it]:
                        R[it] = r
                r = np.sqrt(np.sum((r_i - r_j) ** 2))
                qr = q * r

                if q_min==0:
                    S_Q[it][0] += 2
                    S_Q[it][1:] += 2 * np.sin(qr[1:]) / qr[1:]
                else:
                    S_Q[it] += 2 * np.sin(qr) / qr

        S_Q[it] /= n
        R[it] /= 2
        rho += 3 * S_Q[it][0] / (4 * np.pi * R[it] ** 3)

        if it >= min_iter:
            if it % check_step == 0:
                print("checking convergence at iteration", it)
                ind = ((it - min_iter) // check_step) % 4
                S_Q_temp = np.sum(S_Q[:it+1], axis=0) / (it + 1)
                if it <= min_iter + 3 * check_step:
                    check_matrix[ind, :] = S_Q_temp
                    it += 1
                    continue

                conv_test = np.max(np.abs(1 - S_Q_temp / check_matrix), axis=-1)
                if np.any(conv_test<conv_eps):
                    print('Convergence reached at iteration', it + 1)
                    S_Q[:] = np.sum(S_Q[:it+1], axis=0) / (it + 1)
                    rho /= (it + 1)
                    return q, S_Q[0], rho
                else:
                    check_matrix[ind, :] = S_Q_temp
        it += 1

    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    rho /= Number_for_average_conf

    return q, S_Q[0], rho


def _thermal_sigma(vec_len, u):
    """Expands the displacement std ``u`` to one value per coordinate column, exactly as thermalize does."""
    u_len = np.size(u)
    u_new = np.array(u, dtype=np.float64)
    if not (vec_len % u_len) & (vec_len != u_len):
        while np.size(u_new) < vec_len:
            u_new = np.append(u_new, u)
    else:
        u_new = np.append(u_new, 0)
    if (vec_len == 4) & (u_new[-1] != 0):
        u_new[-1] = 0

    return u_new


def S_Q_from_model_GPU(filename, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                       , thermal: np.bool_ = False, Number_for_average_conf: dc.int64 = 1, u=np.array([0, 0, 0]),
                       conv_eps: dc.float64 = 1e-6, min_iter: dc.int64 = 10, check_step: dc.int64 = 5,
                       mem_fraction: dc.float64 = 0.25):
    """Given a model and a q-range, returns the orientation averaged structure factor.

    ``filename`` is either the path of a .dol or .pdb file, or a lattice matrix (the content of a .dol,
    an Nx3 or Nx4 array of [x, y, z(, radius)] rows).

    Same math as S_Q_from_model_slow, but the O(n^2) pair loop is vectorized on the GPU with cupy.
    The pairwise distances are built in row blocks and the sin(qr)/qr sum in pair blocks, so the
    temporaries never exceed ``mem_fraction`` of the free GPU memory. Returns numpy arrays."""

    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError('S_Q_from_model_GPU needs cupy (e.g. "pip install cupy-cuda12x"). '
                          'Use S_Q_from_model_slow for a CPU-only run.') from e

    r_mat, n = _model_coordinates(filename)
    q = np.arange(q_min, q_max + dq, dq)
    q_len = q.shape[0]

    q_g = cp.asarray(q)
    r_g = cp.asarray(r_mat)
    S_Q = cp.full([Number_for_average_conf, q_len], float(n))
    check_matrix = cp.zeros([4, q_len])
    R = np.zeros(Number_for_average_conf)
    rho = 0.0

    if thermal:
        r_g_old = r_g.copy()
        u_g = cp.asarray(_thermal_sigma(r_mat.shape[1], u))

    # Block sizes: a row block holds rb * n doubles, a pair block pc * q_len doubles.
    free_bytes = cp.cuda.Device().mem_info[0]
    budget = max(int(free_bytes * mem_fraction), 1 << 20)
    row_chunk = max(1, int(budget // (8 * n)))
    pair_chunk = max(1, int(budget // (8 * q_len)))

    it = 0
    if Number_for_average_conf>1:
        print("Starting simulation...")

    while it < Number_for_average_conf:
        if thermal:
            r_g = r_g_old + u_g * cp.random.standard_normal(r_g_old.shape)

        sq_norm = (r_g ** 2).sum(axis=1)
        R[it] = float(cp.sqrt(sq_norm.max()))

        pair_sum = cp.zeros(q_len)
        for a in range(0, n - 1, row_chunk):
            b = min(a + row_chunk, n - 1)
            # |r_i - r_j|^2 = |r_i|^2 + |r_j|^2 - 2 r_i.r_j, upper triangle only (j > i)
            d2 = sq_norm[a:b, None] + sq_norm[None, a + 1:] - 2 * (r_g[a:b] @ r_g[a + 1:].T)
            keep = cp.arange(a + 1, n)[None, :] > cp.arange(a, b)[:, None]
            d = cp.sqrt(cp.maximum(d2[keep], 0))
            for p in range(0, d.shape[0], pair_chunk):
                d_p = d[p:p + pair_chunk]
                # cp.sinc(x) = sin(pi x) / (pi x), so this is sin(qr)/qr with the qr = 0 limit built in
                pair_sum += 2 * cp.sinc(d_p[:, None] * q_g[None, :] / np.pi).sum(axis=0)

        S_Q[it] = (S_Q[it] + pair_sum) / n
        R[it] /= 2
        rho += 3 * float(S_Q[it][0]) / (4 * np.pi * R[it] ** 3)

        if it >= min_iter:
            if it % check_step == 0:
                # print("checking convergence at iteration", it)
                ind = ((it - min_iter) // check_step) % 4
                S_Q_temp = S_Q[:it + 1].sum(axis=0) / (it + 1)
                if it <= min_iter + 3 * check_step:
                    check_matrix[ind, :] = S_Q_temp
                    it += 1
                    continue

                conv_test = cp.max(cp.abs(1 - S_Q_temp / check_matrix), axis=-1)
                if bool(cp.any(conv_test < conv_eps)):
                    print('Convergence reached at iteration', it + 1)
                    S_Q_fin = S_Q[:it + 1].sum(axis=0) / (it + 1)
                    rho /= (it + 1)
                    return q, cp.asnumpy(S_Q_fin), rho
                else:
                    check_matrix[ind, :] = S_Q_temp
        it += 1

    S_Q_fin = S_Q.sum(axis=0) / Number_for_average_conf
    rho /= Number_for_average_conf

    return q, cp.asnumpy(S_Q_fin), rho


def S_Q_from_model(filename, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                   , thermal: np.bool_ = False, Number_for_average_conf: dc.int64 = 1,
                   u: dc.float64[3] = np.array([0.,0.,0.]), use_GPU: np.bool_ = True):
    """Given a model and a q-range, returns the orientation averaged structure factor.

    ``filename`` is either the path of a .dol or .pdb file, or a lattice matrix (the content of a .dol,
    an Nx3 or Nx4 array of [x, y, z(, radius)] rows)."""

    r_mat, n = _model_coordinates(filename)
    r_mat = np.copy(r_mat)
    q = np.arange(q_min, q_max + dq/2, dq)
    if thermal:
        r_mat_old = r_mat
    q_len = q.shape[0]
    # if thermal:
    #     r_mat_old = r_mat
    S_Q = n * np.ones([Number_for_average_conf, q_len])
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    while it < Number_for_average_conf:
        print('Finished iteration ' + str(it + 1) + ' of ' + str(Number_for_average_conf) + ' iterations.')
        if thermal:
            r_mat[:] = thermalize(np.copy(r_mat_old), u)
        S_Q_pretemp = np.copy(S_Q[it])
        R[it], S_Q[it, :] = compute_sq(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        # if Number_for_average_conf > 1:
        # if use_GPU:
        #     R[it], S_Q[it, :] = compute_sq_GPU(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        # else:
        #     R[it], S_Q[it, :] = compute_sq_CPU(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        it += 1
    R_fin = np.max(R) / 2
    S_Q /= n
    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    rho += 3 * S_Q[0, 0] / (4 * np.pi * R_fin ** 3)
    rho /= Number_for_average_conf

    return q, S_Q[0], rho

def S_Q_average_box(xyz, qmax, q_points, size_min, size_max, mean, sigma, axes=np.array([1, 1, 1], dtype=np.bool_),
                    default_rep=np.array([0, 0, 0]), file_path=r'.\S_Q_average', qmin=0, normalize=True, make_fig=True,
                    num_of_s_q_shown=-1, slow=False, thermal=False, u=np.array([0, 0, 0]), verbose=False):
    # TODO: Find a way to make this work with GPU

    num_s_q = int(size_max - size_min)
    if num_of_s_q_shown < 0:
        num_of_s_q_shown = num_s_q
    S_q_all = np.zeros([num_s_q, q_points+1])
    dqn= (qmax - qmin) / (q_points - 1)
    if (file_path[-3:] == 'dol') | (file_path[-3:] == 'out'):
        file_path = file_path[:-4]

    if slow:
        for i in range(size_min, size_max):
            ind = int(i - size_min)
            size = i * axes + default_rep
            dol_name = file_path + '\\box_size_' + str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) + r'.dol'
            if not os.path.exists(dol_name):
                build_crystal(xyz, *size, dol_name)
            q, s_q_temp, _ = S_Q_from_model_slow(dol_name, q_min=qmin, q_max=qmax, thermal=thermal, u=u)
            if normalize:
                S_q_all[ind] = s_q_temp / s_q_temp[0]
                write_to_out(file_path + '\\box_size_' +str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) +'.out', q, S_q_all[ind])
            else:
                S_q_all[ind] = s_q_temp
                write_to_out(
                    file_path + '\\box_size_' + str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) + '_no_normalization.out', q,
                    S_q_all[ind])
            if not verbose:
                continue
            else:
                print('Done with', i)
    else:
        for i in range(size_min, size_max):
            ind = int(i - size_min)
            size = i * axes + default_rep
            dol_name = file_path + '\\box_size_' + str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) + r'.dol'
            if not os.path.exists(dol_name):
                build_crystal(xyz, *size, dol_name)
            q, s_q_temp, _ = S_Q_from_model(dol_name, q_min=qmin, q_max=qmax, dq=dqn, thermal=thermal, u=u)#,use_GPU=use_GPU)
            if normalize:
                S_q_all[ind] = s_q_temp / s_q_temp[0]
                write_to_out(file_path + '\\box_size_' + str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) + '.out', q, S_q_all[ind])
            else:
                S_q_all[ind] = s_q_temp
                write_to_out(
                    file_path + '\\box_size_' + str(size[0]) + '_' + str(size[1]) + '_' + str(size[2]) + '_no_normalization.out', q,
                    S_q_all[ind])
            if not verbose:
                continue
            else:
                print('Done with', i)

    weight = stats.norm.pdf(range(size_min, size_max), mean, sigma)
    weight /= weight.sum()
    S_q_final = np.average(S_q_all, weights=weight, axis=0)
    if normalize:
        out_path = file_path +'\\no_box_normalized.out'
    else:
        out_path = file_path +'\\no_box_no_normalization.out'
    write_to_out(out_path, q, S_q_final)

    if make_fig:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.semilogy(q, S_q_final, lw=3, label='averaged')
        for k in range(num_s_q-num_of_s_q_shown, num_s_q):
            ax2.semilogy(q, S_q_all[k], label=str(k + size_min), lw=3)
        ax1.legend(fontsize=14, loc='upper right')
        ax2.legend(fontsize=14, loc='upper right', ncols=2)
        ax2.set_xlabel('q $[nm^{-1}]$', size=14)
        ax1.set_ylabel('S(q) [a.u.]', size=14)
        ax2.set_ylabel('S(q) [a.u.]', size=14)
        ax1.tick_params(which='both', labelsize=11, color='k')
        ax2.tick_params(which='both', labelsize=11, color='k')
        plt.savefig(file_path + '\\final_averaged.png')

    return q, S_q_final, S_q_all


def s_q_from_g_r(r, g_r, rho, q_min=0, q_max=50, dq=0.01, factor=1, type='Simpson'):
    """Given an r-vector r, a g(r) g_r, and a density rho, returns the structure factor in one of two ways: 'DST' or
     'Simpson' as given in type."""
    if type == 'DST':
        n = r.shape[0] * factor
        R = np.linspace(r[0], r[-1], n)
        dr = (max(R) - min(R)) / n  # R[1] - R[0]
        q = fftfreq(n, dr)[1:n // 2] * 2 * np.pi
        G_R = np.interp(R, r, g_r)
        I = dst(4 * np.pi * rho * G_R * R, type=1, norm='ortho')

        if factor % 2:
            s_q = 1 + 1 / q * I[1:-3:2]
        else:
            s_q = 1 + 1 / q * I[:-2:2]

        q_range = (q > q_min) & (q < q_max)
        return q[q_range], s_q[q_range]  # q, s_q

    elif type == 'Simpson':
        q = np.linspace(q_min, q_max, dc.int64((q_max - q_min) / dq) + 1)
        qr = r * np.reshape(q, [q.shape[0], 1])

        I = simpson(g_r * r * np.sin(qr), r)
        s_q = I * (4 * np.pi * rho)
        s_q[q != 0] /= (q[q != 0])
        s_q += 1

        return q, s_q


_MODEL_ALIASES = {  # accepted spellings of the two disorder models, normalised to one key each
    'well': 'well', 'wells': 'well', 'einstein': 'well',      # 'einstein' = the old function name
    'spring': 'spring', 'springs': 'spring', 'corr': 'spring',  # 'corr' = the old function name
    'hooke': 'spring',                                        # the name g_r.MC_Sim uses
}


def _resolve_model(model):
    """Normalise the `model` argument to 'well' or 'spring', or raise if it is neither.

    The two models differ only in how the variance of a pair separation grows with the distance
    between the two atoms, which is the single statement that separates disorder of the first kind
    from a harmonic solid; see structure_factors_DW.tex, Secs. 2 and 3.
    """
    key = str(model).strip().lower()             # tolerate 'Well', 'SPRINGS', stray whitespace
    if key not in _MODEL_ALIASES:                # an unknown model must never fall through silently
        raise ValueError("model must be one of %r, got %r"
                         % (sorted(set(_MODEL_ALIASES)), model))
    return _MODEL_ALIASES[key]


def _pair_variance(model, sigma, r_n, n_steps=None, dim=None, cell=None):
    """The variance sigma_n^2 of the scalar separation of a pair, in the units of `sigma` squared.

    This is the only place the two models differ, and the only place the dimensionality enters.

        well    sigma_n^2 = 2 sigma^2, whatever the separation. Each atom sits in its own harmonic
                well independently of every other, so the relative coordinate of two atoms carries
                sigma^2 + sigma^2. `sigma` is therefore the r.m.s. displacement of ONE ATOM.

        spring  sigma_n^2 grows with the separation, because the two atoms are tied to each other
                through the intervening bonds rather than to fixed sites. `sigma` is the r.m.s.
                fluctuation of ONE NEAREST-NEIGHBOUR SPACING, i.e. sqrt(k_B T / C), so in 1D the
                one-step variance is sigma^2 and not 2 sigma^2.

    The two meanings of `sigma` are not interchangeable: passing the same number to both models
    compares a site r.m.s. against a bond r.m.s. and the curves differ by more than the models do.

    The spring growth laws are the standard harmonic-lattice results (Landau-Peierls):

        1D  sigma_n^2 = n sigma^2                                   linear, no long-range order
        2D  sigma_n^2 = sigma^2 [ln(r_n / eta_0) - Ci(k_d r_n)]     logarithmic, quasi-long-range
        3D  sigma_n^2 = 2 sigma^2 [1 - Si(k_D r_n) / (k_D r_n)]     saturating, long-range order

    with k_d = sqrt(4 pi / A_cell), eta_0 = 2 / (e^gamma k_d) in 2D and k_D = (6 pi^2 / V_cell)^(1/3)
    the Debye wavevector in 3D. The 3D form saturates at 2 sigma^2 as r_n -> infinity, which is why
    a three-dimensional crystal keeps Bragg peaks attenuated by a Debye-Waller factor while the
    lower-dimensional ones broaden away.

    :param model: 'well' or 'spring', already normalised by _resolve_model
    :param sigma: see above - a site r.m.s. for 'well', a bond r.m.s. for 'spring'
    :param r_n: separation(s), same units as the lattice
    :param n_steps: number of lattice steps between the pair; 1D spring only
    :param dim: 1, 2 or 3; ignored by the well model
    :param cell: cell area (2D) or volume (3D), needed by the spring model above 1D

    References
    ----------
    R. E. Peierls, Ann. Inst. Henri Poincare 5, 177 (1935); L. D. Landau, Phys. Z. Sowjetunion 11,
        26 (1937) -- the dimensional hierarchy of positional order.
    A. Guinier, X-Ray Diffraction (Freeman, 1963), Ch. 9 -- disorder of the first and second kind.
    M. Born and K. Huang, Dynamical Theory of Crystal Lattices (OUP, 1954), Ch. 5 -- <u u> from the
        harmonic Hamiltonian by equipartition.
    """
    if model == 'well':                          # flat: the two wells know nothing about each other
        return 2.0 * sigma**2 * np.ones_like(np.asarray(r_n, dtype=np.float64))

    if dim == 1:                                 # a chain is a tree, so variances add along the path
        return n_steps * sigma**2

    if dim == 2:                                 # k_d from the cell area, eta_0 the short-range cutoff
        k_d = np.sqrt(4 * np.pi / cell)
        eta_0 = 1 / (egam * k_d)
        return sigma**2 * (np.log(np.asarray(r_n, dtype=np.float64) / eta_0))

    if dim == 3:                                 # k_D the Debye wavevector of the same cell
        k_D = np.power(6 * np.pi**2 / cell, 1 / 3)
        kdrn = k_D * np.asarray(r_n, dtype=np.float64)
        si, _ = sici(kdrn)
        return 2 * sigma**2 * (1 - si / kdrn)

    raise ValueError('dim must be 1, 2 or 3 for the spring model, got %r' % (dim,))


def _pair_kernel(q, r_n, sigma_n_sq):
    """<sinc(q x)> for a scalar separation x ~ Normal(mean r_n, variance sigma_n_sq).

    This is the one kernel every analytic routine in this module is built from; the models differ
    only in what they hand it as `sigma_n_sq`. Averaging sinc(qx) over the Gaussian gives

        I(q) = sqrt(pi / (2 v)) / q
               * Re[ exp(-r^2 / (2 v)) - exp(-v q^2 / 2 - 1j q r) * erfcx((v q + 1j r) / sqrt(2 v)) ]

    with r = r_n and v = sigma_n_sq. It is exact, not a small-fluctuation (decoupled Debye-Waller)
    approximation: the whole distribution of the separation is integrated, not just its mean.

    The erfcx form is used because the algebraically equivalent exp(-r^2 / 2v) * Re[erf(z)] is
    numerically unusable - erf(z) needs an intermediate of order exp(Im(z)^2), which overflows
    double precision once Im(z) = r / sqrt(2 v) exceeds ~26.6, while the prefactor underflows to
    zero, so the product returns NaN. The identity erf(z) = 1 - exp(-z^2) erfcx(z) cancels the
    divergent factor analytically.

    q = 0 is handled exactly: sinc(0) = 1 for every separation, so the kernel is 1 there whatever
    the variance. sigma_n_sq = 0 falls back to the bare sinc.

    :param q: (n_q,) wavevector magnitudes; q = 0 may be included
    :param r_n: (M,) mean separations, or a scalar
    :param sigma_n_sq: (M,) variances of those separations, or a scalar; broadcast against r_n
    :return: (M, n_q) array of kernel values

    References
    ----------
    P. Debye, Ann. Phys. 351, 809 (1915) -- the orientational average giving sinc.
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions, Sec. 7.1 -- erfcx.
    """
    q = np.asarray(q, dtype=np.float64)
    r_n, v = np.broadcast_arrays(np.atleast_1d(np.asarray(r_n, dtype=np.float64)),
                                 np.atleast_1d(np.asarray(sigma_n_sq, dtype=np.float64)))

    if np.any(v < 0):                            # a negative variance means the growth law was used
        raise ValueError('negative pair variance: the spring variance law was evaluated outside '   # outside its domain
                         'its domain (smallest value %.6g at r_n = %.6g). In 2D this happens when '
                         'the separation falls below the cutoff eta_0' % (v.min(), r_n[np.argmin(v)]))

    out = np.ones([r_n.size, q.size])            # q = 0 columns keep the exact value sinc(0) = 1
    nz = q != 0
    if not np.any(nz):
        return out

    zero_v = v == 0                              # rigid lattice: the Gaussian collapses to a delta
    if np.any(zero_v):
        out[np.ix_(zero_v, nz)] = np.sinc(np.outer(r_n[zero_v], q[nz]) / np.pi)  # np.sinc is sin(pi x)/(pi x)

    fin = ~zero_v
    if np.any(fin):
        r = r_n[fin][:, None]
        vv = v[fin][:, None]
        qq = q[None, nz]
        pref = np.sqrt(np.pi / (2 * vv)) / qq
        body = (np.exp(-r**2 / (2 * vv))
                - np.exp(-vv * qq**2 / 2 - 1j * qq * r)
                * erfcx((vv * qq + 1j * r) / np.sqrt(2 * vv)))
        out[np.ix_(fin, nz)] = pref * np.real(body)

    return out


def _lattice_vectors(lattice):
    """Three lattice vectors (3, 3) from either vectors or the six lattice parameters.

    Same two input forms and the same conversion as dplus.g_r.build_crystal, so a lattice handed to
    both describes the same crystal:

        (3, 3)   rows a, b, c
        (6,)     a, b, c, alpha, beta, gamma, angles in RADIANS, converted as

                     a_vec = (a, 0, 0)
                     b_vec = (b cos(gamma), b sin(gamma), 0)
                     c_vec = (c cos(alpha), c t / sin(gamma), c B / sin(gamma))
                     t = cos(beta) - cos(alpha) cos(gamma)
                     B = sqrt(sin^2(gamma) - sin^2(gamma) cos^2(alpha) - t^2)

    Note that build_crystal takes alpha as the angle between a and c and beta as the one between b
    and c - the crystallographic convention has these two swapped. It is kept here on purpose, so
    the analytic curve and the .dol model agree; it only matters for a cell with alpha != beta.

    A zero length (a, b or c = 0) gives a zero vector, which is how build_crystal writes a chain
    ([0, 0, d, ...]) or a sheet ([0, d, d, ...]).

    References
    ----------
    International Tables for Crystallography, Vol. B (2006), Sec. 1.1.1 -- the parameter-to-vector
        conversion, with the alpha/beta labels as noted above.
    """
    lattice = np.asarray(lattice, dtype=np.float64)          # accept lists as well as arrays
    if lattice.shape == (3, 3):                  # rows are already the vectors a, b, c
        return lattice.copy()
    if lattice.shape == (6,) or lattice.shape == (1, 6):     # a, b, c, alpha, beta, gamma
        a1, b1, c1, alpha, beta, gamma_ = lattice.ravel()
        t = np.cos(beta) - np.cos(alpha) * np.cos(gamma_)    # as in build_crystal
        B = np.sqrt(np.sin(gamma_) ** 2 - np.sin(gamma_) ** 2 * np.cos(alpha) ** 2 - t ** 2)
        return np.array([[a1, 0, 0],                                                 # a along x
                         [b1 * np.cos(gamma_), b1 * np.sin(gamma_), 0],              # b in the xy plane
                         [c1 * np.cos(alpha), c1 * t / np.sin(gamma_), c1 * B / np.sin(gamma_)]])
    raise ValueError('lattice must be 3x3 vectors or 6 lattice parameters, got shape %r'
                     % (lattice.shape,))


def s_q_1d_analytic(q, N, d, sigma, model='spring'):
    '''
    Orientationally averaged structure factor of a finite 1D lattice under either disorder model,
    in exact closed form.

    Computes S(q) for a freely tumbling chain of N identical scatterers on a lattice of spacing d.
    The Debye equation collapses to a single weighted sum over step separations, since every pair
    n steps apart has the same mean separation n*d and the same variance:

        S(q) = 1 + 2 * sum_{n=1}^{N-1} W_n * I_n(q),      W_n = 1 - n/N

    W_n is the finite-chain pair multiplicity, since an N-site chain contains 2(N-n) ordered pairs
    at step separation n. I_n is the Gaussian-averaged sinc of _pair_kernel, and the model enters
    only through the variance it is given:

        model='spring'   sigma_n^2 = n sigma^2      atoms tied to each other by harmonic springs,
                                                    V = sum_i (C/2)[(x_{i+1} - x_i) - d]^2, so each
                                                    bond is an independent Gaussian of variance
                                                    sigma^2 = k_B T / C and the variances add along
                                                    the chain. Disorder of the second kind; in 1D
                                                    this coincides exactly with the paracrystal.

        model='well'     sigma_n^2 = 2 sigma^2      atoms held to their own sites independently,
                                                    V = sum_n (k/2)|r_n - R_n|^2 with
                                                    sigma^2 = k_B T / k. The relative coordinate of
                                                    two atoms carries sigma^2 + sigma^2 whatever the
                                                    separation. Disorder of the first kind, the
                                                    Einstein crystal.

    This is exact, not a small-fluctuation (decoupled Debye-Waller) approximation; see _pair_kernel.
    For the decoupled forms of the same two models see s_q_decoupled and s_q_DW_wells.

    Parameters
    ----------
    q : ndarray, shape (n_q,)
        Wavevector magnitudes, in reciprocal units of `d`. Must be a 1-D NumPy array: a list or
        bare float will fail, since the routine uses len(q) and boolean masking. q = 0 may be
        included and is handled exactly.
    N : int
        Number of scattering sites (N >= 2).
    d : float
        Equilibrium bond length (lattice constant).
    sigma : float
        MEANING DEPENDS ON `model`, and the two are not interchangeable:
          model='spring'  standard deviation of a SINGLE BOND, sqrt(k_B T / C) in the classical
                          limit, so the one-step variance is sigma^2.
          model='well'    standard deviation of a SINGLE ATOM about its own site, sqrt(k_B T / k),
                          so every pair variance is 2 sigma^2.
        In the same units as `d`, and >= 0. sigma = 0 returns the rigid lattice. Only the variance
        enters, so a quantum <u^2> may be substituted directly.
    model : {'spring', 'well'}
        Which disorder model to evaluate. 'springs', 'wells', 'hooke', and the old function-name
        aliases 'corr' and 'einstein' are also accepted.

    Returns
    -------
    ndarray, shape (n_q,)
        S(q), normalised so that S(0) = N and S(q) -> 1 as q -> infinity.

    Notes
    -----
    Assumptions: identical unit scatterers (no atomic form factors; for polyatomic systems each
    term needs f_i(q) f_j(q) and normalisation by sum_i |f_i(q)|^2); harmonic, classical, hence
    Gaussian disorder; and strictly 1-D geometry, so relative displacements are purely longitudinal
    and the transverse variance vanishes. In 2D/3D the transverse channel contributes; use
    s_q_2d_analytic or s_q_3d_analytic.

    For 'spring' the bonds are taken as independent nearest-neighbour Gaussians, so distant sites
    are correlated only through the intervening bonds. That is exact on a chain, which is a tree
    with one path between any two sites, and exact only there - see s_q_DW_springs for what
    replaces it in 2D and 3D.

    The Gaussian model assigns weight Phi(-r_n / sigma_n) to unphysical reversed separations; this
    is exactly the exp(-r_n^2 / 2 sigma_n^2) term in the kernel, and is below 1e-22 for
    sigma/d <= 0.1.

    Cost is O(N * n_q): the double sum over pairs collapses to O(N) kernel evaluations per
    wavevector. An (N-1, n_q) complex array is built internally, about 16 * (N-1) * n_q bytes per
    temporary; chunk over `q` if that is too large.

    Cheap invariants, worth asserting in calling code:
        S(0) == N                      exact sum rule
        S(q) -> 1 at large q           the incoherent floor
        |I_n(q)| <= 1 for all n, q     I_n is an average of sinc

    Examples
    --------
    >>> s_q_1d_analytic(np.array([0.0, 1.0, 2.0]), 6, 1.0, 0.1)
    array([6.        , 2.87066275, 1.53138033])

    >>> s_q_1d_analytic(np.array([0.0]), 50, 1.0, 0.05)          # sum rule: S(0) = N
    array([50.])

    >>> s_q_1d_analytic(np.array([50.0]), 50, 1.0, 0.05)         # large-q floor: S -> 1
    array([0.99932465])

    References
    ----------
    P. Debye, Ann. Phys. 351, 809 (1915) -- the orientational average.
    B. E. Warren, X-ray Diffraction (Dover, 1990), Ch. 3 and 11.
    R. Hosemann and S. N. Bagchi, Direct Analysis of Diffraction by Matter (North-Holland, 1962)
        -- disorder of the second kind, which 'spring' reproduces in 1D.
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions, Sec. 7.1 -- erfc and erfcx.
    '''
    model = _resolve_model(model)                # 'Well' / 'springs' / 'einstein' all land on a key
    q = np.asarray(q, dtype=np.float64)
    n = np.arange(1, N)                          # step separations present on an N-site chain
    W = 1.0 - n / N                              # pair multiplicity 2(N-n), divided by N and halved

    r_n = n * d                                  # every pair n steps apart sits at the same distance
    if sigma == 0:                               # rigid lattice, no averaging to do
        sigma_n_sq = np.zeros_like(r_n, dtype=np.float64)
    else:
        sigma_n_sq = _pair_variance(model, sigma, r_n, n_steps=n, dim=1)

    return 1.0 + 2.0 * np.sum(W[:, None] * _pair_kernel(q, r_n, sigma_n_sq), axis=0)


def s_q_2d_analytic(q, lattice, N_1, N_2, sigma, model='spring'):
    """Orientationally averaged S(q) of a finite 2D lattice under either disorder model.

    The sum runs over signed step separations (n_1, n_2) rather than over pairs, since every pair
    separated by the same step vector has the same mean separation and the same variance. The steps
    must be signed: folding onto n_1, n_2 >= 0 with a 2**(M-1) degeneracy factor silently assumes
    lattice[0] . lattice[1] == 0, and is wrong for any oblique or hexagonal cell. The model
    enters only through _pair_variance:

        model='spring'   sigma_n^2 = sigma^2 [ln(r_n / eta_0) - Ci(k_d r_n)]
        model='well'     sigma_n^2 = 2 sigma^2

    The logarithmic growth is the Landau-Peierls result for a two-dimensional harmonic solid: a
    sheet has quasi-long-range order only, so its Bragg peaks broaden with order instead of merely
    losing weight. See structure_factors_DW.tex, Sec. 3.

    :param q: (n_q,) wavevector magnitudes; q = 0 may be included
    :param lattice: (2, 3) primitive vectors of the sheet
    :param N_1: repeats along lattice[0]
    :param N_2: repeats along lattice[1]
    :param sigma: bond r.m.s. for 'spring', site r.m.s. for 'well' - see s_q_1d_analytic
    :param model: 'spring' or 'well'
    :return: (n_q,) S(q), with S(0) = N_1 * N_2

    References
    ----------
    N. D. Mermin, Phys. Rev. 176, 250 (1968) -- crystalline order in two dimensions.
    A. Caille, C. R. Acad. Sci. Paris B 274, 891 (1972) -- the same physics for smectics.
    """
    model = _resolve_model(model)                # normalise before anything else can use it
    q = np.asarray(q, dtype=np.float64)
    A_cell = np.sqrt(np.linalg.det(lattice @ lattice.T))   # cell area from the Gram determinant

    s_q = np.ones_like(q)
    nz = (q != 0)
    s_q[~nz] *= N_1 * N_2                        # the sum rule S(0) = N, imposed exactly
    if not np.any(nz):
        return s_q

    # The sum runs over SIGNED steps. Folding it onto n_1, n_2 >= 0 with a 2**(M-1) degeneracy
    # factor is valid only when the primitive vectors are orthogonal: as soon as
    # lattice[0] . lattice[1] != 0 the steps (n_1, +n_2) and (n_1, -n_2) have different lengths,
    # so they cannot share a kernel. On a gamma = 120 deg cell, for instance, |a_1 + a_2| = d
    # while |a_1 - a_2| = sqrt(3) d. Folding leaves the total weight - and hence S(0) = N -
    # correct, so the sum rule does not detect the error; only the distance distribution is wrong.
    n_1 = np.arange(-(N_1 - 1), N_1)
    n_2 = np.arange(-(N_2 - 1), N_2)
    M_1, M_2 = np.meshgrid(n_1, n_2, indexing='ij')

    # normalised pair multiplicity: the fraction of ordered site pairs with signed step (n_1, n_2)
    w = ((1 - np.abs(M_1) / N_1) * (1 - np.abs(M_2) / N_2)).ravel()
    r = np.linalg.norm(M_1.ravel()[:, None] * lattice[0]
                       + M_2.ravel()[:, None] * lattice[1], axis=1)

    keep = r > 0                                 # drops (0, 0), the self term counted as the 1
    w, r = w[keep], r[keep]
    r_u, inv = np.unique(np.round(r, 9), return_inverse=True)   # one kernel call per length
    w_u = np.bincount(inv, weights=w)

    sigma_n_sq = (0.0 if sigma == 0 else
                  _pair_variance(model, sigma, r_u, dim=2, cell=A_cell))
    s_q[nz] += w_u @ _pair_kernel(q[nz], r_u, sigma_n_sq)

    return s_q


def s_q_3d_erf(q, lattice, N_1, N_2, N_3, sigma, model='spring'):
    """Orientationally averaged S(q) of a finite 3D lattice under either disorder model.

    As s_q_2d_analytic, with the three-dimensional variance law:

        model='spring'   sigma_n^2 = 2 sigma^2 [1 - Si(k_D r_n) / (k_D r_n)]
        model='well'     sigma_n^2 = 2 sigma^2

    The spring law saturates at 2 sigma^2 as r_n -> infinity, so beyond the first few shells the
    two models coincide and S(q) tends to a genuine Debye-Waller attenuation of the rigid lattice,
    with peaks that lose weight but do not broaden. That is the Landau-Peierls statement that a
    three-dimensional crystal has true long-range order, and it is why the well model is a usable
    approximation in 3D and not below it. See structure_factors_DW.tex, Sec. 3.

    :param q: (n_q,) wavevector magnitudes; q = 0 may be included
    :param lattice: (3, 3) primitive vectors
    :param N_1: repeats along lattice[0]
    :param N_2: repeats along lattice[1]
    :param N_3: repeats along lattice[2]
    :param sigma: bond r.m.s. for 'spring', site r.m.s. for 'well' - see s_q_1d_analytic
    :param model: 'spring' or 'well'
    :return: (n_q,) S(q), with S(0) = N_1 * N_2 * N_3

    References
    ----------
    M. Born and K. Huang, Dynamical Theory of Crystal Lattices (OUP, 1954), Ch. 5 -- the Debye
        model of <u u> from which the Si form follows.
    """
    model = _resolve_model(model)                # normalise before anything else can use it
    q = np.asarray(q, dtype=np.float64)
    V_cell = np.sqrt(np.abs(np.linalg.det(lattice @ lattice.T)))   # cell volume from the Gram det

    s_q = np.ones_like(q)
    nz = (q != 0)
    s_q[~nz] *= N_1 * N_2 * N_3                  # the sum rule S(0) = N, imposed exactly
    if not np.any(nz):
        return s_q

    # Signed steps, for the reason given in s_q_2d_analytic: the 2**(M-1) folding is valid only
    # for mutually orthogonal primitive vectors, and any lattice[a] . lattice[b] != 0 breaks the
    # assumption that the sign combinations of (|n_1|, |n_2|, |n_3|) all have the same length.
    n_1 = np.arange(-(N_1 - 1), N_1)
    n_2 = np.arange(-(N_2 - 1), N_2)
    n_3 = np.arange(-(N_3 - 1), N_3)
    M_1, M_2, M_3 = np.meshgrid(n_1, n_2, n_3, indexing='ij')

    w = ((1 - np.abs(M_1) / N_1) * (1 - np.abs(M_2) / N_2)
         * (1 - np.abs(M_3) / N_3)).ravel()
    r = np.linalg.norm(M_1.ravel()[:, None] * lattice[0]
                       + M_2.ravel()[:, None] * lattice[1]
                       + M_3.ravel()[:, None] * lattice[2], axis=1)

    keep = r > 0                                 # drops (0, 0, 0), the self term
    w, r = w[keep], r[keep]
    r_u, inv = np.unique(np.round(r, 9), return_inverse=True)   # one kernel call per length
    w_u = np.bincount(inv, weights=w)

    sigma_n_sq = (0.0 if sigma == 0 else
                  _pair_variance(model, sigma, r_u, dim=3, cell=V_cell))
    s_q[nz] += w_u @ _pair_kernel(q[nz], r_u, sigma_n_sq)

    return s_q


def s_q_3d_analytic(q, lattice, sigma):
    """Azimuthally averaged structure factor of a model held together by springs between its
        neighbours - a harmonic lattice.

            <S(q)> = 1 + (2/N) sum_{i<j} sinc(q R_ij) * exp(-q**2 * Var_ij / 2)

        with Var_ij the variance of the separation of the pair, taken from the spring network itself
        rather than assumed - see _harmonic_variance - and scaled so that one nearest-neighbour spacing
        has variance sigma**2.

        This is NOT a paracrystal. A paracrystal of the second kind takes Var = n * sigma**2, linear in
        the number of steps between the pair, and springs do that only in one dimension. In 2D the
        variance grows logarithmically and in 3D it nearly saturates, so the linear form over-damps by
        six times in 2D and ten in 3D by sixteen steps out. Physically that is Landau-Peierls: a chain
        has no long-range order and its peaks broaden with order, a 3D crystal keeps long-range order
        and its peaks only lose weight to a Debye-Waller factor. In 1D the two coincide exactly and
        this reduces to s_q_1d_DW above.

        Var_ij depends on where the pair sits and not only on how far apart it is, since the block has
        free surfaces. It is averaged over each distinct separation before the sum, which restores the
        gathering in _pair_terms and costs 0.005% of the peak against summing every pair separately.

        Closed form, no sampling. It is what g_r.MC_Sim's 'Hooke' potential gives in the limit of
        infinite averaging, with sigma**2 = 1 / C in kT units.

        :param q: scattering vector magnitudes, 1-D array in 1/nm
        :param lattice: (N, 3) atom positions in nm - the IDEAL lattice, not a displaced snapshot of
            one, since the disorder is what this function adds. It has to be a lattice laid out in loop
            order, because the spring network is read from the row order; see _lattice_indices
        :param sigma: rms fluctuation of one nearest-neighbour spacing, in nm

        :return: S(q), same shape as q
    """
    def _lattice_indices(positions):
        """The lattice index of each atom, taken from its place in the array.

        A model built by nested loops - g_r.build_crystal writes

            for i in range(rep_a):
                for j in range(rep_b):
                    for k in range(rep_c):
                        places[l] = i * a + j * b + k * c

        - already carries its indices in the row order: row l is cell (i, j, k) = unravel_index(l,
        (rep_a, rep_b, rep_c)). So there is nothing to search for geometrically. The loop periods are
        read back by looking at where the step between consecutive rows changes: it is the same vector
        all the way along the innermost axis and breaks at each wrap, so the first break is rep_c, the
        first break between those blocks is rep_b, and rep_a is what is left.

        Doing it this way rather than hunting for primitive vectors among the interatomic distances is
        both simpler and stronger. It never has to decide which vectors are "short", so a lattice whose
        axes are spaced very differently - 0.5 nm in plane against 6 nm between layers - is no harder
        than a cubic one, and an oblique lattice needs no special care either.

        The indices are checked against the positions before being returned, so a model that is not a
        lattice laid out in loop order raises rather than returning something quietly wrong.

        Returns (basis, idx) - the (3, 3) basis vectors, zero along any axis of one cell, and the
        (N, 3) integer indices.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('lattice must be an (N, 3) array of positions, got shape %r'
                             % (positions.shape,))
        N = positions.shape[0]
        if N < 2:
            raise ValueError('a model of one atom has no lattice to index')

        rel = positions - positions[0]  # row 0 is cell (0, 0, 0)
        tol = 1e-6 * max(float(np.abs(rel).max()), 1.0)

        def _period(block):
            """How many rows pass before the step between them changes - the length of one loop."""
            if block.shape[0] < 2:
                return 1
            step = block[1:] - block[:-1]
            same = np.all(np.abs(step - step[0]) <= tol, axis=1)
            return block.shape[0] if same.all() else int(np.argmin(same)) + 1

        rep_c = _period(rel)  # innermost loop
        if N % rep_c:
            raise ValueError('these positions are not a lattice in loop order: the innermost step '
                             'repeats every %i rows, which does not divide the %i atoms' % (rep_c, N))
        rep_b = _period(rel[::rep_c])  # next loop out, over the block starts
        if (N // rep_c) % rep_b:
            raise ValueError('these positions are not a lattice in loop order: the middle step repeats '
                             'every %i blocks, which does not divide the %i blocks of %i'
                             % (rep_b, N // rep_c, rep_c))
        rep_a = N // (rep_c * rep_b)
        reps = (rep_a, rep_b, rep_c)

        idx = np.stack(np.unravel_index(np.arange(N), reps), axis=1).astype(np.int64)
        basis = np.zeros([3, 3])  # an axis of one cell keeps a zero vector
        for ax, first in zip(range(3), (rep_b * rep_c, rep_c, 1)):
            if reps[ax] > 1:
                basis[ax] = rel[first]

        worst = float(np.abs(rel - idx @ basis).max())
        scale = float(np.linalg.norm(basis, axis=1).max())
        if worst > 1e-6 * scale:
            raise ValueError('these positions do not form a lattice laid out in loop order: reading the '
                             'row order as %r puts the worst atom %.4g off its site, against a longest '
                             'basis vector of %.4g. A cell with several atoms in it, a partly filled '
                             'lattice, a glass, a thermalised snapshot or a reordered model cannot be '
                             'indexed, and without indices the step count n_ij is not defined. Pass the '
                             'ideal lattice - the disorder is what this function is here to add'
                             % (reps, worst, scale))
        return basis, idx

    def _harmonic_variance(reps):
        """Variance of the separation between every pair of atoms of a finite spring lattice, in
        units of kT / C. Exact - no asymptotic form, no fitting, surfaces included.

        Springs between neighbours with the same stiffness make the energy k * L (x) I, with L the
        graph Laplacian of the lattice, so per axis the displacement covariance is its pseudo-
        inverse G and the variance of a separation is G_ii + G_jj - 2 G_ij.

        A rectangular block with free ends is a Kronecker sum of one-dimensional path graphs, whose
        modes are known in closed form - eigenvalue 4 sin^2(pi m / 2n), eigenvector the DCT-II basis
        - so the whole thing is assembled directly, about ten times faster than diagonalising the
        Laplacian and agreeing with it to 1e-13.

        This is what makes the function a spring model rather than a paracrystal. Springs accumulate
        disorder linearly only in 1D; measured against the step count n, this variance grows as

            1D  1, 2, 4, 8, 16       linear, so the paracrystal form is exact
            2D  0.76, 1.06, 1.44, 1.95, 2.72     logarithmic
            3D  0.72, 0.89, 1.03, 1.21, 1.63     nearly saturating

        at n = 1, 2, 4, 8, 16. Using n * sigma**2 in 2D would over-damp by six times at n = 16, and
        in 3D by ten. That is Landau-Peierls: a chain has no long-range order, a sheet has
        quasi-long-range order, and a three-dimensional crystal has true long-range order, so its
        Bragg peaks survive with a Debye-Waller factor instead of broadening away.
        """
        def _path(n):
            m = np.arange(n)
            lam = 4 * np.sin(np.pi * m / (2 * n)) ** 2             # free-end path eigenvalues
            V = (np.cos(np.pi * np.outer(np.arange(n) + 0.5, m) / n)
                 * np.sqrt(np.where(m == 0, 1.0, 2.0) / n))        # orthonormal DCT-II basis
            return lam, V

        lams, Vs = zip(*[_path(int(r)) for r in reps])
        lam = (lams[0][:, None, None] + lams[1][None, :, None]
               + lams[2][None, None, :]).ravel()                   # Kronecker sum of the axes
        V = np.einsum('ia,jb,kc->ijkabc', *Vs).reshape(int(np.prod(reps)), -1)
        inv = np.zeros_like(lam)
        inv[lam > 1e-12] = 1.0 / lam[lam > 1e-12]                  # drop the translational mode
        G = (V * inv) @ V.T
        diag = np.diag(G)
        return diag[:, None] + diag[None, :] - 2.0 * G


    from scipy.spatial.distance import pdist

    q = np.asarray(q, dtype=np.float64)
    positions = np.ascontiguousarray(lattice, dtype=np.float64)
    N = positions.shape[0]

    _, idx = _lattice_indices(positions)
    reps = tuple(int(v) for v in idx.max(axis=0) + 1)

    # np.triu_indices(N, 1) walks the pairs in the same order pdist returns them
    var = _harmonic_variance(reps)[np.triu_indices(N, 1)]
    r = pdist(positions)

    # The Laplacian variance is ALREADY in units of kT / C: the spring energy is
    # 0.5 * C * u^T L u with L unweighted, so the covariance is (kT / C) L^+. sigma**2 = kT / C is
    # therefore the whole scaling and nothing may be renormalised on top of it. In particular do NOT
    # force a nearest-neighbour pair to have variance sigma**2 - it only does so in 1D, where the
    # one-step variance is exactly 1; in 2D it is 0.761 and in 3D 0.718, so imposing sigma**2 there
    # inflates every variance by 1/0.761 and 1/0.718 and over-damps the whole curve
    var = var * sigma ** 2

    # average the variance over each distinct separation, so the sum gathers as it does for wells
    shell, inv = np.unique(r, return_inverse=True)
    count = np.bincount(inv.ravel(), minlength=shell.size)
    var_shell = np.bincount(inv.ravel(), weights=var, minlength=shell.size) / count

    # np.sinc(x) is sin(pi x) / (pi x), hence the division
    phase = np.sinc(np.outer(shell, q) / np.pi)
    damp = np.exp(-0.5 * np.outer(var_shell, q ** 2))
    return 1.0 + (2.0 / N) * (count[:, None] * phase * damp).sum(axis=0)


def s_q_decoupled(q, sigma, dim, lattice=None, N=None, d=None, model='spring'):
    """Orientationally averaged S(q) of a finite 1D, 2D or 3D lattice in the decoupled approximation.

        <S(q)> = 1 + sum'_n W_n I_n(q),    W_n = prod_a (1 - |n_a| / N_a),
        I_n(q) ~ sinc(q R_n) exp(-q^2 sigma_n^2 / 2)

    The sum runs over signed step vectors n = (n_1, ..., n_dim), |n_a| <= N_a - 1, with n = 0 left
    out (the prime) since it is the self term counted as the 1. Steps are signed so oblique cells are
    handled correctly (see the folding note in s_q_2d_analytic). sigma_n^2 comes from _pair_variance,
    so the variance laws are the ones used by the exact s_q_*d_analytic routines:

        model='spring'   1D  |n| sigma^2
                         2D  sigma^2 ln(R_n / eta_0)
                         3D  2 sigma^2 [1 - Si(k_D R_n) / (k_D R_n)]
        model='well'         2 sigma^2

    This differs from s_q_*d_analytic only in the kernel: those average sinc(qx) over the full
    Gaussian distribution of the separation, while here the Debye-Waller factor is pulled out of the
    orientational average. The two coincide when Delta^2 = sigma_L^2 - sigma_T^2 = 0; otherwise the
    decoupled form errs by O(sigma_n^2 / R_n^2), largest for the nearest neighbours.

    The lattice is given either explicitly, or through N and d as an orthogonal one:

        lattice given   3x3 vectors or 6 parameters, as in build_crystal (see _lattice_vectors).
                        Zero-length vectors are dropped; exactly `dim` must remain.
                        N is an int, one count per remaining vector, or (rep_a, rep_b, rep_c)
                        as passed to build_crystal (entries on zero vectors are ignored).
        lattice None    chain (dim=1), square (dim=2) or cube (dim=3) of spacing d, laid along
                        x, then y, then z, with N an int or one count per axis

    :param q: (n_q,) wavevector magnitudes; q = 0 may be included and gives S(0) = prod N_a
    :param sigma: bond r.m.s. for 'spring', site r.m.s. for 'well' - see s_q_1d_analytic
    :param dim: 1, 2 or 3
    :param lattice: 3x3 vectors or 6 lattice parameters (radians), or None to use d
    :param N: repeats per axis, see above
    :param d: spacing of the orthogonal lattice; only when lattice is None
    :param model: 'spring' or 'well'
    :return: (n_q,) S(q)

    References
    ----------
    P. Debye, Ann. Phys. 351, 809 (1915) -- the orientational average giving sinc.
    A. Guinier, X-Ray Diffraction (Freeman, 1963), Ch. 9 -- the Debye-Waller factor and its
        decoupling from the pair sum.
    """
    model = _resolve_model(model)                # normalise before anything else can use it
    if dim not in (1, 2, 3):                     # _pair_variance knows only these three laws
        raise ValueError('dim must be 1, 2 or 3, got %r' % (dim,))
    if N is None:                                # the pair weights cannot be formed without repeats
        raise ValueError('N (repeats per axis) is required')
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))   # accept a scalar q as well as an array
    N = np.atleast_1d(np.asarray(N, dtype=int))  # an int becomes a length-1 array

    if lattice is None:                          # orthogonal lattice: chain, square or cube
        if d is None:                            # nothing to build it from
            raise ValueError('give either lattice or d')
        vecs = d * np.eye(3)[:dim]               # rows d x_hat, d y_hat, d z_hat, as many as dim
    else:
        if d is not None:                        # two sources of geometry could silently disagree
            raise ValueError('give lattice or d, not both')
        full = _lattice_vectors(lattice)         # (3, 3) rows a, b, c
        active = np.linalg.norm(full, axis=1) > 0   # zero vectors are axes the crystal does not use
        if active.sum() != dim:                  # the variance law must match the real geometry
            raise ValueError('lattice has %d nonzero vectors but dim=%d' % (active.sum(), dim))
        vecs = full[active]                      # (dim, 3) the vectors actually repeated
        if N.size == 3 and dim != 3:             # (rep_a, rep_b, rep_c) as given to build_crystal
            N = N[active]                        # keep the counts on the nonzero vectors

    if N.size == 1:                              # one count for every axis
        N = np.repeat(N, dim)
    if N.size != dim:                            # anything else cannot be matched to the axes
        raise ValueError('N must be an int or have %d entries, got %r' % (dim, N.tolist()))

    # cell measure from the Gram determinant: length in 1D, area in 2D, volume in 3D
    cell = np.sqrt(np.abs(np.linalg.det(vecs @ vecs.T)))

    axes = [np.arange(-(N_a - 1), N_a) for N_a in N]          # -(N_a - 1) .. N_a - 1 per axis
    steps = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing='ij')], axis=1)  # (K, dim)

    w = np.prod(1 - np.abs(steps) / N, axis=1)   # W_n = prod_a (1 - |n_a| / N_a)
    r = np.linalg.norm(steps @ vecs, axis=1)     # R_n = |sum_a n_a a_a|

    keep = r > 0                                 # drops n = 0, the self term counted as the 1
    w, r = w[keep], r[keep]
    a_1 = np.linalg.norm(vecs[0])                # length scale for grouping equal distances
    _, inv = np.unique(np.round(r / a_1, 9), return_inverse=True)   # one kernel row per distinct length
    w_u = np.bincount(inv, weights=w)            # summed weight of every step with that length
    r_u = np.bincount(inv, weights=w * r) / w_u  # weighted mean true length, not the rounded key

    # in 1D every length is a whole number of steps, which the spring law needs explicitly
    n_steps = np.rint(r_u / cell) if dim == 1 else None
    sigma_n_sq = _pair_variance(model, sigma, r_u, n_steps=n_steps, dim=dim, cell=cell)
    sigma_n_sq = np.broadcast_to(sigma_n_sq, r_u.shape)          # 'well' may hand back a scalar

    phase = np.sinc(np.outer(r_u, q) / np.pi)    # sinc(q R_n); np.sinc is sin(pi x) / (pi x)
    damp = np.exp(-0.5 * np.outer(sigma_n_sq, q ** 2))           # exp(-q^2 sigma_n^2 / 2)
    return 1.0 + w_u @ (phase * damp)            # 1 + sum'_n W_n I_n(q)


def g_r_from_s_q(q, s_q, rho, r_min=0, r_max=15, dr=0.01, factor=1, type='Simpson'):
    """Given a q-vector q, an S(q) s_q, and a density rho, returns the radial distribution function in one of two ways:
    'DST' or 'Simpson' as given in type."""

    if type == 'DST':
        n = q.shape[0] * factor
        Q = np.linspace(0, q[-1], n)
        dq = (max(Q) - min(Q)) / n
        r = fftfreq(n, dq)[1:n // 2] * 2 * np.pi
        Yminus1 = np.interp(Q, q, s_q - 1)
        I = dst(Yminus1 * Q, type=1, norm='ortho')

        if factor % 2:
            g_r = 1 / (2 * rho * np.pi ** 2 * r) * I[1:-2:2]
        else:
            g_r = 1 / (2 * rho * np.pi ** 2 * r) * I[:-2:2]

        r_range = (r > r_min) & (r < r_max)

        return r[r_range], g_r[r_range]  # r, g_r

    elif type == 'Simpson':
        r = np.linspace(r_min, r_max, dc.int64((r_max - r_min) / dr) + 1)
        qr = q * np.reshape(r, [r.shape[0], 1])

        Yminus1 = s_q - 1
        I = simpson(Yminus1 * q * np.sin(qr), q)
        g_r = I / (2 * np.pi ** 2 * rho)
        g_r[r != 0] /= r[r != 0]

        return r, g_r


def g_r_from_model_slow(file, size_or_reps, file_triple='', radius=0, r_min=0, r_max = 15, dr = 0.01,
                        thermal=False,  u=np.array([0., 0., 0., 0.]), cube=True, lattice_vecs=np.zeros(6),
                        Number_for_average_atoms = 1, Number_for_average_conf=1):
    """Given a file of a structure and the box size, finds the radial distribution function."""
    print('Calculating g(r) from model...')
    vec, n = read_from_file(file, radius)
    if cube:
        vec_triple = np.zeros([Number_for_average_conf, 27 * n, 4])
    else:
        vec_triple = np.zeros([Number_for_average_conf, 27 * n, 3])

    # if not thermal:
    #     vec_triple[0] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs, thermal, u)
    # if Number_for_average_conf != 1:
    for conf in range(Number_for_average_conf):
        # vec = thermalize(np.copy(vec), u)
        vec_triple[conf] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs, thermal, u)
    # else:
    #     vec = thermalize(np.copy(vec), u)
    #     vec_triple[0] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs, thermal, u)

    num = 0
    it_tot = 0
    it_conf = 0
    rho = 0

    bins = np.arange(0, r_max + dr, dr)
    m = len(bins)
    g_r = np.zeros([Number_for_average_atoms*Number_for_average_conf, m])

    while it_conf < Number_for_average_conf:
        my_rand = np.random.randint(0, n, Number_for_average_atoms)
        it_atom = 0
        while it_atom < Number_for_average_atoms:
            r_0 = vec[my_rand[it_atom]]

            if radius == 0:
                for j in range(vec_triple.shape[1]):
                    row_j = vec_triple[it_conf, j]
                    d = np.sqrt((row_j[0] - r_0[0]) ** 2 + (row_j[1] - r_0[1]) ** 2 + (row_j[2] - r_0[2]) ** 2)
                    if (d < r_min) | (r_max < d) | (d < 1e-10):
                        pass

                    num += 1
                    for i in range(m):
                        if bins[i] < d:
                            pass
                        else:
                            if bins[i] >= d:
                                g_r[it_tot, i] += 1
                                break
            else:
                for j in range(vec_triple.shape[1]):
                    row = vec_triple[it_conf, j]
                    d = np.sqrt((row[0] - r_0[0]) ** 2 + (row[1] - r_0[1]) ** 2 + (row[2] - r_0[2]) ** 2)
                    # r = radius #row[3]
                    d_min = d - radius
                    d_max = d + radius
                    vol_tot = 4 * np.pi * radius ** 3 / 3
                    nn = int(N_R(radius, dr))

                    if (d < r_min) | (r_max < d_min) | (d < 1e-10):
                        pass
                    else:
                        num += 1

                        for i in range(m):
                            if bins[i] < d_min:
                                pass
                            else:
                                if bins[i] > d_max:
                                    g_r[it_tot, i] += 1
                                    break
                                else:
                                    counted_vol = 0
                                    for k in range(nn):
                                        if i + k < m:
                                            if bins[i + k] <= d_max:
                                                Vol = Lens_Vol(bins[i + k], radius, d) - counted_vol
                                                g_r[it_tot, i + k] += Vol / vol_tot
                                                counted_vol += Vol
                                            else:
                                                g_r[it_tot, i + k] += 1 - counted_vol / vol_tot
                                        else:
                                            break
                                    break

            rho_temp = 3 * np.sum(g_r[it_tot]) / (4 * np.pi * bins[-1] ** 3)
            rho += rho_temp
            if it_tot == 0:
                rad = rad_balls(bins, g_r[it_tot])
            g_r[it_tot, 1:] /= (rho_temp * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3))
            it_tot += 1
            it_atom += 1
        it_conf += 1

    # while it < Number_for_average_atoms:
    #     my_rand = np.random.randint(0, n)
    #     r_0 = vec[my_rand]
    #
    #     if radius == 0:
    #         for row_j in vec_triple:
    #                 d = np.sqrt((row_j[0] - r_0[0])**2 + (row_j[1] - r_0[1])**2 + (row_j[2] - r_0[2])**2)
    #                 if (r_max < d) | (d < 1e-10):
    #                     continue
    #                 num += 1
    #                 for i in range(m):
    #                     if bins[i] < d:
    #                         continue
    #                     else:
    #                         if bins[i] >= d:
    #                             g_r[it][i] += 1
    #                             break
    #     else:
    #         for row in vec_triple:
    #             d = np.sqrt((row[0] - r_0[0]) ** 2 + (row[1] - r_0[1]) ** 2 + (row[2] - r_0[2]) ** 2)
    #             r = row[3]
    #             d_min = d - r
    #             d_max = d + r
    #             vol_tot = 4 * np.pi * r ** 3 / 3
    #             N = int(N_R(r, dr))
    #
    #             if (r_max < d_min) | (d < 1e-10):
    #                 continue
    #             num += 1
    #
    #             for i in range(m):
    #                 if bins[i] < d_min:
    #                     continue
    #                 else:
    #                     if bins[i] > d_max:
    #                         g_r[0][i] += 1
    #                         break
    #                     else:
    #                         counted_vol = 0
    #                         for j in range(N):
    #                             if i + j < m:
    #                                 if bins[i + j] < d_max:
    #                                     Vol = Lens_Vol(bins[i + j], r, d) - counted_vol
    #                                     g_r[it][i + j] += Vol / vol_tot
    #                                     counted_vol += Vol
    #                                 else:
    #                                     g_r[it][i + j] += 1 - counted_vol / vol_tot
    #                             else:
    #                                 break
    #                         break
    #
    #     rho = 3 * sum(g_r[it]) / (4 * np.pi * bins[-1] ** 3)
    #     if it == 0:
    #         rad = rad_balls(bins, g_r[it])
    #     g_r[it][1:] /= (rho * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3))
    #     it += 1
    #
    # g_r = sum(g_r) / Number_for_average_atoms
    #
    r_range = (bins > r_min) & (bins < r_max)
    g_r[:] = np.sum(g_r, axis=0) / (Number_for_average_atoms * Number_for_average_conf)
    rho /= (Number_for_average_conf * Number_for_average_atoms)


    return bins[r_range], g_r[0, r_range], rho, rad

def g_r_from_model(file: str, size_or_reps: dc.float64[3], radius: dc.float64 = 0., r_min: dc.float64 = 0.,
                   r_max: dc.float64 = 15., dr: dc.float64 = 0.01, thermal: np.bool_ = False, u: dc.float64[4] =
                   np.array([0., 0., 0., 0.]), file_triple: str = '', cube: np.bool_ = True, lattice_vecs:
                   dc.float64[6] = np.array([0., 0., 0., 0., 0., 0.]), Number_for_average_conf: dc.int64 = 1,
                   Number_for_average_atoms: dc.int64 = 1):
    """Given a (dol/pdb) file of a structure and the box size, finds the radial distribution function. It is possible to
     enter thermal fluctuations by giving 'u' and thermal = 1. u is either int or 3 vector [ux, uy, uz], i.e. if int,
      same displacement in all directions else the given displacement in each directions. The displacement is given
      randomly according to a Gaussian distribution (np.random.normal)"""
    print('Calculating g(r)...')
    vec, n = read_from_file(file, radius)
    vec = np.copy(vec)
    if not thermal:
        # vec_triple = np.zeros([1, 27 * n, 4])
        if cube:
            vec_triple = np.zeros([1, 27 * n, 4])
        else:
            vec_triple = np.zeros([1, 27 * n, 3])
        vec_triple[0] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs)
    elif Number_for_average_conf != 1:
        if cube:
            vec_triple = np.zeros([Number_for_average_conf, 27 * n, 4])
        else:
            vec_triple = np.zeros([Number_for_average_conf, 27 * n, 3])
        for conf in range(Number_for_average_conf):
            # vec = thermalize(np.copy(vec), u)
            vec_triple[conf] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs, thermal, u)
    else:
        if cube:
            vec_triple = np.zeros([Number_for_average_conf, 27 * n, 4])
        else:
            vec_triple = np.zeros([Number_for_average_conf, 27 * n, 3])
        for conf in range(Number_for_average_conf):
            # vec = thermalize(np.copy(vec), u)
            vec_triple[conf] = triple(vec, size_or_reps, file_triple, cube, lattice_vecs, thermal, u)
    # vec_old = np.copy(vec)
    len_bins = int((r_max - r_min) / dr)
    bins = np.linspace(r_min, r_max, len_bins, endpoint=False)
    return compute_gr(np.copy(bins), vec_triple, vec, radius, dr, r_min, r_max,
                      Number_for_average_atoms=Number_for_average_atoms,
                      Number_for_average_conf=Number_for_average_conf, M=len_bins, TV=int(n*27), TF=np.shape(vec_triple)[2], V=n)


# @dc.program(auto_optimize=True, regenerate_code=True, device=dtypes.DeviceType.CPU)
# def compute_sq_CPU(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
#     qr: dc.float64[Q]
#
#     R = 0.0
#     for i in range(L - 1):
#         r_i = r_mat[i]
#         if i == 0:
#             r = np.sqrt(np.sum(r_i ** 2))
#             if r > R:
#                 R = r
#         for j in range(i + 1, L):
#             r_j = r_mat[j]
#             if i == 0:
#                 r = np.sqrt(np.sum(r_j ** 2))
#                 if r > R:
#                     R = r
#             r = np.sqrt(np.sum((r_i - r_j) ** 2))
#             qr = q * r
#             S_Q[0] += 2
#
#             S_Q[1:] += 2 * np.sin(qr[1:]) / qr[1:]
#
#     return R, S_Q
#
# @dc.program(auto_optimize=True, regenerate_code=True, device=dtypes.DeviceType.GPU)
# def compute_sq_GPU(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
#     qr: dc.float64[Q]
#
#     R = 0.0
#     for i in range(L - 1):
#         r_i = r_mat[i]
#         if i == 0:
#             r = np.sqrt(np.sum(r_i ** 2))
#             if r > R:
#                 R = r
#         for j in range(i + 1, L):
#             r_j = r_mat[j]
#             if i == 0:
#                 r = np.sqrt(np.sum(r_j ** 2))
#                 if r > R:
#                     R = r
#             r = np.sqrt(np.sum((r_i - r_j) ** 2))
#             qr = q * r
#             S_Q[0] += 2
#
#             S_Q[1:] += 2 * np.sin(qr[1:]) / qr[1:]
#
#     return R, S_Q

Number_for_average_atoms = dc.symbol('Number_for_average_atoms')
Number_for_average_conf = dc.symbol('Number_for_average_conf')


def compute_sq(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 3]):
    qr: dc.float64[Q]

    R = 0.0
    for i in range(L - 1):
        r_i = r_mat[i]
        if i == 0:
            r = np.sqrt(np.sum(r_i ** 2))
            if r > R:
                R = r
        for j in range(i + 1, L):
            r_j = r_mat[j]
            if i == 0:
                r = np.sqrt(np.sum(r_j ** 2))
                if r > R:
                    R = r
            r = np.sqrt(np.sum((r_i - r_j) ** 2))
            qr = q * r
            S_Q[0] += 2

            S_Q[1:] += 2 * np.sin(qr[1:]) / qr[1:]

    return R, S_Q


@dc.program(auto_optimize=True, regenerate_code=True)
def compute_gr(bins: dc.float64[M], vec_triple: dc.float64[NFC, TV, TF], vec: dc.float64[V, 4], radius=0., dr=0.01,
               r_min=0., r_max=15.):
    rho: dc.float64
    r_0: dc.float64[RO]
    g_r = np.zeros([Number_for_average_atoms * NFC, M])

    num = 0
    it_tot = 0
    it_conf = 0

    while it_conf < NFC:  # Number_for_average_conf
        my_rand: dc.int64[Number_for_average_atoms] = np.random.randint(0, V, Number_for_average_atoms)
        it_atom = 0
        while it_atom < Number_for_average_atoms:
            r_0 = vec[my_rand[it_atom]]

            if radius == 0.:
                for j in range(vec_triple.shape[1]):
                    row_j = vec_triple[it_conf, j]
                    d = np.sqrt((row_j[0] - r_0[0]) ** 2 + (row_j[1] - r_0[1]) ** 2 + (row_j[2] - r_0[2]) ** 2)
                    if (d < r_min) | (r_max < d) | (d < 1e-10):
                        pass

                    num += 1
                    for i in range(M):
                        if bins[i] < d:
                            pass
                        else:
                            if bins[i] >= d:
                                g_r[it_tot, i] += 1
                                break
            else:
                for j in range(vec_triple.shape[1]):
                    row = vec_triple[it_conf, j]
                    d = np.sqrt((row[0] - r_0[0]) ** 2 + (row[1] - r_0[1]) ** 2 + (row[2] - r_0[2]) ** 2)
                    # r = row[3]
                    d_min = d - radius
                    d_max = d + radius
                    vol_tot = 4 * np.pi * radius ** 3 / 3
                    nn = dc.int64(N_R(radius, dr))

                    if (d < r_min) | (r_max < d_min) | (d < 1e-10):
                        pass
                    else:
                        num += 1
                        for i in range(M):
                            if bins[i] < d_min:
                                pass
                            else:
                                if bins[i] > d_max:
                                    g_r[it_tot, i] += 1
                                    break
                                else:
                                    counted_vol = 0
                                    for k in range(nn[0]):
                                        if i + k < M:
                                            if bins[i + k] < d_max:
                                                Vol = Lens_Vol(bins[i + k], radius, d) - counted_vol
                                                g_r[it_tot, i + k] += Vol / vol_tot
                                                counted_vol += Vol
                                            else:
                                                g_r[it_tot, i + k] += 1 - counted_vol / vol_tot
                                        else:
                                            break
                                    break

            rho_temp = 3 * np.sum(g_r[it_tot]) / (4 * np.pi * bins[-1] ** 3)
            g_r[it_tot, 1:] /= (rho_temp * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3))
            it_tot += 1
            it_atom += 1
        it_conf += 1

    g_r[:] = np.sum(g_r, axis=0) / (Number_for_average_atoms * Number_for_average_conf)
    rho = 3 * np.sum(g_r[0]) / (4 * np.pi * bins[-1] ** 3)

    return bins, g_r[0], rho


my_gen = default_rng()  # Generator(PCG64())


def MC_Sim(dol_in, dol_out, temperature, MaxDistance, rest_distance, iterations, sampling_sigma,
           my_pot, *args, use_gpu=None, pop_out_num=1e3, min_accepted=1000, record_every=1,
           resync_every=100000):
    """
    Colour-sweep Metropolis Monte Carlo on a .dol model, on the GPU when cupy is available and on
    the CPU otherwise. Both backends run the same code against numpy or cupy, so a seeded run is
    reproducible on either one.

    How it samples
    --------------
    The bond list is built once, from the input geometry: every pair closer than MaxDistance is
    bonded, and those springs are then permanent. MaxDistance is never consulted again. The bond
    graph is coloured so that two atoms of the same colour are never bonded to each other, and one
    sweep proposes every colour class in turn - the whole class at once, with a separate
    displacement, a separate energy change and a separate accept/reject per atom.

    That is exact, not an approximation. Every partner of a moving atom belongs to a different
    colour and so is sitting still during that sub-step, which means each atom sees precisely the
    energy change a sequential single-atom move would have given it. Independent accept/reject on
    independent energies is identical to doing them one after another, so detailed balance is
    untouched, while the work is done as a handful of array operations per sweep instead of N
    sequential steps. Unlike a multi-atom move tested on the summed energy change - whose acceptance
    decays like exp(-c * n_move) and is numerically zero past about 32 atoms - every atom here keeps
    the full single-atom acceptance no matter how many move together. On a bipartite lattice half
    the model moves per sub-step.

    Why the bond list is frozen
    ---------------------------
    A live MaxDistance cut truncates the potential without shifting it, so a pair stepping past the
    cut sheds its whole strain energy in one downhill move that is always accepted, while re-forming
    costs the same energy uphill. A bonded pair at rest and a dissociated pair both sit at zero
    energy and the dissociated one has unbounded phase space, so such a model is only metastable and
    evaporates once 0.5 * k * (MaxDistance - rest_distance)**2 comes within reach of kT. Frozen
    springs cannot do that at any temperature, and they are also what keeps the colouring valid for
    the whole run: two same-colour atoms that drifted within a live cut would start interacting and
    their accept/reject would silently stop being independent.

    Potentials
    ----------
    'Hooke' and 'LJ' are VECTOR potentials on the frozen bonds. Each bond carries the unit direction
    n_hat it had in the input geometry, and the energy penalises the whole bond vector b rather than
    its length alone, so a bond constrains 3 numbers instead of 1. That is what removes the floppy
    folding modes which let a nearest-neighbour lattice crumple at constant bond length.
    'Gauss_Well' is not a pair potential at all - see below.

        'Hooke'  V = 0.5 * k * |b - rest_distance * n_hat|**2
                 ('Hook' and 'HookVec' are accepted as aliases, for older scripts.)

        'LJ'     V = 4 * eps * ((sig/|b|)**12 - (sig/|b|)**6) + 0.5 * k_LJ * |b - (b.n_hat) n_hat|**2

                 The first term is the ordinary Lennard-Jones in the bond length, repulsive wall,
                 finite well depth and all. The second penalises the part of the bond perpendicular
                 to its reference direction, which is what plain LJ leaves free and what a vector
                 potential has to supply. Its stiffness is not a new parameter: k_LJ is LJ's own
                 curvature at its minimum,

                     k_LJ = V''(r_min) = 72 * eps / (2**(1/3) * sig**2),   r_min = 2**(1/6) * sig

                 so close to equilibrium the well is isotropic in the bond vector, with exactly the
                 same curvature transversely as longitudinally - the same shape 'Hooke' has, and the
                 reason a lattice cannot crumple at constant bond length. Far from equilibrium it is
                 real Lennard-Jones along the bond, so a bond can still be pulled apart at high
                 temperature, unlike a spring.

                 rest_distance is not used by 'LJ' - the equilibrium bond length is the potential's
                 own r_min. Note that only the pairs in the frozen bond list interact at all, so
                 this is a genuine neighbour-list truncation of LJ, not the full pair sum.

    The price of a vector potential is the same in both cases: transverse stiffness is tied to
    longitudinal stiffness, where a real bond is usually far softer to bending, and a rigid rotation
    of the whole model now costs energy because n_hat is fixed in the lab frame. Translation is
    still free, since only bond vectors appear.

    Only the DIRECTION is taken from the input geometry. The length comes from rest_distance (or,
    for 'LJ', from sig), so a model that starts out slightly imperfect relaxes to the right spacing
    instead of freezing its own error in.

        'Gauss_Well'  V = 0.5 * k * |x_i - r0_i|**2, one term per ATOM

                 The Einstein crystal: every atom is tied by its own spring to its own site r0_i,
                 taken from the input geometry, and no two atoms interact at all. Nothing above
                 applies to it - there is no bond list, MaxDistance and rest_distance are ignored,
                 and since no pair of atoms is coupled the colouring is trivial: one class holding
                 the whole model, so an entire sweep is a single array operation.

                 It is the reference case worth having, because everything about it is known in
                 closed form. Each live axis of each atom is an independent gaussian of variance
                 1/k, so

                     <u_axis**2> = 1/k,   <|u|**2> = dim/k,   <U> = dim * N / 2 kT

                 with no zero modes to subtract - the wells pin absolute positions, so translation
                 is not free the way it is for a bond network. As a structure it is a paracrystal
                 with disorder of the FIRST kind: displacements are uncorrelated between atoms, the
                 lattice keeps long-range order, and the Bragg peaks lose weight to a Debye-Waller
                 factor without broadening. A bonded network gives disorder of the second kind
                 instead, where the peaks broaden with order.

                 Distance_Vector holds each atom's distance from its OWN well centre here, rather
                 than a list of bond lengths.

    Per-bond parameters
    -------------------
    rest_distance and the potential constants in *args each take a scalar OR a sequence. A scalar
    applies to every bond, which is the isotropic case and the default. A sequence is read as one
    value per bond DIRECTION when its length matches the number of distinct directions in the bond
    list, and as one value per bond when its length matches the number of bonds.

    Per-direction is the useful form. A cutoff bond list on a non-cubic lattice contains bonds of
    several different lengths and orientations, and a single rest_distance forces all of them to the
    same spacing, which builds in strain that can never relieve. Giving one rest_distance and one k
    per direction is the cutoff-built equivalent of writing the potential as a sum over the lattice
    axes,

        V = sum_axes 0.5 * C_axis * (|r_{i+e_axis} - r_i| - a_axis)**2

    with the difference that the cutoff finds every bond of the nearest-neighbour shell, including
    the ones that are not primitive-vector displacements. On simple cubic the two agree exactly, at
    3N bonds. On a 2D triangular net the cutoff finds 3 bond directions where the primitive vectors
    give 2, and on fcc it finds 6 where they give 3, because a1-a2 and its relatives sit at the same
    distance as a1 and a2 themselves.

    The directions are grouped up to sign, to 4 decimal places on the unit vector, and printed at
    setup with their bond counts and mean lengths, IN THE ORDER a per-direction sequence is read.
    Check that listing before trusting a per-direction run. 'Gauss_Well' has no bonds, so its k is a
    scalar or one value per ATOM instead.

    :param dol_in: filepath to the .dol to run the simulation on
    :param dol_out: filepath of the final model
    :param temperature: simulation temperature in K. Energies are in kT, so this is carried into the
        .dol header only
    :param MaxDistance: maximal distance to be considered as bound (in nm). Used once, at setup, to
        decide which pairs are bonded, and ignored entirely by 'Gauss_Well'
    :param rest_distance: the rest distance between two bonded molecules (in nm). 'Hooke' only -
        'LJ' takes its equilibrium length from sig, and 'Gauss_Well' has no bonds at all. Scalar,
        one value per bond direction, or one value per bond
    :param iterations: number of single-atom proposals. One sweep proposes every atom exactly once,
        so the run is iterations // N sweeps
    :param sampling_sigma: width of the gaussian the step length is drawn from, in nm. The step is
        that length times a direction drawn uniformly over the axes the model actually occupies. The
        length is centred on zero, so the displacement density is even, the proposal is symmetric
        and the plain exp(-dE) test is correct
    :param my_pot: 'Hooke' (or its aliases 'Hook', 'HookVec'), 'LJ', or 'Gauss_Well'
    :param args: for 'Hooke' the spring constant k in kT/nm**2; for 'LJ' epsilon in kT and sigma in
        nm; for 'Gauss_Well' the spring constant k in kT/nm**2. Each takes a scalar, one value per
        bond direction, or one value per bond ('Gauss_Well': per atom)
    :param use_gpu: None auto-detects cupy and falls back to numpy, True demands the GPU and raises
        if cupy is missing, False forces the CPU
    :param pop_out_num: PROPOSALS between snapshots. A snapshot can only land on a sweep boundary and
        a sweep accepts up to N moves at once, so counting proposals rather than accepted states
        keeps the schedule exact and independent of the acceptance rate. Rounded down to whole sweeps
    :param min_accepted: PROPOSALS before the first snapshot is written - the burn-in
    :param record_every: proposals between entries in Energy_Vector / Distance_Vector, rounded to a
        whole number of sweeps; record_every=N gives one entry per sweep. What is recorded is the
        state AFTER the sweep, not a trial state, so the mean of Energy_Vector is directly comparable
        to theory. Pass 0 to record nothing beyond the initial state
    :param resync_every: proposals between full recomputes of the running energy, which is otherwise
        accumulated from dE and drifts on float64 roundoff over millions of steps

    :return: Energy_Vector, Distance_Vector, acceptance_rate. Distance_Vector holds the bond
        lengths for 'Hooke' and 'LJ', and each atom's displacement from its own well centre for
        'Gauss_Well'
    """
    # ------------------------------------------------------------------ backend
    # numpy and cupy expose the same names for everything used below, so the sampler is written once
    # against `xp` and the choice of device is made here and nowhere else
    if use_gpu is None:
        try:
            import cupy as xp
            xp.cuda.Device().compute_capability          # fails if the driver or a device is missing
            on_gpu = True
        except Exception:
            xp, on_gpu = np, False
    elif use_gpu:
        try:
            import cupy as xp
        except ImportError as e:
            raise ImportError('use_gpu=True needs cupy (e.g. "pip install cupy-cuda12x"). Pass '
                              'use_gpu=None to fall back to the CPU automatically.') from e
        on_gpu = True
    else:
        xp, on_gpu = np, False

    def _host(a):
        """Device array to numpy, on either backend."""
        return xp.asnumpy(a) if on_gpu else np.asarray(a)

    # ------------------------------------------------------------------ potential name and arity
    # Only the NAME and the number of constants are settled here. Their values cannot be resolved
    # until the bond list exists, since a sequence is read against the bond directions it found
    _POT_ALIAS = {'Hooke': 'Hooke', 'Hook': 'Hooke', 'HookVec': 'Hooke', 'LJ': 'LJ',
                  'Gauss_Well': 'Gauss_Well'}
    pot = _POT_ALIAS.get(my_pot)
    if pot is None:
        raise NotImplementedError("my_pot must be 'Hooke' (aliases 'Hook', 'HookVec'), 'LJ' or "
                                  "'Gauss_Well', got %r" % (my_pot,))
    well = pot == 'Gauss_Well'                           # a single-site potential, not a pair one
    if pot in ('Hooke', 'Gauss_Well'):
        if len(args) < 1:
            raise TypeError("my_pot=%r needs the spring constant k, in kT/nm**2" % (pot,))
    elif len(args) < 2:
        raise TypeError("my_pot='LJ' needs epsilon (in kT) and sigma (in nm)")

    def _pair_energy(bvec, dirs, par):
        """Per-bond energy. `bvec` is the CURRENT bond vector i -> j, `dirs` the bond's unit
        direction in the input geometry - both (..., 3) and broadcast together - and `par` that
        bond's constants, each entry shaped like bvec[..., 0]. Returns (...,).

        A padding entry in a neighbour table produces a finite number here, never a nan or an inf,
        so that multiplying by the validity mask afterwards really does remove it. Padding carries
        k = 0 (or eps = 0), which already zeroes it, and a positive sigma so no division blows up."""
        if pot == 'Hooke':
            k, rest = par
            dv = bvec - rest[..., None] * dirs           # deviation from the ideal bond vector
            return 0.5 * k * (dv ** 2).sum(axis=-1)
        eps, sig, kt, floor = par
        r2 = (bvec ** 2).sum(axis=-1)                    # squared bond length
        b_par = (bvec * dirs).sum(axis=-1)               # longitudinal projection onto n_hat
        t2 = xp.maximum(r2 - b_par ** 2, 0.0)            # squared transverse part, clamped off zero
        sig_r = sig / xp.sqrt(xp.maximum(r2, floor ** 2))
        s6 = sig_r ** 6
        return 4 * eps * (s6 * s6 - s6) + 0.5 * kt * t2

    # ------------------------------------------------------------------ model and geometry
    Initial_Positions, Lattice_Number = read_from_file(dol_in, 0)          # (N, 4): x, y, z, radius
    Initial_Positions = np.ascontiguousarray(Initial_Positions[:, :3], dtype=np.float64)
    N = int(Lattice_Number)
    iterations = int(iterations)

    where_true = np.any(Initial_Positions, axis=0)       # which axes the model actually occupies
    dim = int(np.sum(where_true))                        # 1D, 2D or 3D
    live_ax = np.nonzero(where_true)[0]                  # indices of those axes
    if dim == 0:
        raise ValueError('every atom sits at the origin - nothing to move')

    sweeps = iterations // N
    if sweeps == 0:
        raise ValueError('iterations=%i is short of one sweep of %i atoms - raise it to at least %i'
                         % (iterations, N, N))

    def _build_bond_table(positions):
        """Bond list, direction groups and neighbour table, frozen at the input configuration.

        Returns (i_pair, j_pair, pair_dir, bond0, dir_id, n_dirs, nbr_idx, nbr_msk, nbr_dir,
        _scatter). i_pair/j_pair list each bond once, for the total energy, and bond0 holds their
        input lengths. dir_id labels each bond with its direction group and n_dirs counts them.
        nbr_idx/nbr_msk give, per atom, the atoms it is bonded to, padded to a rectangle so a whole
        colour class can gather its neighbours in one indexing operation; padding entries point at
        atom 0 and are FALSE in nbr_msk, so no unmasked sum may ever touch them. pair_dir and
        nbr_dir are the bonds' unit directions in the input geometry, pair_dir pointing i -> j and
        nbr_dir pointing away from the atom whose row it sits in. _scatter maps any per-bond array
        into the same (N, max_deg) neighbour layout."""
        from scipy.spatial import cKDTree

        pairs = cKDTree(positions).query_pairs(MaxDistance, output_type='ndarray')
        if pairs.shape[0] == 0:
            raise ValueError('MaxDistance=%g bonds no pairs at all - nothing to simulate' % MaxDistance)
        i_pair, j_pair = pairs[:, 0].copy(), pairs[:, 1].copy()

        bvec0 = positions[j_pair] - positions[i_pair]                      # i -> j
        bond0 = np.linalg.norm(bvec0, axis=1)
        pair_dir = bvec0 / bond0[:, None]                                  # unit vectors

        # Direction groups. A bond and its reverse are the same direction, so the sign is pinned by
        # forcing the first significantly nonzero component positive before the vectors are compared
        canon = pair_dir.copy()
        lead = np.argmax(np.abs(canon) > 1e-8, axis=1)                     # first real component
        canon *= np.sign(canon[np.arange(canon.shape[0]), lead])[:, None]
        key = np.round(canon, 4)                                           # tolerance of the grouping
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        first = np.array([int(np.argmax(inv == g)) for g in range(inv.max() + 1)])
        remap = np.empty(first.size, dtype=np.int64)
        remap[np.argsort(first)] = np.arange(first.size)                   # order by first appearance
        dir_id = remap[inv]
        n_dirs = int(first.size)

        # Neighbour table, built without a python loop over bonds: list every bond from both ends,
        # sort by the owning atom, then place each entry at its slot within that atom's block
        i_all = np.concatenate([i_pair, j_pair])
        j_all = np.concatenate([j_pair, i_pair])
        d_all = np.concatenate([pair_dir, -pair_dir])                      # flipped on the reversed
        order = np.argsort(i_all, kind='stable')                           # copy of each bond
        i_s, j_s, d_s = i_all[order], j_all[order], d_all[order]
        deg = np.bincount(i_s, minlength=N)                                # coordination of each atom
        max_deg = int(deg.max())
        block_start = np.concatenate([[0], np.cumsum(deg)[:-1]])           # where each atom's run begins
        slot = np.arange(i_s.size) - block_start[i_s]                      # position within that run

        nbr_idx = np.zeros([N, max_deg], dtype=np.int64)
        nbr_msk = np.zeros([N, max_deg], dtype=bool)
        nbr_dir = np.zeros([N, max_deg, 3])
        nbr_idx[i_s, slot] = j_s
        nbr_msk[i_s, slot] = True
        nbr_dir[i_s, slot] = d_s

        def _scatter(v, fill=0.0):
            """A per-bond array in the (N, max_deg) neighbour layout. `fill` is what the padding
            entries carry, and must keep _pair_energy finite there - zero for a stiffness, a
            positive number for anything that ends up in a denominator."""
            out = np.full([N, max_deg], float(fill))
            out[i_s, slot] = np.concatenate([v, v])[order]
            return out

        print('frozen bond list: %i atoms, %i bonds, max coordination %i, bond lengths %.4g to %.4g'
              % (N, i_pair.size, max_deg, bond0.min(), bond0.max()))
        print('bond directions: %i group%s, in the order a per-direction sequence is read'
              % (n_dirs, '' if n_dirs == 1 else 's'))
        for g in range(min(n_dirs, 12)):
            sel = dir_id == g
            d = canon[np.argmax(sel)]
            print('  [%2i] (%7.4f, %7.4f, %7.4f)  %6i bonds, length %.4g to %.4g'
                  % (g, d[0], d[1], d[2], int(sel.sum()), bond0[sel].min(), bond0[sel].max()))
        if n_dirs > 12:
            print('  ... and %i more' % (n_dirs - 12))
        return (i_pair, j_pair, pair_dir, bond0, dir_id, n_dirs, nbr_idx, nbr_msk, nbr_dir, _scatter)

    def _colour_bond_graph(i_pair, j_pair, nbr_idx, nbr_msk):
        """Colour the frozen bond graph so that two atoms of the same colour are never bonded, and
        their energy changes are therefore independent.

        A 2-colouring is tried first, by breadth-first search over whole frontiers at a time. A
        chain, a square lattice and a simple cubic lattice with nearest-neighbour bonds are all
        bipartite, and this finds the optimal 2 colours where greedy colouring can waste a third on
        something as simple as a chain. Anything carrying an odd cycle - fcc and hcp among them -
        falls through to Welsh-Powell greedy, which takes as many colours as it needs. Fewer colours
        means fewer sub-steps per sweep and larger classes, so the attempt is worth making."""
        deg = nbr_msk.sum(axis=1)
        colour = np.full(N, -1, dtype=np.int64)
        for root in range(N):                            # entered once per connected component
            if colour[root] >= 0:
                continue
            colour[root] = 0
            frontier = np.array([root], dtype=np.int64)
            c = 0
            while frontier.size:
                c ^= 1                                   # alternate the two colours by BFS layer
                nb = np.unique(nbr_idx[frontier][nbr_msk[frontier]])
                nb = nb[colour[nb] < 0]
                colour[nb] = c
                frontier = nb

        if np.any(colour[i_pair] == colour[j_pair]):     # an odd cycle: no 2-colouring exists
            colour = np.full(N, -1, dtype=np.int64)
            for i in np.argsort(-deg, kind='stable'):    # highest degree first
                used = colour[nbr_idx[i][nbr_msk[i]]]
                used = used[used >= 0]
                c = 0
                if used.size:
                    taken = np.zeros(int(used.max()) + 2, dtype=bool)
                    taken[used] = True
                    c = int(np.argmin(taken))            # lowest colour no neighbour has taken
                colour[i] = c

        if np.any(colour[i_pair] == colour[j_pair]):     # cheap, and the whole method rests on it
            raise RuntimeError('colouring failed: a bond joins two atoms of the same colour')
        return colour, int(colour.max()) + 1

    # ------------------------------------------------------------------ bonds, colours, constants
    if well:
        # No two atoms are coupled, so there is no graph to build and nothing to colour. One class
        # holding every atom is a valid colouring, and it makes a whole sweep one array operation
        colour, n_colours = np.zeros(N, dtype=np.int64), 1
        i_pair = j_pair = pair_dir = None

        def _per_atom(value, name):
            """A scalar or one value per atom, as an (N,) array."""
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0 or arr.size == 1:
                return np.full(N, float(arr.reshape(-1)[0]))
            if arr.ndim == 1 and arr.size == N:
                return np.ascontiguousarray(arr)
            raise ValueError("%s for 'Gauss_Well' takes a scalar or one value per atom (%i), got a "
                             "sequence of %i" % (name, N, arr.size))

        k_atom = _per_atom(args[0], 'k')
        par_pair = par_nbr = None
    else:
        (i_pair, j_pair, pair_dir, bond0, dir_id, n_dirs,
         nbr_idx, nbr_msk, nbr_dir, _scatter) = _build_bond_table(Initial_Positions)
        colour, n_colours = _colour_bond_graph(i_pair, j_pair, nbr_idx, nbr_msk)
        n_bonds = int(i_pair.size)
        k_atom = None

        def _per_bond(value, name):
            """A scalar, one value per bond DIRECTION or one value per bond, as an (n_bonds,) array.
            The per-direction reading is tried first; when the two lengths coincide the direction
            groups are in bond order anyway, so both readings agree."""
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0 or arr.size == 1:
                return np.full(n_bonds, float(arr.reshape(-1)[0]))
            if arr.ndim != 1:
                raise TypeError('%s must be a scalar or a 1-D sequence, got shape %r'
                                % (name, tuple(arr.shape)))
            if arr.size == n_dirs:                       # one per bond direction
                return arr[dir_id]
            if arr.size == n_bonds:                      # one per bond, in bond-list order
                return np.ascontiguousarray(arr)
            raise ValueError('%s has %i entries: give 1 (the same for every bond), %i (one per bond '
                             'direction, in the order printed above) or %i (one per bond)'
                             % (name, arr.size, n_dirs, n_bonds))

        def _report(name, v, unit):
            """Say what a constant came out as, so a per-direction run is checkable from the log."""
            if v.min() == v.max():
                return
            print('  %s varies over the bond list: %.4g to %.4g %s' % (name, v.min(), v.max(), unit))

        if pot == 'Hooke':
            k_b = _per_bond(args[0], 'k')
            rest_b = _per_bond(rest_distance, 'rest_distance')
            _report('k', k_b, 'kT/nm**2')
            _report('rest_distance', rest_b, 'nm')
            # A bond held far from its own rest length carries permanent strain it can never
            # relieve. With one rest_distance over a multi-shell bond list that is the usual mistake
            strain = 0.5 * k_b * (bond0 - rest_b) ** 2
            if strain.max() > 10:
                worst = int(np.argmax(strain))
                print('warning: bond %i sits at %g against rest_distance %g, %.1f kT of built-in '
                      'strain. Lower MaxDistance below that shell, or give one rest_distance per '
                      'bond direction, or the lattice will distort to relieve it'
                      % (worst, bond0[worst], rest_b[worst], strain.max()))
            par_pair = (k_b, rest_b)
            # padding gets k = 0, so its energy is zero whatever geometry the padded index implies
            par_nbr = (_scatter(k_b, 0.0), _scatter(rest_b, 0.0))
        else:
            eps_b = _per_bond(args[0], 'epsilon')
            sig_b = _per_bond(args[1], 'sigma')
            _report('epsilon', eps_b, 'kT')
            _report('sigma', sig_b, 'nm')
            kt_b = 0 #  72.0 * eps_b / (2.0 ** (1.0 / 3.0) * sig_b ** 2)        # V''(r_min), see docstring
            floor_b = 1e-3 * sig_b                       # hard wall, keeps (sig/r)**12 finite
            r_min = 2.0 ** (1.0 / 6.0) * sig_b           # the LJ minimum, bond by bond
            off = (bond0 > 3 * r_min) | (bond0 < 0.5 * r_min)
            if off.any():
                worst = int(np.argmax(np.abs(bond0 - r_min)))
                print('warning: %i bond(s) sit far from their LJ minimum, worst %g against %.4g. '
                      'The model will relax hard towards that spacing' % (int(off.sum()),
                                                                         bond0[worst], r_min[worst]))
            par_pair = (eps_b, sig_b, kt_b, floor_b)
            # padding gets eps = 0 and kt = 0, so its energy is zero, and a positive sigma and floor
            # so nothing is divided by zero on the way there
            par_nbr = (_scatter(eps_b, 0.0), _scatter(sig_b, 1.0),
                       _scatter(kt_b, 0.0), _scatter(floor_b, 1.0))

    # ------------------------------------------------------------------ proposals
    def _random_steps(size):
        """`size` independent displacements: a gaussian step length centred on zero times a
        direction drawn uniformly over the live axes. A zero-centred length makes the displacement
        density even, p(d) = p(-d), so T(x -> x+d) = T(x+d -> x) and the plain exp(-dE) Metropolis
        test samples Boltzmann. Drawn on the host from the module-level my_gen, so seeding my_gen
        reproduces a run on either backend."""
        R = my_gen.normal(0.0, sampling_sigma, size=size)                  # step lengths, signed
        out = np.zeros([size, 3])
        if dim == 1:                                     # 1D: along the single live axis
            out[:, live_ax[0]] = R
        elif dim == 2:                                   # 2D: uniform angle in the live plane
            theta = 2 * np.pi * my_gen.random(size)
            out[:, live_ax[0]] = R * np.cos(theta)
            out[:, live_ax[1]] = R * np.sin(theta)
        else:                                            # 3D: uniform direction on the sphere
            theta = np.arccos(2 * my_gen.random(size) - 1)                 # polar, area-weighted
            phi = 2 * np.pi * my_gen.random(size)                          # azimuthal
            sin_theta = np.sin(theta)
            out[:, 0] = R * sin_theta * np.cos(phi)
            out[:, 1] = R * sin_theta * np.sin(phi)
            out[:, 2] = R * np.cos(theta)
        return out

    # ------------------------------------------------------------------ output
    def _print_last_state(filepath, acceptance_rate, state_energy, positions, run_number):
        """One .dol snapshot, with the run's parameters in the header."""
        if filepath[-3:] != 'dol':                       # tolerate a path given without extension
            filepath += '.dol'
        filepath = filepath[:-4] + '_run_' + str(run_number) + '.dol'
        with open(filepath, 'w', encoding='utf-8', newline='\n') as file:
            outfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            outfile.writerow(["# potential:", pot])
            outfile.writerow(["# potential args:", str(args)])
            outfile.writerow(["# temperature:", temperature])
            outfile.writerow(["# MaxDistance:", MaxDistance])
            outfile.writerow(["# rest_distance:", str(rest_distance)])
            outfile.writerow(["# sampling_sigma:", sampling_sigma])
            outfile.writerow(["# iterations:", iterations])
            outfile.writerow(["# Acceptance Rate:", acceptance_rate])
            outfile.writerow(["# Last state energy:", state_energy])
            for i in range(positions.shape[0]):          # index, x, y, z, then three unused columns
                outfile.writerow([i, *positions[i], 0, 0, 0])
        return

    # ------------------------------------------------------------------ device arrays
    # xp.array copies on both backends, unlike xp.asarray, which hands numpy back the array it was
    # given. pos MUST be a copy of its own: it is written in place every sweep, and on the CPU an
    # aliased r0_g would drag every well centre along with its atom and pin the model at u = 0
    pos = xp.array(Initial_Positions, dtype=xp.float64)  # live positions, kept on the device
    if well:
        r0_g = xp.array(Initial_Positions, dtype=xp.float64)               # each atom's own well
        #                                                                    centre, never moved
        k_atom_g = xp.asarray(k_atom)
        i_pair_g = j_pair_g = pair_dir_g = None
        nbr_idx_g = nbr_msk_g = nbr_dir_g = par_pair_g = par_nbr_g = None
    else:
        r0_g = k_atom_g = None
        i_pair_g, j_pair_g = xp.asarray(i_pair), xp.asarray(j_pair)
        pair_dir_g = xp.asarray(pair_dir)
        nbr_idx_g, nbr_msk_g = xp.asarray(nbr_idx), xp.asarray(nbr_msk)
        nbr_dir_g = xp.asarray(nbr_dir)
        par_pair_g = tuple(xp.asarray(a) for a in par_pair)                # (n_bonds,) each
        par_nbr_g = tuple(xp.asarray(a) for a in par_nbr)                  # (N, max_deg) each
    classes = [xp.asarray(np.nonzero(colour == c)[0]) for c in range(n_colours)]
    accepted_g = xp.zeros(1, dtype=xp.int64)             # kept on device, so no sync per sweep

    def _state_energy():
        """Total energy, and the per-term distances it was summed over: bond lengths for the pair
        potentials, each atom's displacement from its own centre for 'Gauss_Well'. No distance cut
        anywhere - the bond list already decided what interacts."""
        if well:
            u = xp.sqrt(((pos - r0_g) ** 2).sum(axis=-1))
            return (0.5 * k_atom_g * u ** 2).sum(), u
        bvec = pos[j_pair_g] - pos[i_pair_g]
        E = _pair_energy(bvec, pair_dir_g, par_pair_g).sum()
        bonds = xp.sqrt((bvec ** 2).sum(axis=-1))
        return E, bonds

    def _class_dE(idx, r_i, r_new):
        """Energy change of every atom in one colour class, one entry per atom.

        For the pair potentials only a moving atom's own bonds change, and every partner is of a
        different colour and so is standing still, which is what makes these energies independent.
        For 'Gauss_Well' nothing is shared in the first place: each atom answers to its own centre."""
        if well:
            c = r0_g[idx]                                # the movers' own well centres
            return 0.5 * k_atom_g[idx] * (((r_new - c) ** 2).sum(axis=-1)
                                          - ((r_i - c) ** 2).sum(axis=-1))
        nb, mk = nbr_idx_g[idx], nbr_msk_g[idx]          # (C, D) partners and validity
        dirs = nbr_dir_g[idx]                            # (C, D, 3) ideal directions, atom -> partner
        par = tuple(a[idx] for a in par_nbr_g)           # (C, D) constants, this atom's own bonds
        r_j = pos[nb]                                    # (C, D, 3) partners, all standing still
        V_old = _pair_energy(r_j - r_i[:, None, :], dirs, par) * mk        # padding killed by the mask
        V_new = _pair_energy(r_j - r_new[:, None, :], dirs, par) * mk
        return (V_new - V_old).sum(axis=1)

    # ------------------------------------------------------------------ cadence, in sweeps
    snap_every = max(1, int(pop_out_num) // N)           # sweeps between snapshots
    first_snap = max(1, (int(min_accepted) + N - 1) // N)              # first sweep that may snapshot
    rec_every = max(1, int(record_every) // N) if record_every else 0

    print('MC_Sim: %s on %s, %iD, %i atoms, %i colours, largest class %i, %i sweeps of %i '
          'proposals, snapshot every %i sweeps from sweep %i'
          % (pot, 'GPU' if on_gpu else 'CPU', dim, N, n_colours, int(np.bincount(colour).max()),
             sweeps, N, snap_every, first_snap))
    if well:                                             # everything about this case is known exactly
        k_lo, k_hi = float(k_atom.min()), float(k_atom.max())
        if k_lo == k_hi:
            print('Gauss_Well with k = %.4g: per-axis variance 1/k = %.4g, <|u|**2> = dim/k = %.4g, '
                  '<U> = dim * N / 2 = %.4g kT' % (k_lo, 1.0 / k_lo, dim / k_lo, dim * N / 2.0))
        else:
            print('Gauss_Well with k from %.4g to %.4g: per-axis variance 1/k runs %.4g to %.4g, '
                  '<U> = dim * N / 2 = %.4g kT'
                  % (k_lo, k_hi, 1.0 / k_hi, 1.0 / k_lo, dim * N / 2.0))

    E_gpu, bonds = _state_energy()
    E = float(E_gpu)                                     # running energy as a host float
    Energy_Vector = [E]
    Distance_Vector = [_host(bonds)]

    # ------------------------------------------------------------------ the sweeps
    for s in range(sweeps):
        for idx in classes:
            C = int(idx.size)
            r_i = pos[idx]                               # (C, 3) where the movers are now
            r_new = r_i + xp.asarray(_random_steps(C))   # (C, 3) proposals
            dE = _class_dE(idx, r_i, r_new)              # (C,) one energy change per atom

            # One Metropolis test per atom. The maximum keeps the exponent from overflowing on a
            # strongly downhill move; those are accepted by the dE <= 0 arm regardless
            u = xp.asarray(my_gen.random(C))
            accept = (dE <= 0) | (xp.exp(-xp.maximum(dE, 0.0)) >= u)
            pos[idx] = xp.where(accept[:, None], r_new, r_i)
            accepted_g[0] += accept.sum()

        if rec_every and (s + 1) % rec_every == 0:       # the state after the sweep, not a trial
            E_rec, bonds = _state_energy()
            E = float(E_rec)
            Energy_Vector.append(E)
            Distance_Vector.append(_host(bonds))
        elif resync_every and ((s + 1) * N) % resync_every < N:         # kill accumulated float drift
            E = float(_state_energy()[0])

        if (s + 1) >= first_snap and (s + 1 - first_snap) % snap_every == 0:
            number_of_accepted_states = int(accepted_g[0])
            n_proposals = (s + 1) * N
            acceptance_rate = number_of_accepted_states / n_proposals
            print('sweep %i of %i, %i proposals, %i accepted, acceptance rate is %.3f'
                  % (s + 1, sweeps, n_proposals, number_of_accepted_states, acceptance_rate))
            _print_last_state(dol_out, acceptance_rate, E, _host(pos), n_proposals)

    acceptance_rate = int(accepted_g[0]) / (sweeps * N)
    print('acceptance rate is %f' % acceptance_rate)
    if acceptance_rate < 0.05:
        print('warning: acceptance %.4f is very low - lower sampling_sigma' % acceptance_rate)
    elif acceptance_rate > 0.95:
        print('warning: acceptance %.4f is very high, the model is barely moving - raise '
              'sampling_sigma' % acceptance_rate)
    _print_last_state(dol_out, acceptance_rate, float(_state_energy()[0]), _host(pos), sweeps * N)

    return np.asarray(Energy_Vector), np.concatenate(Distance_Vector), acceptance_rate


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    deg2rad = np.pi / 180

    filename = r'./Beck.dol'
    xyz_Beck = np.array([0.54338, 3.55503, 1.19651, 90 * deg2rad, 101.18 * deg2rad, 90 * deg2rad])
    beck_mat = build_crystal(xyz_Beck, 15, 15, 15, filename)
    reps = np.array([15, 15, 15])
    R = 0.25
    rmax = 4
    sigma = np.array([0.1, 0.1, 0.1])

    r_model, g_r_model, _ = g_r_from_model(filename, reps, radius=R, r_min=R, r_max=rmax, cube=False, lattice_vecs=xyz_Beck)
    r_model_2, g_r_model_2, _ = g_r_from_model(filename, reps, radius=R, r_min=R, r_max=rmax, cube=False,
                                            lattice_vecs=xyz_Beck)
    r_model_3, g_r_model_3, _ = g_r_from_model(filename, reps, radius=R, r_min=R, r_max=rmax, cube=False,
                                            lattice_vecs=xyz_Beck)
    r_model_4, g_r_model_4, _ = g_r_from_model(filename, reps, radius=R, r_min=R, r_max=rmax, cube=False,
                                            lattice_vecs=xyz_Beck)
    # r_model_therm, g_r_model_therm, _ = g_r_from_model(filename, reps, radius=R, r_min=R, r_max=rmax, cube=False,
    #                                                    lattice_vecs=xyz_Beck, Number_for_average_conf=100, thermal=True,
    #                                                    u=sigma)

    # plt.plot(r_model_therm, g_r_model_therm, label='With fluctuations', lw=3)
    plt.plot(r_model, g_r_model, label='With radius', lw=3)
    plt.plot(r_model_2, g_r_model_2, label='With radius 2', lw=3)
    plt.plot(r_model_3, g_r_model_3, label='With radius 3', lw=3)
    plt.plot(r_model_4, g_r_model_4, label='With radius 4', lw=3)
    plt.xlabel('r [nm]', size=14)
    plt.ylabel('$\\rho(r)/\\rho_{b}$', size=14)
    plt.legend(fontsize=14, loc='upper left')

    # file_single = r'.\cube_g_r_test.dol'
    # xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # build_crystal(xyz, 10, 10, 10, file_single)
    # Lx, Ly, Lz = 9., 9., 9.
    #
    # # r_slow, g_r_slow, _, _ = g_r_from_model_slow(file_single, Lx, Ly, Lz, r_max=5)
    # r_dace, g_r_dace, _ = g_r_from_model(file_single, [Lx, Ly, Lz], r_max=5)
    #
    # # q_slow, s_q_slow, _ = S_Q_from_model_slow(file_single, q_max=20)
    # # q_dace, s_q_dace, _ = S_Q_from_model(file_single, q_max=20)#, use_GPU=True)
    #
    # plt.figure()
    # plt.plot(r_dace, g_r_dace, label='DaCe')
    # # plt.plot(r_slow, g_r_slow, label='No DaCe')
    # plt.legend()
    plt.show()

    # plt.figure()
    # plt.semilogy(q_dace, s_q_dace, label='DaCe')
    # plt.semilogy(q_slow, s_q_slow, label='No DaCe')
    # plt.legend()
    # plt.show()
    #
    # q_final_dace, S_q_final_dace, S_Q_all_dace = S_Q_average_box(xyz, 20, 2002, 7, 13, 10, 2, r'.\s_q_example.dol',
    #                                                              slow=0)
    # q_final_slow, S_q_final_slow, S_Q_all_slow = S_Q_average_box(xyz, 20, 2002, 7, 13, 10, 2, r'.\s_q_example.dol',
    #                                                        slow=1)
    # exit()
    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    #
    # file_single = r'D:\Eytan\g_r_test\DOL\cube_g_r_test.dol'
    # build_crystal(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 10, 10, 10, file_single)
    # # file_single = r'D:\Eytan\g_r_test\DOL\thermal_cube.dol'
    # file_triple = r'D:\Eytan\g_r_test\DOL\thermal_cube_triple.dol'
    # vec, n = read_from_file(file_single, 0.01)
    # # write_to_dol(file_single[:-4] + '_test.dol', vec)
    # Lx = 9.0
    # Ly = 9.0
    # Lz = 9.0
    #
    # # vec_thermalized = thermalize(np.copy(vec), np.array([0.2, 0.2, 0.2, 0]))
    # # vec_3 = triple(np.copy(vec), Lx, Ly, Lz)#, file_triple)
    #
    # r, g_r, rho, rad = g_r_from_model_slow(file_single, Lx, Ly, Lz, thermal=True, Number_for_average_conf=150, u=0.05,
    #                                        r_max=5)
    # r_slow, g_r_slow, rho_slow, rad_slow = g_r_from_model_slow(file_single, Lx, Ly, Lz, r_max=5)
    # plt.plot(r, g_r)
    # plt.plot(r_slow, g_r_slow)
    #
    #
    # # q, s_q, rho_2 = S_Q_from_model(file_single, q_max=12)
    # # q_slow, s_q_slow, rho_2_slow = S_Q_from_model(file_single, q_max=12)
    # # plt.figure()
    # # plt.semilogy(q, s_q)
    # # plt.semilogy(q_slow, s_q_slow)
    # #
