import numpy as np
from numpy.random import default_rng, Generator, PCG64, randint, random, normal
from scipy.integrate import simpson
from scipy.fft import dst, fftfreq
from scipy.constants import k as Kb
import scipy.stats as stats
import csv
import math
import dace as dc
from dace import dtypes


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
        a1, b1, c1, alpha, beta, gamma = lattice
        t = np.cos(beta) - np.cos(alpha) * np.cos(gamma)
        B = np.sqrt(np.sin(gamma) ** 2 - np.sin(gamma) ** 2 * np.cos(alpha) ** 2 - t ** 2)

        a = np.array([a1, 0, 0])
        b = np.array([b1 * np.cos(gamma), b1 * np.sin(gamma), 0])
        c = np.array([c1 * np.cos(alpha), c1 * t / np.sin(gamma), c1 * B / np.sin(gamma)])
    elif m[0] == 3:
        a, b, c = lattice
    else:
        raise ValueError('Your lattice has to be either 3X3 or 1X6 i.e. X, Y, Z vectors, or a, b, c, alpha, beta, '
                         'gamma.')
    places = np.zeros([rep_a * rep_b * rep_c, 3])

    l = 0
    if ran == 0:
        for i in range(rep_a):
            for j in range(rep_b):
                for k in range(rep_c):
                    places[l] = i * a + j * b + k * c
                    l += 1
    else:
        change = 2 * ran * np.random.rand(rep_a * rep_b * rep_c, 3) - ran
        for i in range(rep_a):
            for j in range(rep_b):
                for k in range(rep_c):
                    places[l] = (i * a + a * change[l][0]) + (j * b + b * change[l][1]) + (k * c + c * change[l][2])
                    l += 1

    if move_to_GC:
        places = MoveToGC(places)

    if dol_out:
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
    """For now the function only returns the atomic radius of [H, He, Li, Be, B, C, N, O, Na, Cl]."""

    atm_type = atm_type.replace(' ', '')
    rad_dic = {'H': 53.0, 'He': 31.0, 'Li': 167.0, 'Be': 112.0, 'B': 87.0, 'C': 67.0, 'N': 56.0, 'O': 48.0, 'Na': 190.0,
               'Cl': 79.0}
    atm_rad = rad_dic[atm_type]
    return atm_rad / 1e3


def read_from_file(filename, r=0.1):
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
                        vec = np.append(vec, line[1:4])
                        vec = np.append(vec, r)
                except:  ## Needed for dol files created with PDB units
                    dol = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                    vec = np.array([])
                    for line in dol:
                        vec = np.append(vec, line[1:4])
                        vec = np.append(vec, r)
        except:
            with open(filename, encoding='utf-16') as file:
                dol = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                vec = np.array([])
                for line in dol:
                    vec = np.append(vec, line[1:4])
                    vec = np.append(vec, r)

    elif filename[-3:] == 'pdb':
        with open(filename, encoding='utf-8') as pdb:
            vec = np.array([])
            for line in pdb:
                if (line[:6] == 'ATOM  ') | (line[:6] == 'HETATM'):
                    atm_rad = find_atm_rad(line[76:78])
                    vec = np.append(vec, [float(line[30:38]) / 10, float(line[38:46]) / 10, float(line[46:54]) / 10,
                                          atm_rad])
                else:
                    continue
    n = dc.int64(vec.shape[0] / 4)
    vec = np.reshape(vec, [n, 4])
    # print('Done reading file')
    return vec, n


def draw_symmetry(filepath):
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
    V = (np.pi / (12 * d)) * (R + r - d) ** 2 * (d ** 2 + 2 * d * r - 3 * r ** 2 + 2 * d * R + 6 * r * R - 3 * R ** 2)
    return V


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
    if cube:
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
                    mat_triple[it:itpn] += mat_single + np.array([lx_t[dim_1], ly_t[dim_2], lz_t[dim_3], .0])
                    if file_triple:
                        write_to_dol(file_triple, mat_triple)

    else:
        a, b, c, alpha, beta, gamma = lattice_vecs[0], lattice_vecs[1], lattice_vecs[2], lattice_vecs[3],\
            lattice_vecs[4], lattice_vecs[5]
        new_repx, new_repy, new_repz = size_or_reps[0] * 3, size_or_reps[1] * 3, size_or_reps[2] * 3
        mat_triple = build_crystal(lattice_vecs, new_repx, new_repy, new_repz, file_triple)
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
    # TODO::Make this actually work well...
    u_len = np.size(u)
    vec_len = np.size(vec, axis=1)
    u_new = np.array(u)
    if not (vec_len % u_len) & (vec_len!=u_len):
        while np.size(u_new) < vec_len:
            u_new = np.append(u_new, u)
    else:
        u_new = np.append(u, 0)
    if (vec_len == 4) & (u_new[-1] != 0):
        u_new[-1] = 0
    new_vec = np.random.normal(vec, u_new)
    # if u_len == 1:
    #     u_new = np.array([u, u, u])
    #     new_vec = np.random.normal(vec, u_new)  # Radius is still inside
    # elif u_len == 2:
    #     u_new = np.array([*u, 0])
    #     new_vec = np.random.normal(vec, u_new)  # Radius is still inside
    # else:
    #     new_vec = np.random.normal(vec, u)  # Radius is still inside

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


def Amp_of_SF(dol_filename, ampj_filename, grid_size, q_max = 1, q_min = 0):
    a = Amplitude(grid_size, q_max, q_min)
    dol_f = read_from_file(dol_filename)[0][:, :3]
    a.fill(fill_SF, dol_f)
    if not ampj_filename[-5:] == '.ampj':
        a.save(ampj_filename + ".ampj")
    else:
        a.save(ampj_filename)


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


def Amp_multi(SF_Ampj, FF_Ampj, Mult_amp, N = 1):
    from dplus.Amplitudes import Amplitude
    if not SF_Ampj[-5:] == '.ampj':
        SF_Ampj += '.ampj'
    if not FF_Ampj[-5:] == '.ampj':
        FF_Ampj += '.ampj'
    if not Mult_amp[-5:] == '.ampj':
        Mult_amp += '.ampj'

    SF = Amplitude.load(SF_Ampj)
    FF = Amplitude.load(FF_Ampj)
    grid_min_size = np.max([SF.helper_grid.grid_size, FF.helper_grid.grid_size])
    q_min_size = np.max([SF.helper_grid.q_min, FF.helper_grid.q_min])
    q_max_size = np.min([SF.helper_grid.q_max, FF.helper_grid.q_max])
    multi_amp = Amplitude(grid_min_size, q_max_size, q_min_size)
    multi_amp.fill(fillmultigrid, SF, FF, N)
    multi_amp.save(Mult_amp)


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


def S_Q_from_model_slow(filename: str, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                        , thermal: np.bool_ = False, Number_for_average_conf: dc.int64 = 1, u=np.array([0, 0, 0, 0])):
    """Given a .dol or .pdb filename and a q-range, returns the orientation averaged structure factor."""

    r_mat, n = read_from_file(filename)
    r_mat = np.copy(r_mat)
    if thermal:
        r_mat_old = r_mat
    q = np.arange(q_min, q_max + dq, dq)
    S_Q = np.zeros([Number_for_average_conf, len(q)])
    S_Q += n
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    while it < Number_for_average_conf:
        if thermal:
            r_mat[:] = thermalize(np.copy(r_mat_old), u)

        # for i in dc.map[0:n-1]:
        for i in range(n - 1):
            r_i = r_mat[i]
            if i == 0:
                r = np.sqrt(np.sum(r_i ** 2))
                if r > R[it]:
                    R[it] = r
            # for j in dc.map[i+1:n]:
            for j in range(i + 1, n):
                r_j = r_mat[j]
                if i == 0:
                    r = np.sqrt(np.sum(r_j ** 2))
                    if r > R[it]:
                        R[it] = r
                r = np.sqrt(np.sum((r_i - r_j) ** 2))
                qr = q * r
                S_Q[it][0] += 2

                S_Q[it][1:] += 2 * np.sin(qr[1:]) / qr[1:]

        S_Q[it] /= n
        R[it] /= 2
        rho += 3 * S_Q[it][0] / (4 * np.pi * R[it] ** 3)
        it += 1
    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    rho /= Number_for_average_conf

    return q, S_Q[0], rho


def S_Q_from_model(filename: str, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                   , thermal: np.bool_ = False, Number_for_average_conf: dc.int64 = 1,
                   u: dc.float64[4] = np.array([0,0,0,0])):#, use_GPU: np.bool_ = True):
    """Given a .dol or .pdb filename and a q-range, returns the orientation averaged structure factor."""

    r_mat, n = read_from_file(filename)
    r_mat = np.copy(r_mat)
    if thermal:
        r_mat_old = r_mat
    q = np.arange(q_min, q_max + dq/2, dq)
    q_len = q.shape[0]
    if thermal:
        r_mat_old = r_mat
    S_Q = n * np.ones([Number_for_average_conf, q_len])
    # S_Q[:] += n
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    while it < Number_for_average_conf:
        if thermal:
            r_mat[:] = thermalize(np.copy(r_mat_old), u)
        S_Q_pretemp = np.copy(S_Q[it])
        R[it], S_Q[it, :] = compute_sq(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        # if use_GPU:
        #     R[it], S_Q[it, :] = compute_sq_GPU(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        # else:
        #     R[it], S_Q[it, :] = compute_sq_CPU(q, S_Q_pretemp, r_mat, Q=q_len, L=n)
        it += 1
    R_fin = np.max(R) / 2
    S_Q /= n
    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    # R /= 2
    rho += 3 * S_Q[0] / (4 * np.pi * R_fin ** 3)
    # rho += 3 * S_Q[0] / (4 * np.pi * R[it] ** 3)
    rho /= Number_for_average_conf

    return q, S_Q[0], rho


def S_Q_average_box(xyz, qmax, q_points, size_min, size_max, mean, sigma, file_path=r'.\S_Q_average', qmin=0,
                    normalize=1, make_fig=1, num_of_s_q_shown=-1, slow=0, thermal=False, u=np.array([0, 0, 0, 0])):#,
    # use_GPU=True):

    num_s_q = int(size_max - size_min)
    if num_of_s_q_shown < 0:
        num_of_s_q_shown = num_s_q
    S_q_all = np.zeros([num_s_q, q_points])
    dqn= (qmax - qmin) / (q_points - 1)
    if (file_path[-3:] == 'dol') | (file_path[-3:] == 'out'):
        file_path = file_path[:-4]

    if slow:
        for i in range(size_min, size_max):
            ind = int(i - size_min)
            build_crystal(xyz, i, i, i, file_path + '_box_size_' + str(i) + r'.dol')  # If files already exist,
            # this line can be commented out
            q, s_q_temp, _ = S_Q_from_model_slow(file_path + '_box_size_' + str(i) + r'.dol',q_min=qmin, q_max=qmax,
                                                 thermal=thermal, u=u)
            write_to_out(file_path + '_box_size_' + str(i) +'.out', q, s_q_temp)
            if normalize:
                S_q_all[ind] = s_q_temp / s_q_temp[0]
            else:
                S_q[ind] = s_q_temp
            print('Done with', i)
    else:
        for i in range(size_min, size_max):
            ind = int(i - size_min)
            build_crystal(xyz, i, i, i, file_path + '_box_size_' + str(i) + r'.dol')  # If files already exist,
            # this line can be commented out
            q, s_q_temp, _ = S_Q_from_model(file_path + '_box_size_' + str(i) + r'.dol', q_min=qmin, q_max=qmax, dq=dqn,
                                            thermal=thermal, u=u)#, use_GPU=use_GPU)
            write_to_out(file_path + '_box_size_' + str(i) + '.out', q, s_q_temp)
            if normalize:
                S_q_all[ind] = s_q_temp / s_q_temp[0]
            else:
                S_q_all[ind] = s_q_temp
            print('Done with', i)

    weight = stats.norm.pdf(range(size_min, size_max), mean, sigma)
    S_q_final = np.average(S_q_all, weights=weight, axis=0)
    write_to_out(file_path + '_no_box.out', q, S_q_final)

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
        plt.savefig(file_path + '.png')
        plt.close()

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

def g_r_from_model_slow(file, size_or_reps, file_triple=None, radius=0, r_min=0, r_max = 15, dr = 0.01,
                        Number_for_average_atoms = 1, Number_for_average_conf=1, thermal=False, u=np.array([0., 0., 0.,
                                                                                                        0.])):
    """Given a file of a structure and the box size, finds the radial distribution function."""
    vec, n = read_from_file(file, radius)
    vec_triple = triple(vec, size_or_reps, file_triple)
    vec_triple = np.zeros([Number_for_average_conf, 27 * n, 4])
    if not thermal:
        vec_triple[0] = triple(vec, size_or_reps)
    elif Number_for_average_conf != 1:
        for conf in range(Number_for_average_conf):
            vec = thermalize(np.copy(vec), u)
            vec_triple[conf] = triple(vec, size_or_reps)
    else:
        vec = thermalize(np.copy(vec), u)
        vec_triple[0] = triple(vec, size_or_reps)

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
                    r = row[3]
                    d_min = d - r
                    d_max = d + r
                    vol_tot = 4 * np.pi * r ** 3 / 3
                    nn = int(N_R(r, dr))

                    if (d < r_min) | (r_max < d_min) | (d < 1e-10):
                        pass
                    else:
                        num += 1

                        for i in range(m):
                            if bins[i] < d_min:
                                pass
                            else:
                                if bins[i] > d_max:
                                    g_r[0, i] += 1
                                    break
                                else:
                                    counted_vol = 0
                                    for k in range(nn):
                                        if i + k < m:
                                            if bins[i + k] < d_max:
                                                Vol = Lens_Vol(bins[i + k], r, d) - counted_vol
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

def g_r_from_model(file: str, size_or_reps: dc.float64[3], radius: dc.float64 = 0, r_min: dc.float64 = 0,
                   r_max: dc.float64 = 15, dr: dc.float64 = 0.01, Number_for_average_atoms: dc.int64 = 1,
                   thermal: np.bool_ = False, u: dc.float64[4] = np.array([0., 0., 0., 0.]), Number_for_average_conf:
        dc.int64 = 1, file_triple: str = '', cube: np.bool_ = True, lattice_vecs: dc.float64[6] = np.array([0., 0.,
                                                                                                            0., 0., 0., 0.])):
    """Given a (dol/pdb) file of a structure and the box size, finds the radial distribution function. It is possible to
     enter thermal fluctuations by giving 'u' and thermal = 1. u is either int or 3 vector [ux, uy, uz], i.e. if int,
      same displacement in all directions else the given displacement in each directions. The displacement is given
      randomly according to a Gaussian distribution (np.random.normal)"""

    vec, n = read_from_file(file, radius)
    vec = np.copy(vec)
    if not thermal:
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
    vec_old = np.copy(vec)

    bins = np.linspace(r_min, r_max, int((r_max - r_min) / dr), endpoint=False)
    return compute_gr(np.copy(bins), vec_old, file_triple, vec_triple, vec, radius, dr, r_min, r_max,
                      Number_for_average_atoms=Number_for_average_atoms,
                      Number_for_average_conf=Number_for_average_conf)


@dc.program(auto_optimize=True, regenerate_code=True, device=dtypes.DeviceType.CPU)
def compute_sq_CPU(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
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

@dc.program(auto_optimize=True, regenerate_code=True, device=dtypes.DeviceType.GPU)
def compute_sq_GPU(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
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
def compute_sq(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
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

Number_for_average_atoms = dc.symbol('Number_for_average_atoms')
Number_for_average_conf = dc.symbol('Number_for_average_conf')

@dc.program(auto_optimize=True, regenerate_code=True)
def compute_gr(bins: dc.float64[M], vec_old: dc.float64[V, 4], file_triple: str, vec_triple: dc.float64[NFC, TV, TF],
               vec: dc.float64[V, 4], radius=0, dr=0.01, r_min=0, r_max=15):
    rho: dc.float64
    r_0: dc.float64[4]
    # rad: dc.float64[A, 2]
    g_r = np.zeros([Number_for_average_atoms * Number_for_average_conf, M])

    num = 0
    it_tot = 0
    it_conf = 0

    while it_conf < NFC:#Number_for_average_conf:
        # if thermal == True:
        #     vec[:] = thermalize(np.copy(vec_old), u)
        #     vec_triple[:, :] = triple(np.copy(vec), Lx, Ly, Lz)#, file_triple)
        my_rand: dc.int64[Number_for_average_atoms] = np.random.randint(0, V, Number_for_average_atoms)
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
                    r = row[3]
                    d_min = d - r
                    d_max = d + r
                    vol_tot = 4 * np.pi * r ** 3 / 3
                    nn = dc.int64(N_R(r, dr))

                    if (d < r_min) | (r_max < d_min) | (d < 1e-10):
                        pass
                    else:
                        num += 1

                        for i in range(M):
                            if bins[i] < d_min:
                                pass
                            else:
                                if bins[i] > d_max:
                                    g_r[0, i] += 1
                                    break
                                else:
                                    counted_vol = 0
                                    for k in range(nn[0]):
                                        if i + k < M:
                                            if bins[i + k] < d_max:
                                                Vol = Lens_Vol(bins[i + k], r, d) - counted_vol
                                                g_r[it_tot, i + k] += Vol / vol_tot
                                                counted_vol += Vol
                                            else:
                                                g_r[it_tot, i + k] += 1 - counted_vol / vol_tot
                                        else:
                                            break
                                    break

            rho_temp = 3 * np.sum(g_r[it_tot]) / (4 * np.pi * bins[-1] ** 3)
            # if it_tot == 0:
            #     rad = rad_balls(bins, g_r[it_tot])
            g_r[it_tot, 1:] /= (rho_temp * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3))
            it_tot += 1
            it_atom += 1
        it_conf += 1

    g_r[:] = np.sum(g_r, axis=0) / (Number_for_average_atoms * Number_for_average_conf)
    rho = 3 * np.sum(g_r[0]) / (4 * np.pi * bins[-1] ** 3)

    return bins, g_r[0], rho#, rad

my_gen = default_rng()  # Generator(PCG64())

def Calc_Distance_Matrix_AS_Bound_version(positions, MaxDistance):
    Lattice_Number = np.shape(positions)[0]

    Distance_Matrix = np.zeros([Lattice_Number, Lattice_Number])

    for j in range(Lattice_Number):
        A = - np.eye(Lattice_Number)
        A[j, :] = 1
        A[:, :j+1] = 0
        X = np.matmul(positions[:, 0], A)
        Y = np.matmul(positions[:, 1], A)
        Z = np.matmul(positions[:, 2], A)

        Distance_Matrix[j, :] = np.sqrt(X**2 + Y**2 + Z**2)
        Distance_Matrix[Distance_Matrix > MaxDistance] = 0

    return Distance_Matrix

def Calc_State_Energy(Distance_Matrix, Rest_Distance, potential_type='Hook', *args):
    # throw out all non - connected atoms (zeros in Initial_distance_matrix), transfer to meters
    Distance_of_Connected_Atoms = Distance_Matrix[np.nonzero(Distance_Matrix)] * 1e-9
    if potential_type == 'Hook':  # Calculate hook potential for all connected pairs
        Potential = 0.5 * args[0] * (Distance_of_Connected_Atoms - Rest_Distance * 1e-9)**2
    elif potential_type == 'LJ':  # Calculate Lennard-Jones potential for all connected pairs
        sig_r = args[1] / Distance_of_Connected_Atoms
        Potential = 4 * args[0] * (np.power(sig_r, 12) - np.power(sig_r, 6))
    else:
        print('You promised me a potential...')
        return NotImplementedError
    # sum up the energy of all connected pairs in the system
    State_Energy = np.sum(Potential)

    return State_Energy

def Randomize_step(Step_Size, Sigma, dim, where_true):
    if dim == 1:
        R = my_gen.normal(Step_Size, Sigma)
        xyz = where_true * R
    elif dim == 2:
        R = my_gen.normal(Step_Size, Sigma)  # the radius of the step in nm, this random function is choosing the step
        # according to normal Gaussian distribution around step_size
        theta = 2 * np.pi * my_gen.random()  # the degree of the step in radians

        x = R * np.cos(theta)
        y = R * np.sin(theta)

        if where_true[0]:
            if where_true[1]:
                xyz = np.array([x, y, 0])
            else:
                xyz = np.array([x, 0, y])
        else:
            xyz = np.array([0, x, y])

    elif dim == 3:
        R = my_gen.normal(Step_Size, Sigma)
        theta = np.arccos(2 * my_gen.random() - 1)
        phi = 2 * np.pi * my_gen.random()

        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        xyz = np.array([x, y, z])

    return xyz

# def print_states(filepath, k_spring, temperature, MaxDistance, rest_distance, step_size, iterations, state_energy,
#                  positions, run_number):
#     filename = filepath + r'.\pyNew_State_' + str(run_number) + r'.dol'
#
#     with open(filename, 'w', newline='', encoding='utf-8') as file:
#         outfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
#         outfile.writerow(['# k_spring:', k_spring])
#         outfile.writerow(['# temperature:', temperature])
#         outfile.writerow(['# MaxDistance:', MaxDistance])
#         outfile.writerow(['# rest_distance:', rest_distance])
#         outfile.writerow(['# Kb:', Kb])
#         outfile.writerow(['# step_size:', step_size])
#         outfile.writerow(['# iterations:', iterations])
#         outfile.writerow(['# Last_state_enrgy:', state_energy])
#         for i in range(positions.shape[0]):
#             outfile.writerow([i, *positions[i], 0, 0, 0])
#     return

def print_last_state(filepath, k_spring, temperature, MaxDistance, rest_distance, step_size, iterations,
                     acceptance_rate, state_energy, positions, run_number):
    if filepath[-3:] != 'dol':
        filepath += '.dol'

    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        outfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        outfile.writerow(['# k_spring:', k_spring])
        outfile.writerow(['# temperature:', temperature])
        outfile.writerow(['# MaxDistance:', MaxDistance])
        outfile.writerow(['# rest_distance:', rest_distance])
        outfile.writerow(['# Kb:', Kb])
        outfile.writerow(['# step_size:', step_size])
        outfile.writerow(['# iterations:', iterations])
        outfile.writerow(['# Acceptance Rate:', acceptance_rate])
        outfile.writerow(['# Last state energy:', state_energy])
        for i in range(positions.shape[0]):
            outfile.writerow([i, *positions[i], 0, 0, 0])
    return

def MC_Sim(dol_in, dol_out, temperature, MaxDistance, rest_distance, step_size, iterations, sigma, my_pot, *args):
    '''
    :param dol_in: filepath to .dol to do the simulation on
    :param dol_out: filepath of the final model
    :param temperature: simulation temperature in K
    :param MaxDistance: maximal distance a molecule can move (in nm)
    :param rest_distance: the rest distance between two molecules (in nm)
    :param step_size: the center of the gaussian distribution according to which the molecule will move (in nm)
    :param iterations: number of iterations to run
    :param sigma: the sampling limit around step_size
    :param my_pot: 'Hook' or 'LJ' (for Lennard-Jones)
    :param args: for Hook, the spring constant, for LJ, epsilon and sigma (V=4*eps*((sig/r)**12-(sig/r)**6))

    :return: Energy_Vector, Distance_Vector
    '''

    Initial_Positions, Lattice_Number = read_from_file(dol_in, 0)
    Initial_Positions = Initial_Positions[:, :3]
    where_true = np.any(Initial_Positions, axis=0)
    dim = np.sum(where_true)
    # if np.any(Initial_Positions[:, 2]):
    #     dim = 3
    # else:
    #     dim = 2

    # defined variables
    Energy_Vector = []
    Distance_Vector = []

    New_Positions = np.copy(Initial_Positions)
    number_of_accepted_states = 0

    New_distance_matrix = Calc_Distance_Matrix_AS_Bound_version(Initial_Positions, MaxDistance)
    New_State_energy = Calc_State_Energy(New_distance_matrix, rest_distance, my_pot, *args)

    Distance_Vector = np.append(Distance_Vector, New_distance_matrix[np.nonzero(New_distance_matrix)] * 1e-9)
    Energy_Vector = np.append(Energy_Vector, New_State_energy)

    # creating one new random step
    Chosen_Atom = my_gen.integers(Lattice_Number, size=int(iterations))  # choose the atom that moves
    rand_num = my_gen.random(iterations)

    ## creating new state
    for i in range(iterations):
        Last_State_energy = np.copy(New_State_energy)  # transfer New_State_energy to Last_State_energy
        Last_distance_matrix = np.copy(New_distance_matrix)  # transfer New_distance_matrix to Last_distance_matrix each
        # iteration, if accepted then it will be the change and if not then it will be without the state (see
        # metropolis condition)
        Last_Positions = np.copy(New_Positions)  # create new state according to number of iteration

        xyz = Randomize_step(step_size, sigma, dim, where_true)
        New_Positions[Chosen_Atom[i]] += xyz

        # calculate distance only on connected atoms
        New_distance_matrix = Calc_Distance_Matrix_AS_Bound_version(New_Positions, MaxDistance)

        # stop the simulation if the lattice is broken
        B = New_distance_matrix > MaxDistance  # the condition is MaxDistance, should be controlled using well_depth.
        A = np.sum(B)
        if A != 0:
            break

        # calculate energy
        New_State_energy = Calc_State_Energy(New_distance_matrix, rest_distance, my_pot, *args)
        Energy_Vector = np.append(Energy_Vector, New_State_energy)
        Distance_Vector = np.append(Distance_Vector, New_distance_matrix[np.nonzero(New_distance_matrix)] * 1e-9)

        ## metropolis condition
        Energy_Change = np.abs(New_State_energy - Last_State_energy)  # the energy difference between each new state and the initial state

        p = np.exp(-(Energy_Change / (Kb * temperature)))  # the probabilty of each new state

        if (New_State_energy < Last_State_energy) or (p < rand_num[i]):  # the condition to accept or deny new state
            number_of_accepted_states += 1  # counting each accepted state
            # printing the state every 10, 000 accepted states
            if number_of_accepted_states % 1e4 == 0:
                print('accepted %i states' %(number_of_accepted_states))
        else:
            # if the change is not accepted then you save the last state as the new state
            New_Positions = np.copy(Last_Positions)
            New_State_energy = np.copy(Last_State_energy)
            New_distance_matrix = np.copy(Last_distance_matrix)

    # acceptance rate
    acceptance_rate = number_of_accepted_states / iterations

    # printing the last state
    print_last_state(dol_out, args, temperature, MaxDistance, rest_distance, step_size, iterations,
                     acceptance_rate, New_State_energy, New_Positions, i)

    return Energy_Vector, Distance_Vector


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
