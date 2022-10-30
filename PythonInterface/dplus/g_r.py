import numpy as np
from scipy.integrate import simpson
from scipy.fft import dst, fftfreq
import csv
import math
import dace as dc

V = dc.symbol('V')
W = dc.symbol('W')
Y = dc.symbol('Y')
M = dc.symbol('M')
TV = dc.symbol('TV')
Q = dc.symbol('Q')
L = dc.symbol('L')
U = dc.symbol('U')
S = dc.symbol('S')


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
    n = len(g_r)
    rad = np.array([])
    for i in range(n):
        if g_r[i] != 0.0:
            rad = np.append(rad, [r[i], g_r[i]])
    rad_n = rad.reshape([int(len(rad) / 2), 2])
    return rad_n


def build_crystal(lattice, rep_a, rep_b, rep_c, dol_out, ran=0, move_to_GC=1):
    """Given lattice vectors a, b, c and the number of repetitions of each vector, builds a .dol file at location
     dol_out (dol_out must contain both the location and the file name)."""

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

    with open(dol_out, 'w', newline='', encoding='utf-8') as file:
        dolfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        # dolfile.writerow(['## vec a = ', a, 'rep a = ', rep_a, 'vec b = ', b, 'rep b = ', rep_b, 'vec c = ', c, 'rep c = ', rep_c])
        for i in range(0, np.shape(places)[0]):
            dolfile.writerow([i, *places[i], 0, 0, 0])

    return


def write_to_out(out_file, q, I):
    with open(out_file, 'w', newline='', encoding='utf-8') as file:
        dolfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        # dolfile.writerow(['## q', 'I(q)'])
        for i in range(len(q)):
            dolfile.writerow([q[i], I[i]])
    return


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
    n = int(len(vec) / 4)
    vec = np.reshape(vec, [n, 4])
    # print('Done reading file')
    return vec, n


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
    N_R = math.ceil(2 * r / dR) + 1
    return N_R


def balls_in_spheres(vec_triple, x_0, y_0, z_0, R_max):
    """Given a matrix vec_triple, a point of reference (x_0, y_0, z_0), and a radius R_max, returns the number of points
    that are inside the sphere."""

    m = np.shape(vec_triple)[0]
    dist = np.zeros(m)
    for i in range(m):
        dist[i] = np.sqrt(sum((vec_triple[i][:3] - np.array([x_0, y_0, z_0])) ** 2))
    num = sum((dist < R_max) & (dist > 1e-10))

    return num


def find_N(r, g_r, rho, r_min=0, r_max=0.56402):
    """Find the number of atoms inside the range (r_min, r_max), given r, g_r and density rho."""

    r_range = (r > r_min) & (r < r_max)
    N = simpson(rho * g_r[r_range] * 4 * np.pi * r[r_range] ** 2, r[r_range])

    return N


def triple(mat_single, Lx, Ly, Lz, file_triple=None):
    n = np.shape(mat_single)[0]
    trans = [-1, 0, 1]
    mat_triple = mat_single

    for i in trans:
        for j in trans:
            for k in trans:
                if i == j == k == 0:
                    continue
                new_loc = mat_single + np.array([i * Lx, j * Ly, k * Lz, 0])
                mat_triple = np.append(mat_triple, new_loc, axis=0)

    if file_triple:
        with open(file_triple, 'w', newline='', encoding='utf-8') as file:
            dolfile = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            for i in range(0, np.shape(mat_triple)[0]):
                dolfile.writerow([i, *mat_triple[i][:-1], 0, 0, 0])

    return mat_triple, 27 * n


def thermalize(vec, u):
    if (u != 1) & (np.shape(vec)[1] != len(u)):
        u = np.append(u, 0)
    elif u == 1:
        u = np.append(u, [u, u, 0])

    new_vec = np.random.normal(vec, u)  # Radius is still inside
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


def S_Q_from_model_slow(filename: dc.string, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.1
                        , thermal: dc.float64 = 0, Number_for_average_conf: dc.int64 = 1, u: dc.float64[U] = 0):
    """Given a .dol or .pdb filename and a q-range, returns the orientation averaged structure factor."""

    r_mat, n = read_from_file(filename)
    r_mat = np.copy(r_mat)
    if thermal:
        r_mat_old = r_mat
    q = np.arange(q_min, q_max + dq, dq)
    S_Q = np.zeros([Number_for_average_conf, Q])
    S_Q += L
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    while it < Number_for_average_conf:
        if thermal:
            r_mat[:] = thermalize(r_mat_old, u)

        # for i in dc.map[0:n-1]:
        for i in range(L - 1):
            r_i = r_mat[i]
            if i == 0:
                r = np.sqrt(sum(r_i ** 2))
                if r > R[it]:
                    R[it] = r
            # for j in dc.map[i+1:n]:
            for j in range(i + 1, L):
                r_j = r_mat[j]
                if i == 0:
                    r = np.sqrt(sum(r_j ** 2))
                    if r > R[it]:
                        R[it] = r
                r = np.sqrt(sum((r_i - r_j) ** 2))
                qr = q * r
                S_Q[it][0] += 2

                S_Q[it][1:] += 2 * np.sin(qr[1:]) / qr[1:]

        S_Q[it] /= L
        R[it] /= 2
        rho += 3 * S_Q[it][0] / (4 * np.pi * R[it] ** 3)
        it += 1
    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    rho /= Number_for_average_conf

    return q, S_Q, rho

    return compute_sq(q, r_mat, thermal, Number_for_average_conf, u,
                      Number_for_average_conf=Number_for_average_conf)  # , r_mat_old


def S_Q_from_model(filename: str, q_min: dc.float64 = 0, q_max: dc.float64 = 100, dq: dc.float64 = 0.01
                   , thermal: dc.bool = False, Number_for_average_conf: dc.int64 = 1, u: dc.float64[U] = 0):
    """Given a .dol or .pdb filename and a q-range, returns the orientation averaged structure factor."""

    r_mat, n = read_from_file(filename)
    r_mat = np.copy(r_mat)
    if thermal:
        r_mat_old = r_mat
    q = np.arange(q_min, q_max + dq, dq)
    q_len = q.shape[0]
    if thermal:
        r_mat_old = r_mat
    S_Q = np.zeros([Number_for_average_conf, q_len])
    S_Q[:] += n
    R = np.zeros(Number_for_average_conf)
    rho = 0
    it = 0
    while it < Number_for_average_conf:
        if thermal:
            r_mat[:] = thermalize(r_mat_old, u)
        S_Q_pretemp = np.copy(S_Q[it])
        R[it], S_Q[it, :] = compute_sq(q, S_Q_pretemp, r_mat)
        # R_temp, S_Q_temp = compute_sq(q, S_Q_pretemp, r_mat)
        # R[it] = R_temp
        # S_Q[it, :] = S_Q_temp
        it += 1
    R_fin = np.sum(R, axis=0) / 2
    S_Q /= n
    S_Q[:] = np.sum(S_Q, axis=0) / Number_for_average_conf
    # R /= 2
    rho += 3 * S_Q[0] / (4 * np.pi * R_fin ** 3)
    # rho += 3 * S_Q[0] / (4 * np.pi * R[it] ** 3)
    rho /= Number_for_average_conf

    return q, S_Q[0], rho


def s_q_from_g_r(r, g_r, rho, q_min=0, q_max=50, dq=0.01, factor=1, type='Simpson'):
    """Given an r-vector r, a g(r) g_r, and a density rho, returns the structure factor in one of two ways: 'DST' or
     'Simpson' as given in type."""
    if type == 'DST':
        n = len(r) * factor
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
        q = np.linspace(q_min, q_max, int((q_max - q_min) / dq) + 1)
        qr = r * np.reshape(q, [len(q), 1])

        I = simpson(g_r * r * np.sin(qr), r)
        s_q = I * (4 * np.pi * rho)
        s_q[q != 0] /= (q[q != 0])
        s_q += 1

        return q, s_q


def g_r_from_s_q(q, s_q, rho, r_min=0, r_max=15, dr=0.01, factor=1, type='Simpson'):
    """Given a q-vector q, an S(q) s_q, and a density rho, returns the radial distribution function in one of two ways:
    'DST' or 'Simpson' as given in type."""

    if type == 'DST':
        n = len(q) * factor
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
        r = np.linspace(r_min, r_max, int((r_max - r_min) / dr) + 1)
        qr = q * np.reshape(r, [len(r), 1])

        Yminus1 = s_q - 1
        I = simpson(Yminus1 * q * np.sin(qr), q)
        g_r = I / (2 * np.pi ** 2 * rho)
        g_r[r != 0] /= r[r != 0]

        return r, g_r


def g_r_from_model(file: dc.string, Lx: dc.float64, Ly: dc.float64, Lz: dc.float64, file_triple: dc.string = None,
                   radius: dc.float64 = 0, r_max: dc.float64 = 15, dr: dc.float64 = 0.01,
                   Number_for_average_atoms: dc.int64 = 1
                   , thermal: dc.bool = 0, u: dc.float64[3] = np.array([0, 0, 0]),
                   Number_for_average_conf: dc.int64 = 1):
    """Given a (dol/pdb) file of a structure and the box size, finds the radial distribution function. It is possible to
     enter thermal fluctuations by giving 'u' and thermal = 1. u is either int or 3 vector [ux, uy, uz], i.e. if int,
      same displacement in all directions else the given displacement in each directions. The displacement is given
      randomly according to a Gaussian distribution (np.random.normal)"""

    vec, n = read_from_file(file, radius)
    vec = np.copy(vec)
    if thermal != 1:
        vec_triple = triple(vec, Lx, Ly, Lz, file_triple)
    vec_old = vec

    bins = np.arange(0, r_max + dr, dr)
    return compute_gr(bins, thermal, vec_old, Lx, Ly, Lz, file_triple,
                      vec_triple, vec, radius, u, dr, r_max,
                      Number_for_average_atoms=Number_for_average_atoms,
                      Number_for_average_conf=Number_for_average_conf)


@dc.program(auto_optimize=True)
def compute_sq(q: dc.float64[Q], S_Q: dc.float64[Q], r_mat: dc.float64[L, 4]):
    qr: dc.float64[Q]

    R = 0
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


@dc.program(auto_optimize=True)
def compute_gr(bins: dc.float64[M], thermal: dc.bool, vec_old: dc.float64[V, 4], Lx: dc.int64, Ly: dc.int64,
               Lz: dc.int64, file_triple: str, vec_triple, vec, radius=0, u=np.array([0, 0, 0]), dr=0.01, r_max=15):
    g_r = np.zeros([Number_for_average_atoms * Number_for_average_conf, M])

    num = 0
    it_tot = 0
    it_conf = 0

    while it_conf < Number_for_average_conf:
        if thermal == 1:
            vec[:] = thermalize(vec_old, u)
            vec_triple[:] = triple(vec, Lx, Ly, Lz, file_triple)

        it_atom = 0
        while it_atom < Number_for_average_atoms:
            my_rand: int = np.random.randint(0, V)
            r_0 = vec[my_rand]

            if radius == 0:
                for j in range(vec_triple.shape[0]):
                    row_j = vec_triple[j]
                    d = np.sqrt((row_j[0] - r_0[0]) ** 2 + (row_j[1] - r_0[1]) ** 2 + (row_j[2] - r_0[2]) ** 2)
                    if (r_max < d) | (d < 1e-10):
                        continue
                    num += 1
                    for i in range(M):
                        if bins[i] < d:
                            pass
                        else:
                            if bins[i] >= d:
                                g_r[it_tot, i] += 1
                                break
            else:
                for j in range(vec_triple.shape[0]):
                    row = vec_triple[j]
                    d = np.sqrt((row[0] - r_0[0]) ** 2 + (row[1] - r_0[1]) ** 2 + (row[2] - r_0[2]) ** 2)
                    r = row[3]
                    d_min = d - r
                    d_max = d + r
                    vol_tot = 4 * np.pi * r ** 3 / 3
                    nn = int(N_R(r, dr))

                    if (r_max < d_min) | (d < 1e-10):
                        continue
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

            rho = 3 * sum(g_r[it_tot]) / (4 * np.pi * bins[-1] ** 3)
            # if it_tot == 0:
            #     rad = rad_balls(bins, g_r[it_tot])
            g_r[it_tot, 1:] /= (rho * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3))
            it_tot += 1
            it_atom += 1
        it_conf += 1

    g_r[:] = np.sum(g_r) / (Number_for_average_atoms * Number_for_average_conf)

    return bins, g_r, rho  # , rad


# def new_g_r(file, Lx, Ly, Lz, R_max=5, dr = 0.01, it = 1):  #, r0=np.array([0,0,0])
#
#     vec, n = read_from_file(file, 0)
#     vec_triple, m = triple(vec, Lx, Ly, Lz)
#     r = np.arange(0, R_max + dr, dr)
#     lr = len(r)
#     iter_num = 0
#     # g_r = np.zeros(lr)
#
#     while iter_num < it:
#         accumulator = np.zeros(lr)
#         rho_r = np.zeros(lr)
#         r0 = vec[np.random.randint(n)][:3]
#
#         for i in range(lr):
#             # bis = balls_in_spheres(vec, Lx, Ly, Lz, *r0, r[i])
#             # if bis == 1:
#             #     accumulator[i] = 0
#             # else:
#             accumulator[i] = balls_in_spheres(vec_triple, *r0, r[i])
#         # accumulator[0] = accumulator[1]
#         # print(accumulator)
#         rho = accumulator[-1] / (4/3 * np.pi * R_max**3)
#
#         for j in range(lr - 1):
#             rho_r[j] = (accumulator[j+1] - accumulator[j]) / (4/3 * np.pi * (r[j+1]**3 - r[j]**3))
#
#         g_r = rho_r / rho
#         # print(g_r)
#         iter_num += 1
#
#     g_r /= it
#
#     return r, g_r, rho

if __name__ == '__main__':
    file_single = r'D:\Eytan\g_r_test\DOL\thermal_cube.dol'
    file_triple = r'D:\Eytan\g_r_test\DOL\thermal_cube_triple.dol'
    vec, n = read_from_file(file_single,0.01)
    Lx = 50
    Ly = 50
    Lz = 50

    vec_3 = triple(vec, Lx, Ly, Lz, file_triple)
