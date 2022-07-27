"""
Created on Wed Feb 16 13:30:34 2022

@author: Eytan Balken
"""

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner
from dplus.DataModels.models import Sphere
from dplus.DataModels import ManualSymmetry
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import dplus.g_r as g
import time as t
import numpy as np

def test_1():
    print('test 1\n')

    a, b, c = np.array([5,0,0]), np.array([0,5,0]), np.array([0,0,5])
    rep_a, rep_b, rep_c = 10, 10, 10
    Lx, Ly, Lz = 50, 50, 50
    r_max, dr = 25, 1e-2
    q_max = 100
    dq = 0.1
    integ_max = [5.2, 8.8]

    filename_single = r'D:\Eytan\g_r_test\DOL\cube.dol'
    filename_triple = r'D:\Eytan\g_r_test\DOL\triple_cube.dol'

    g.build_crystal(a, b, c, rep_a, rep_b, rep_c, filename_single)

    I_sfs = CalculationInput()
    f_sfs = CalculationInput()
    # I_sfs.DomainPreferences.signal_file = r'D:\Eytan\g_r_test\out\cube.out'
    # f_sfs.DomainPreferences.signal_file = r'D:\Eytan\g_r_test\OUT\sphere_2p5.out'
    sp = Sphere()
    ms = ManualSymmetry()

    sp.radius = 2.5
    sp.use_grid = True

    ms.read_from_dol(filename_single)
    ms.children.append(sp)
    ms.use_grid = False

    I_sfs.DomainPreferences.grid_size = 50
    I_sfs.DomainPreferences.generated_points = 1500
    I_sfs.DomainPreferences.orientation_iterations = 1e6
    I_sfs.DomainPreferences.convergence = 0.001
    I_sfs.DomainPreferences.q_max = 15
    I_sfs.DomainPreferences.orientation_method = 'Adaptive (VEGAS) Monte Carlo'

    f_sfs.DomainPreferences.grid_size = 50
    f_sfs.DomainPreferences.generated_points = 1500
    f_sfs.DomainPreferences.orientation_iterations = 1e6
    f_sfs.DomainPreferences.convergence = 0.001
    f_sfs.DomainPreferences.q_max = 15
    f_sfs.DomainPreferences.orientation_method = 'Adaptive (VEGAS) Monte Carlo'

    I_sfs.Domain.populations[0].add_model(ms)
    f_sfs.Domain.populations[0].add_model(sp)

    runner = LocalRunner()
    I_calc = runner.generate(I_sfs)
    print('calculated I')
    I_calc.save_to_out_file(filename_single)
    f_calc = runner.generate(f_sfs)
    print('calculated f')
    f_calc.save_to_out_file(r'D:\Eytan\g_r_test\OUT\sphere_'+str(sp.radius)+r'.out')

    n_sfs = rep_a * rep_b * rep_c
    q = np.array(list(f_calc.graph.keys()))
    # q = np.array(f_sfs.x)

    S_q_sfs = g.S_Q_from_I(np.array(list(I_calc.graph.values())), np.array(list(f_calc.graph.values())), n_sfs)
    # S_q_sfs = g.S_Q_from_I(np.array(I_sfs.y), np.array(f_sfs.y), n_sfs)
    print('calculated S(q) from I')
    q_mod, S_Q_mod, rho_mod = g.S_Q_from_model(filename_single, q_max=q_max, dq=dq)
    print('calculated S(q) from model')
    r_mod, g_r_mod, rho, rad_balls = g.g_r_from_model(filename_single, Lx, Ly, Lz, file_triple=filename_triple, radius=0
                                                      , r_max=r_max, dr=dr, Number_for_average=1)
    print('calculated g(r) from model')
    r_sfs, g_r_sfs = g.g_r_from_s_q(q_mod, S_Q_mod, rho, r_max=r_max, type='Simpson')
    r_mod_s, g_r_mod_s = g.g_r_from_s_q(q_mod, S_Q_mod, rho, r_max=r_max, type='DST')
    print('calculated g(r) from S(q)')

    plt.figure()
    plt.semilogy(q, S_q_sfs, label='From I(q)')
    plt.semilogy(q_mod, S_Q_mod, label='From model')
    plt.title('S(q)')
    plt.xlabel('$q [nm^{-1}]$')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(r_mod, g_r_mod, label='From Model')
    plt.plot(r_sfs, g_r_sfs, label='From I(q) with Simpson')
    plt.plot(r_mod_s, g_r_mod_s, label='From I(Q) with DST')
    plt.xlabel('r[nm]')
    plt.title('g(r)')
    plt.legend()
    plt.show()

    for i in integ_max:
        print('N_mod = ', simpson(rho * g_r_mod[r_mod < i] * 4 * np.pi * r_mod[r_mod < i]**2, r_mod[r_mod < i]))
        print('N_simpson = ', simpson(rho * g_r_sfs[r_sfs < i] * 4 * np.pi * r_sfs[r_sfs < i]**2, r_sfs[r_sfs < i]))
        print('N_dst = ', simpson(rho * g_r_mod_s[r_mod_s < i] * 4 * np.pi * r_mod_s[r_mod_s < i]**2, r_mod_s[r_mod_s < i]))

    # k=0
    # while k<10:
    q_back_simps, s_q_back_simps = g.s_q_from_g_r(r_sfs, g_r_sfs, rho, q_max=q_max, type='Simpson')
    q_back_dst, s_q_back_dst = g.s_q_from_g_r(r_mod_s, g_r_mod_s, rho, q_max=q_max, type='DST')

    plt.figure()
    plt.semilogy(q, S_q_sfs, label='From I(q)')
    plt.semilogy(q_back_simps, s_q_back_simps, label='Backwards Simpson')
    plt.semilogy(q_back_dst, s_q_back_dst, label='Backwards DST')
    plt.title('S(q)')
    plt.xlabel('$q [nm^{-1}]$')
    plt.legend()
    plt.show()

    r_back_simps, g_r_back_simps = g.g_r_from_s_q(q_back_simps, s_q_back_simps, rho, r_max=r_max, type='Simpson')
    r_back_dst, g_r_back_dst = g.g_r_from_s_q(q_back_simps, s_q_back_simps, rho, r_max=r_max, type='DST')

    plt.figure()
    plt.plot(r_mod, g_r_mod, label='From Model')
    plt.plot(r_back_simps, g_r_back_simps, label='Backwards Backwards Simpson')
    plt.plot(r_back_dst, g_r_back_dst, label='Backwards Backwards DST')
    plt.title('g(r)')
    plt.xlabel('$r [nm]$')
    plt.legend()
    plt.show()
    # k+=1

    for i in integ_max:
        print('N_simps',i,' = ', simpson(rho * g_r_back_simps[r_back_simps < i] * 4 * np.pi * r_back_simps[r_back_simps < i]**2, r_back_simps[r_back_simps < i]))
        print('N_dst',i,' = ', simpson(rho * g_r_back_dst[r_back_dst < i] * 4 * np.pi * r_back_dst[r_back_dst < i]**2, r_back_dst[r_back_dst < i]))

    print('N_s(q) tot', simpson(rho * g_r_sfs * 4 * np.pi * r_sfs**2, r_sfs))
    print('N_mod tot', simpson(rho * g_r_mod * 4 * np.pi * r_mod**2, r_mod))
    print('N_mod_s(q) tot', simpson(rho * g_r_mod_s * 4 * np.pi * r_mod_s**2, r_mod_s))

def test_2():
    print('test 2\n')

    I_sfs = CalculationInput()
    sp = Sphere()
    sp.radius = 2.5
    sp.use_grid = True

    I_sfs.DomainPreferences.grid_size = 50
    I_sfs.DomainPreferences.generated_points = 1500
    I_sfs.DomainPreferences.orientation_iterations = 1e6
    I_sfs.DomainPreferences.convergence = 0.001
    I_sfs.DomainPreferences.q_max = 15
    I_sfs.DomainPreferences.orientation_method = 'Adaptive (VEGAS) Monte Carlo'
    I_sfs.Domain.populations[0].add_model(sp)

    runner = EmbeddedLocalRunner()
    I_calc = runner.generate(I_sfs)

    I_sfs.DomainPreferences.signal_file = r'D:\Eytan\g_r_test\out\cube.out'

    s_q_tuple = g.S_Q_from_I(I_sfs.y, I_sfs.y, 1)
    s_q_dict = g.S_Q_from_I(I_calc.graph.values(), I_calc.graph.values(), 1)

    # print(all(s_q_tuple == np.ones(len(I_sfs.y))) & all(s_q_dict == np.ones(len(I_calc.graph.values()))))
    assert all(s_q_tuple == np.ones(len(I_sfs.y))) & all(s_q_dict == np.ones(len(I_calc.graph.values())))

def test_4():
    print('test 4:')

    Lx_4, Ly_4, Lz_4 = 8.4603, 8.4603, 8.4603
    r_max_4, dr_4 = 2.5, 1e-2
    integ_max = np.array(range(40, 51, 1))/100
    q_max_4, dq_4 = 25, 25/1e3

    dol_name = r'\\raviv_backup\raviv_group\public\To_Transfer\Eytan\D+\NaCl\DOL\NaCl.dol'
    dol_name_triple = r'\\raviv_backup\raviv_group\public\To_Transfer\Eytan\D+\NaCl\DOL\NaCl_triple.dol'

    q_mod, S_Q_mod, rho_mod = g.S_Q_from_model(dol_name, q_max=q_max, dq=dq)
    r_mod, g_r_mod, rho_mod, rad_balls = g.g_r_from_model(dol_name, Lx, Ly, Lz, file_triple=dol_name_triple, r_max=r_max
                                               , dr=dr, Number_for_average=1, radius=0)

    r_simps, g_r_simps = g.g_r_from_s_q(q_mod, S_Q_mod, rho_mod, r_max=r_max, type='Simpson')
    r_dst, g_r_dst = g.g_r_from_s_q(q_mod, S_Q_mod, rho_mod, r_max=r_max, type='DST')
    for i in integ_max:
        print('For model, The number of NN up to', i, ':',
              simpson(rho_mod * g_r_mod[r_mod < i] * 4 * np.pi * r_mod[r_mod < i]**2, r_mod[r_mod < i]))
        print('For simpson, The number of NN up to', i, ':',
              simpson(rho_mod * g_r_simps[r_simps < i] * 4 * np.pi * r_simps[r_simps < i] ** 2, r_simps[r_simps < i]))
        print('For DST, The number of NN up to', i, ':',
              simpson(rho_mod * g_r_dst[r_dst < i] * 4 * np.pi * r_dst[r_dst < i] ** 2, r_dst[r_dst < i]))

    plt.figure()
    plt.semilogy(q_mod, S_Q_mod, label='Model')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(r_mod, g_r_mod, label='Model')
    plt.plot(r_simps, g_r_simps, label='Simpson')
    plt.plot(r_dst, g_r_dst, label='DST')
    plt.legend()
    plt.show()

def test_6():
    print('test 6: \n')

    a, b, c = np.array([5,0,0]), np.array([0,5,0]), np.array([0,0,5])
    rep_a, rep_b, rep_c = 10, 10, 10
    Lx, Ly, Lz = 50, 50, 50
    r_max, dr = 25, 1e-2
    integ_max = [5.2, 8.8]
    rand_perc = 0.05

    filename_single_6 = r'D:\Eytan\g_r_test\DOL\thermal_cube.dol'
    filename_triple_6 = r'D:\Eytan\g_r_test\DOL\thermal_triple_cube.dol'

    g.build_crystal(a, b, c, rep_a, rep_b, rep_c, filename_single_6, ran=rand_perc)

    q_mod_6, S_Q_mod_6, rho_mod_6 = g.S_Q_from_model(filename_single_6)
    print('calculated S(q) from model')
    r_mod_6, g_r_mod_6, rho_6, rad_balls_6 = g.g_r_from_model(filename_single_6, Lx, Ly, Lz, filename_triple_6, radius=0
                                                              , r_max=r_max, dr=dr, Number_for_average=10)
    print('calculated g(r) from model')
    r_simps_6, g_r_simps_6 = g.g_r_from_s_q(q_mod_6, S_Q_mod_6, rho_6, r_max=r_max, type='Simpson')
    r_dst_6, g_r_dst_6 = g.g_r_from_s_q(q_mod_6, S_Q_mod_6, rho_6, r_max=r_max, type='DST')
    print('calculated g(r) from S(q)')

    plt.figure()
    plt.semilogy(q_mod_6, S_Q_mod_6, label='From model')
    plt.title('S(q)')
    plt.xlabel('$q [nm^{-1}]$')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(r_mod_6, g_r_mod_6, label='From Model')
    plt.plot(r_simps_6, g_r_simps_6, label='From I(q) with Simpson')
    plt.plot(r_dst_6, g_r_dst_6, label='From I(Q) with DST')
    plt.xlabel('r[nm]')
    plt.title('g(r)')
    plt.legend()
    plt.show()

    print('N_mod = ', simpson(rho_6 * g_r_mod_6[r_mod_6 < 8.8] * 4 * np.pi * r_mod_6[r_mod_6 < 8.8]**2, r_mod_6[r_mod_6 < 8.8]))
    print('N_simpson = ', simpson(rho_6 * g_r_simps_6[r_simps_6 < 8.8] * 4 * np.pi * r_simps_6[r_simps_6 < 8.8]**2, r_simps_6[r_simps_6 < 8.8]))
    print('N_dst = ', simpson(rho_6 * g_r_dst_6[r_dst_6 < 8.8] * 4 * np.pi * r_dst_6[r_dst_6 < 8.8]**2, r_dst_6[r_dst_6 < 8.8]))

def test_7(a, b, c, rep_a, rep_b, rep_c, Lx, Ly, Lz, r_max, dr, q_max, dq, integ_max, rand_perc):
    print('test 7: \n')

    a_7, b_7, c_7 = np.array([5, 0, 0]), np.array([0, 5, 0]), np.array([0, 0, 5])
    rep_a_7, rep_b_7, rep_c_7 = 10, 10, 10
    Lx_7, Ly_7, Lz_7 = 50, 50, 50
    r_max_7, dr_7 = 20, 1e-2
    integ_max_7 = [5.7, 9.3]
    rand_perc_7 = 0.05
    q_max_7 = 15
    dq_7 = 15/1e3

    filename_single_7 = r'D:\Eytan\g_r_test\DOL\thermal_cube.dol'

    I_sfs = CalculationInput()
    I_sfs.DomainPreferences.signal_file = r'D:\Eytan\g_r_test\out\cube.out'
    f_sfs = CalculationInput()
    f_sfs.DomainPreferences.signal_file = r'D:\Eytan\g_r_test\DOL\sphere_2p5.out'
    n_sfs = 10 ** 3
    q = np.array(f_sfs.x)

    S_q_sfs = g.S_Q_from_I(np.array(I_sfs.y), np.array(f_sfs.y), n_sfs)

    q_mod_7, S_Q_mod_7, rho_mod_7, r_mod_7, g_r_mod_7 = 0, 0, 0, 0, 0

    n = 15

    for i in range(n):
        g.build_crystal(a, b, c, rep_a, rep_b, rep_c, filename_single_7, ran=rand_perc)
        q_temp, s_q_temp, rho_temp_S = g.S_Q_from_model(filename_single_7, q_max=q_max, dq=dq)
        r_temp, g_r_temp, rho_temp_g, rad_balls_7 = g.g_r_from_model(filename_single_7, Lx, Ly, Lz, radius=0, r_max=r_max
                                                                     , dr=dr, Number_for_average=1)
        q_mod_7 += q_temp
        S_Q_mod_7 += s_q_temp
        r_mod_7 += r_temp
        g_r_mod_7 += g_r_temp
        rho_mod_7 += rho_temp_g

    q_mod_7 /= n
    S_Q_mod_7 /= n
    r_mod_7 /= n
    g_r_mod_7 /= n
    rho_mod_7 /= n

    print('calculated S(q) from model')
    r_simps_7, g_r_simps_7 = g.g_r_from_s_q(q_mod_7, S_Q_mod_7, rho_mod_7, r_max=r_max, type='Simpson')
    r_dst_7, g_r_dst_7 = g.g_r_from_s_q(q_mod_7, S_Q_mod_7, rho_mod_7, r_max=r_max, type='DST')
    print('calculated g(r) from S(q)')

    plt.figure()
    plt.semilogy(q, S_q_sfs, label='From I(q) at 0C')
    plt.semilogy(q_mod_7, S_Q_mod_7, label='From thermal model')
    plt.title('S(q)')
    plt.xlabel('$q [nm^{-1}]$')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(r_mod_7, g_r_mod_7, label='From Model')
    plt.plot(r_simps_7, g_r_simps_7, label='From I(q) with Simpson')
    plt.plot(r_dst_7, g_r_dst_7, label='From I(Q) with DST')
    plt.xlabel('r[nm]')
    plt.title('g(r)')
    plt.legend()
    plt.show()

    for i in integ_max:
        print('N_mod', i, ' = ', simpson(rho_mod_7 * g_r_mod_7[r_mod_7 < i] * 4 * np.pi * r_mod_7[r_mod_7 < i]**2, r_mod_7[r_mod_7 < i]))
        print('N_simpson', i, ' = ', simpson(rho_mod_7 * g_r_simps_7[r_simps_7 < i] * 4 * np.pi * r_simps_7[r_simps_7 < i]**2, r_simps_7[r_simps_7 < i]))
        print('N_dst', i, ' = ', simpson(rho_mod_7 * g_r_dst_7[r_dst_7 < i] * 4 * np.pi * r_dst_7[r_dst_7 < i]**2, r_dst_7[r_dst_7 < i]))

    r_range_mod = (r_mod_7 > 4.3) & (r_mod_7 < 5.6)
    r_range_simps = (r_simps_7 > 4.3) & (r_simps_7 < 5.6)
    r_range_dst = (r_dst_7 > 4.3) & (r_dst_7 < 5.6)

    print('N_mod = ',
          simpson(rho_mod_7 * g_r_mod_7[r_range_mod] * 4 * np.pi * r_mod_7[r_range_mod] ** 2, r_mod_7[r_range_mod]))
    print('N_simpson = ',
          simpson(rho_mod_7 * g_r_simps_7[r_range_simps] * 4 * np.pi * r_simps_7[r_range_simps] ** 2,
                  r_simps_7[r_range_simps]))
    print('N_dst = ',
          simpson(rho_mod_7 * g_r_dst_7[r_range_dst] * 4 * np.pi * r_dst_7[r_range_dst] ** 2, r_dst_7[r_range_dst]))

# def test_8(file, Lx, Ly, Lz, r_max, dr):
#     t_new = t.process_time_ns()
#     r, g_r, rho = g.new_g_r(file, Lx, Ly, Lz, r_max, dr)
#     elapsed_t_new = t.process_time() - t_new
#     t_old = t.process_time_ns()
#     r_1, g_r_1, rho_1, rad = g.g_r_from_model(file, Lx, Ly, Lz, r_max=r_max, dr=dr)
#     elapsed_t_old = t.process_time() - t_old
#     print('old:', elapsed_t_old)
#     print('new:', elapsed_t_new)
#
#     plt.figure()
#     plt.plot(r, g_r, label='Spheres')
#     plt.plot(r_1, g_r_1, label='Binning')
#     plt.legend()
#     plt.title('g(r)')
#     plt.xlabel('r[nm]')
#     plt.show()
#     return elapsed_t_new, elapsed_t_old



if __name__ == '__main__':
    test_1()

    test_2()
    
    test_4()
    
    test_6()

    test_7()
    # filename_8 = r'D:\Eytan\g_r_test\DOL\cube.dol'
    # Lx, Ly, Lz = 50, 50, 50
    # r_max = 22.5
    # dr = 0.01
    # t_1, t_2 = test_8(filename_8, Lx, Ly, Lz, r_max, dr)

