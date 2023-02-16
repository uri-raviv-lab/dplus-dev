import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from os.path import join, dirname

from dplus.Amplitudes import Amplitude

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import EmbeddedLocalRunner

from tests.test_settings import USE_GPU

root_path = os.path.dirname(os.path.abspath(__file__))
file_dir = join(dirname( __file__ ), "files_for_tests")

def test_generate_1():
    state_file_path = os.path.join(root_path, "files_for_tests", "sphere5points.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU)
    runner = EmbeddedLocalRunner()
    res = runner.generate(calc_input)

    status = runner.get_job_status()

    while status and status['isRunning'] and status['code']==-1:
        status = runner.get_job_status()
        time.sleep(0.1)
    print("end", status)
    if status['code'] == 0:
        result = runner.get_generate_results(calc_input)
        print(result.processed_result)
        if result.error["code"] != 0:
            print("Result returned error:", result.error)
    else:
        print("error", status)

    model_ptrs = runner.get_model_ptrs()
    print('model_ptrs', model_ptrs)
    for ptr in model_ptrs:
        print('ptr:', ptr)
    ptr = model_ptrs[-1]
    runner.save_amp(ptr, "amp-{}.ampj".format(ptr))
    print("the Amp was saved")

def test_generate_2():
    state_file_path = os.path.join(root_path, "files_for_tests", "sphere5points.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU)
    runner = EmbeddedLocalRunner()
    res = runner.generate(calc_input)
    print(res)
    plt.plot(res.signal.x, res.signal.y)
    plt.show()

def print_2D_result_to_csv(res, filename):
    f = open(filename, 'w')
    for row in res:
        str = ""
        for v in row:
            str += f"{v}, "
        str += '\n'
        f.write(str)
    f.close()

def test_save_amp():

    model = "helix"
    state_name = model + ".state"
    state_file_path = os.path.join(root_path, "files_for_tests", state_name)

    runner = EmbeddedLocalRunner()
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU, is2D=True)
    calc_input.DomainPreferences.use_grid = True
    gen_res = runner.generate2D(calc_input)

    amp_filename = join(file_dir, model + ".ampj")
    runner.save_amp(calc_input.Domain.populations[0].children[0].model_ptr, amp_filename)


def test_generate_2D_compare_to_amp():
    model = "helix"

    res_dir = os.path.join(root_path, "files_for_tests", "compare_" + model + "_res_" + time.strftime("%m%d_%H%M"))
    os.makedirs(res_dir, exist_ok=True)

    state_name = model + ".state"
    state_file_path = os.path.join(root_path, "files_for_tests", state_name)
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU, is2D=True)
    runner = EmbeddedLocalRunner()
    
    # Generate 2D:
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU, is2D=True)
    gen_res = runner.generate2D(calc_input)
    gen_result_name = model + "_generate2d_result.txt"
    gen_csv_file_name = os.path.join(res_dir, gen_result_name)
    print_2D_result_to_csv(gen_res.y, gen_csv_file_name)


     # Amplitude:
    #my_amp = runner.get_amp(calc_input.Domain.populations[0].children[0].model_ptr)
    amp_filename = join(file_dir, model + ".ampj")
    my_amp = Amplitude.load(amp_filename)
    out_with_amp = my_amp.get_intensity(calc_input.DomainPreferences.q_max, 
    calc_input.DomainPreferences.generated_points + 1, 
    max_iter=calc_input.DomainPreferences.orientation_iterations)

    amp_result_name = model + "_amp_result.txt"
    amp_csv_file_name = os.path.join(res_dir, amp_result_name)
    print_2D_result_to_csv(out_with_amp, amp_csv_file_name)
   
    # diff
    diff = [[0 for x in range(len(out_with_amp))] for y in range(len(out_with_amp))] 
    for x in range(len(out_with_amp)):
        for y in range(len(out_with_amp)):
            amp = out_with_amp[x][y]
            gen = list(gen_res.y)[x][y]
            if amp < gen:
                diff[x][y] = 1 - (amp / gen)
            elif gen < amp:
                diff[x][y] = 1 - (gen / amp)
            else: 
                diff[x][y] = 0

    print(max(max(diff)))

    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    im = ax.pcolormesh(diff, cmap=cm.gray, edgecolors='white', linewidths=0,
                   antialiased=True)
    fig.colorbar(im)
    # plt.imshow(list(diff), origin='lower', norm=colors.LogNorm(vmin=0, vmax=max(max(out_with_amp))))
    plt.show()

def test_generate_2D():
    from time import time
    #state_file_path = os.path.join(root_path, "files_for_tests", "uhc.state")
    state_file_path = os.path.join(root_path, "files_for_tests", "hybrid_spaceFilling_AND_UHC.state")
    #state_file_path = os.path.join(root_path, "files_for_tests", "1jff.state")
    #state_file_path = os.path.join(root_path, "files_for_tests", "helix.state")
    #state_file_path = os.path.join(root_path, "files_for_tests", "hybrid_spaceFilling_AND_UHC.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU, is2D=True)
    runner = EmbeddedLocalRunner()
    time_1 = time()
    res = runner.generate2D(calc_input)
    time_2 = time()
    print(f"runner.generate2D took {time_2 - time_1} seconds.")
    
    res_file = state_file_path + ".csv"

    print_2D_result_to_csv(res.y, res_file)

    plt.imshow(list(res.y), origin='lower', norm=colors.LogNorm(vmin=0, vmax=max(max(list(res.y)))))
    plt.show()

def test_generate_2D_with_timers():
    from time import time
    #state_file_path = os.path.join(root_path, "files_for_tests", "uhc.state")
    #state_file_path = os.path.join(root_path, "files_for_tests", "sphere200points.state")
    #state_file_path = os.path.join(root_path, "files_for_tests", "1jff.state")
    # state_file_path = os.path.join(root_path, "files_for_tests", "helix.state")
    state_file_path = os.path.join(root_path, "files_for_tests", "hybrid_spaceFilling_AND_UHC.state")
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU, is2D=True)
    runner = EmbeddedLocalRunner()
    time_1 = time()
    res = runner.generate2D(calc_input)
    time_2 = time()
    print(f"runner.generate2D took {time_2 - time_1} seconds.")
    # res = runner.generate(calc_input)
    my_amp = runner.get_amp(calc_input.Domain.populations[0].children[0].model_ptr)
    time_3 = time()
    out = my_amp.get_intensity(calc_input.DomainPreferences.q_max, 101, max_iter=10000)
    time_4 = time()
    print(f"my_amp.get_intensity took {time_4 - time_3} seconds.")
    diff = np.array(res.y) - np.array(out)
    # res_file = os.path.join(root_path, "files_for_tests", "1jff_res.csv")
    diff_file = os.path.join(root_path, "files_for_tests", "1jff_diff.csv")
    f = open(diff_file, 'w')
    for row in diff:
        str = ""
        for v in row:
            str += f"{v}, "
        str += '\n'
        f.write(str)
    f.close()

    # plt.imshow(list(res.y), origin='lower', norm=colors.LogNorm(vmin=0, vmax=max(max(list(res.y)))))
    # plt.imshow(out, origin='lower', norm=colors.LogNorm(vmin=0, vmax=np.max(out)))
    plt.imshow(diff, origin='lower', norm=colors.LogNorm(vmin=0, vmax=np.max(diff)))
    plt.colorbar()
    plt.show()

def test_scripted_symmetry():
    state_file_path = r'C:\temp\scriptedSymmetry.state'
    calc_input = CalculationInput.load_from_state_file(state_file_path, USE_GPU)
    runner = EmbeddedLocalRunner()
    res = runner.generate(calc_input)
    print(res)

if __name__ == '__main__':
    #test_generate_2D_compare_to_amp()
    test_generate_2D()
