import numpy as np
import csv
import argparse


class ResultEDConverter:
    def __init__(self, _coeff: float, _eED: float) -> None:
        self.coeff = _coeff
        self.eED = _eED


def find_and_write_amp():
    amp_list_D = np.array([])
    amp_list_E = np.array([])
    for filename in [r'.\helper_files\DPlus_FF.csv', r'.\helper_files\EPlus_FF.csv']:
        with open(filename, 'r') as file:
            my_list = csv.reader(file)
            for line in my_list:
                if line[0][0] == "#":
                    continue
                if 'EPlus_FF' in filename:
                    amp = 0
                    for j in line[:-1:2]:
                        amp += float(j)
                    amp_list_E = np.append(amp_list_E, [line[-1], float(amp)])
                else:
                    amp = 0
                    for j in line[:-1:2]:
                        amp += float(j)
                    amp_list_D = np.append(
                        amp_list_D, np.array([line[-1], amp]))

    amp_list_D = np.reshape(amp_list_D, [len(amp_list_D)//2, 2])
    amp_list_E = np.reshape(amp_list_E, [len(amp_list_E)//2, 2])

    filename_out = [r'.\helper_files\D_amp.csv', r'.\helper_files\E_amp.csv']
    for file_out in filename_out:
        with open(file_out, 'w', newline='') as file:
            my_list = csv.writer(file)
            if 'D_amp' in file_out:
                my_list.writerows(amp_list_D)
            else:
                my_list.writerows(amp_list_E)
    return


def find_coeff(atoms, occur, ED):
    filename_out_list = [r'.\helper_files\D_amp.csv',
                         r'.\helper_files\E_amp.csv']
    amp_sum = np.zeros(2)
    i = 0
    for filename_out in filename_out_list:
        with open(filename_out) as file:
            my_list = csv.reader(file)  # , quoting=csv.QUOTE_NONNUMERIC)
            for line in my_list:
                if not line[0] in atoms:
                    continue
                else:
                    if float(line[1]) != 0:
                        amp_sum[i] += float(line[1]) * \
                            float(occur[line[0] == atoms])
                    else:

                        raise KeyError('The atom ' + line[0] + ' is not an atom for which we can calculate the '
                                                               'scattering length...')
        i += 1
    coeff = (amp_sum[0] / amp_sum[1]) ** 2
    eED = ED / coeff

    return ResultEDConverter(coeff, eED)


def different_atoms(file):
    with open(file, encoding='utf-8') as pdb:
        atom_list = np.array(['Fake'])
        atom_reps = np.array([0])
        for line in pdb:
            if (line[:6] == 'ATOM  ') | (line[:6] == 'HETATM'):
                atm_type = line[76:78].replace(' ', '')
                if any(atm_type == atom_list):
                    atom_reps[atm_type == atom_list] += 1
                    # continue
                else:
                    atom_list = np.append(atom_list, atm_type)
                    atom_reps = np.append(atom_reps, 1)
                    with open(atm_type + r'.pdb', 'w', encoding='utf-8') as pdb:
                        changed_line = line[:30] + \
                            '   0.000   0.000   0.000' + line[54:]
                        pdb.write(changed_line)

    atom_list = atom_list[1:]
    atom_reps = atom_reps[1:]
    return atom_list, atom_reps


def convert(args):
    if args.PDB != '':
        atoms, occur = different_atoms(args.PDB)
        result = find_coeff(atoms, occur, args.ED)
    else:
        result = find_coeff(np.array(args.A), np.array(args.N), args.ED)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conversion of ED to scattering length of electrons.')
    parser.add_argument('--ED', type=float, default=333,
                        help='The electron density of the what you wish to convert.')
    parser.add_argument(
        '--A', type=str, default=['H', 'O'], nargs='+', help='The atoms in the molecular formula.')
    parser.add_argument('--N', type=int, default=[2, 1], nargs='+', help='Number of occurrences of each atom in '
                                                                         'the molecular formula (in the same order as '
                                                                         'the atoms were given).')
    parser.add_argument('--PDB', type=str, default='', help='The .pdb file of the molecule you want to convert. If a '
                                                            '.pdb file is given, A and N do not need to be given.')
    args = parser.parse_args()

    result = convert(args)
    print('The conversion coefficient is:', result.coeff)
    print('The electron scattering length is now:', result.eED)
