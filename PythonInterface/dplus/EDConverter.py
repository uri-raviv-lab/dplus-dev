import os
import numpy as np
import csv
import argparse


class ResultEDConverter:
    def __init__(self, _coeff: float, _eED: float) -> None:
        self.coeff = _coeff
        self.eED = _eED

    def __repr__(self) -> str:
        return f"The conversion coefficient is: {self.coeff}, The electron scattering length is now: {self.eED}"


# def find_and_write_amp():
#     amp_list_D = np.array([])
#     amp_list_E = np.array([])
#     for filename in [r'.\dplus\helper_files\DPlus_FF.csv', r'.\dplus\helper_files\EPlus_FF.csv']:
#         with open(filename, 'r') as file:
#             my_list = csv.reader(file)
#             for line in my_list:
#                 if line[0][0] == "#":
#                     continue
#                 if 'EPlus_FF' in filename:
#                     amp = 0
#                     for j in line[:-1:2]:
#                         amp += float(j)
#                     amp_list_E = np.append(amp_list_E, [line[-1], float(amp)])
#                 else:
#                     amp = 0
#                     for j in line[:-1:2]:
#                         amp += float(j)
#                     amp_list_D = np.append(
#                         amp_list_D, np.array([line[-1], amp]))

#     amp_list_D = np.reshape(amp_list_D, [len(amp_list_D)//2, 2])
#     amp_list_E = np.reshape(amp_list_E, [len(amp_list_E)//2, 2])

#     filename_out = [r'.\dplus\helper_files\D_amp.csv', r'.\dplus\helper_files\E_amp.csv']
#     for file_out in filename_out:
#         with open(file_out, 'w', newline='') as file:
#             my_list = csv.writer(file)
#             if 'D_amp' in file_out:
#                 my_list.writerows(amp_list_D)
#             else:
#                 my_list.writerows(amp_list_E)
#     return

test_dir = os.path.join(os.path.dirname(__file__), "helper_files")
def find_coeff(atoms, occur, ED):
    filename_out_list = [os.path.join(test_dir, "D_amp.csv"), os.path.join(test_dir, "E_amp.csv")]
    amp_sum = np.zeros(2)
    i = 0
    for filename_out in filename_out_list:
        with open(filename_out, 'r') as file:
            my_list = csv.reader(file)
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
    with open(file, 'r', encoding='utf-8') as pdb:
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
                    # with open(atm_type + r'.pdb', 'w', encoding='utf-8') as pdb:
                    #     changed_line = line[:30] + \
                    #         '   0.000   0.000   0.000' + line[54:]
                    #     pdb.write(changed_line)

    atom_list = atom_list[1:]
    atom_reps = atom_reps[1:]
    return atom_list, atom_reps


def convert(ed=333, a=['H', 'O'], pdb='fake', n=[2, 1]):
    if not ed or (pdb != 'fake' and np.size(a) and np.size(n)):
        raise ValueError("Invalid input parameters. ed={ed}, pdb={pdb}, a={a}, n={n}")
    elif pdb != 'fake':
        atoms, occur = different_atoms(pdb)
        result = find_coeff(atoms, occur, ed)
    else:
        result = find_coeff(np.array(a), np.array(n), ed)
    return result

def main():
    args = {'ed': 333, 'a': np.array([]), 'n': np.array([]), 'pdb': 'fake'}
    print('---------------------------------Conversion of ED to scattering length of '
          'electrons---------------------------------\n')
    args['ed'] = np.float64(input('\nWhat is the system electron density? '))
    pdb_or_atoms = input('\nEnter whether you want to enter a PDB file ("P") or a list of atoms ("A"): ')
    while not ((pdb_or_atoms == 'A') or (pdb_or_atoms == 'P')):
        # print(pdb_or_atoms, type(pdb_or_atoms), pdb_or_atoms == 'P')
        pdb_or_atoms = input('\nYou entered the wrong argument, you can only either enter "A" or "P", what is your '
                             'choice? ')
    if pdb_or_atoms == 'A':
        add_atom = True
        while add_atom:
            args['a'] = np.append(args['a'], input("\nWhat atom do you want to add? "))
            args['n'] = np.append(args['n'], int(input("\nHow many repetitions does the atom have? ")))
            add_atom = int(input('\nDo you want to add another atom? (0 - No, 1 - Yes) '))
            # print(type(add_atom))

    elif pdb_or_atoms == 'P':
        args['pdb'] = input('\nWhat is the location of the pdb file? ')
    # print(args['ed'], args['a'], args['pdb'], args['n'])
    # try:
    print(args['pdb'])
    result = convert(args['ed'], args['a'], args['pdb'], args['n'])
    # except:
    #     input('\nPress any key to close')

    print('\nThe conversion coefficient is: ', result.coeff)
    print('\nThe electron scattering length is now: ', result.eED)

if __name__ == '__main__':
    main()
    input('\nPress any key to close')
    # parser = argparse.ArgumentParser(
    #     description='Conversion of ED to scattering length of electrons.')
    # parser.add_argument('--ed', type=float, default=333,
    #                     help='The electron density of what you wish to convert.')
    # parser.add_argument(
    #     '--a', type=str, default=['H', 'O'], nargs='+', help='The atoms in the molecular formula.')
    # parser.add_argument('--n', type=int, default=[2, 1], nargs='+', help='Number of occurrences of each atom in '
    #                                                                      'the molecular formula (in the same order as '
    #                                                                      'the atoms were given).')
    # parser.add_argument('--pdb', type=str, default='fake', help='The .pdb file of the molecule you want to convert. '
    #                                                             'If a .pdb file is given, A and N do not need to be '
    #                                                             'given.')
    # args = parser.parse_args()
    #
    # result = convert(args.ed, args.a, args.pdb, args.n)

