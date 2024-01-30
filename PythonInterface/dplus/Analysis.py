import os
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


def svd_analysis(folder_path, extension=['out', 'dat'], num_points=1536, delimiter='\t', skiprows=1, I_col=1,
                 sigma_col=2, num_components=5):
    '''Performs SVD analysis on the data according to the number of independent vectors and returns for each number
    of vectors the average of the squared normalized-residuals up to that number of vectors.
    input:
    folder_path - path to the folder containing the data files
    extension - extension of the data files
    num_points - number of points in each file (should be the same for all files)
    delimiter - delimiter of the data in the file (default '\t')
    skiprows - number of rows to skip in the file (default 1)
    I_col - column of the data in the file (default 1)
    sigma_col - column of the measuring errors in the file (default 2)
    num_components - number of independent vectors to use in the analysis (up to this number, incl.)

    output:
    R - array of the average of the squared normalized-residuals for the number of independent vectors (according to
    the index + 1)
    '''

    def get_file_names(folder_path, extension=['out', 'dat']):
        file_names = []
        for file in os.listdir(folder_path):
            if file.endswith(tuple(extension)):
                file_names.append(folder_path + "\\" + file)
        return file_names

    def concat_data(file_names, num_points=1536, delimiter='\t', skiprows=1, I_col=1, sigma_col=2):

        data = np.zeros([num_points, len(file_names)])
        sigmas = np.zeros([num_points, len(file_names)])
        for i, file in enumerate(file_names):
            data[:, i] = np.loadtxt(file, delimiter=delimiter, skiprows=skiprows, usecols=I_col)
            sigmas[:, i] = np.loadtxt(file, delimiter=delimiter, skiprows=skiprows, usecols=sigma_col)

        return data, sigmas

    file_names = get_file_names(folder_path, extension)
    data, sigmas = concat_data(file_names, num_points=num_points, delimiter=delimiter, skiprows=skiprows,
                               I_col=I_col, sigma_col=sigma_col)

    U, s, V = svd(data, full_matrices=False, check_finite=False)
    R = np.zeros(num_components)
    for i in range(1, num_components+1):
        reconstructed_data = np.dot(U[:, :i], np.dot(np.diag(s[:i]), V[:i, :]))
        residuals = data - reconstructed_data
        R_k = residuals / sigmas
        R[i-1] = np.mean(np.power(R_k, 2))

    return R


def zimm_analysis():
    ## TODO: implement zimm analysis
    pass

if __name__ == '__main__':
    folder = r"D:\Eytan\Synchotron\Hamburg_08_23\p3l-raviv-2023-08-14-SECSAXS_Tub_10GTP\analysis\By_Eytan\Peak_2"
    num_of_components = 109
    R = svd_analysis(folder, num_components=num_of_components)

    plt.plot(np.linspace(1, num_of_components, num_of_components), R)
    plt.show()

