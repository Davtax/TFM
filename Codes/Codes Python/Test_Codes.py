import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import modify_plot, save_figure

# Path the to file that we want to load
folder = 'data/'
file = 'FAQUAD_DQD_2HH_Decoherence_reduced'
extension = '.npy'
file_dic = folder + file + extension

data = np.load(file_dic, allow_pickle=True)  # Load the data, allow_pickle enable to load a list
print(data[-1])

final_density_matrix = data[0]
tf_vec = data[1]
Gamma_vec = data[2]

n_tf = len(tf_vec)
n_Gamma = len(Gamma_vec)

probabilities = np.zeros([3, n_tf * n_Gamma])

for i in range(0, n_tf*n_Gamma):
	temp = final_density_matrix[i]
	probabilities[:, i] = np.abs(np.diag(temp))

fidelity = probabilities[1, :].reshape([n_tf, n_Gamma])
population_middle = probabilities[2, :].reshape([n_tf, n_Gamma])
