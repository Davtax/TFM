"""
Scrip to load some data previously computed and manipulate it or make some plots
"""
import numpy as np
import matplotlib.pyplot as plt

# Path the to file that we want to load
folder = 'data/'
file = 'STA_DQD_2HH_Test'
extension = '.npy'
file_dic = folder + file + extension

data = np.load(file_dic, allow_pickle=True)  # Load the data, allow_pickle enable to load a list

# Extract the data
tau_vec = data[3]
tf_vec = data[4]
results = data[0]

n_tf = len(tf_vec)
n_tau = len(tau_vec)

population_middle = np.zeros([n_tau, n_tf])
fidelity = np.zeros([n_tau, n_tf])
for i in range(0, n_tau):
	for j in range(0, n_tf):
		index = i * n_tf + j
		temp = results[index]
		population_middle[i, j] = np.max(temp[:, 2])
		fidelity[i, j] = temp[-1, 1]

fig, ax = plt.subplots()
pos = ax.imshow(fidelity.transpose(), origin='lower', cmap='jet', aspect='auto', extent=[tau_vec[0], tau_vec[-1], tf_vec[0], tf_vec[-1]])
cbar = fig.colorbar(pos, ax=ax)
