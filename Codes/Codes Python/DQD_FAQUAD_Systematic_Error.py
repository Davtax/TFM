import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest, hamiltonian_stochastic_detunning
from general_functions import (solve_system_unpack, sort_solution, save_data, compute_adiabatic_parameter, compute_parameters_interpolation,
	compute_eigensystem, compute_limits, stochastic_noise)
from telegram_bot import message_telegram, image_telegram
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from plotting_functions import save_figure
import time as timer
from scipy.constants import h, e

workers = 8

hbar_muev_ns = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman spliting (ueV)
u = 2000  # Intradot interaction (ueV)
tau = 4
l2 = tau * 0.4
l1 = l2 / 100

limit1 = 0.999
limit2 = 0.999
state_1 = 0
state_2 = 1
adiabatic_state = 1

lim = 10 ** 4
eps_vector_temp = np.linspace(-lim, lim, 2 ** 14) * ET - u
parameters_temp = [eps_vector_temp, u, ET, tau, l1, l2]
lim_T, lim_S = compute_limits(hamiltonian_2QD_1HH_Lowest, parameters_temp, limit1, limit2, state_1, state_2, adiabatic_state, eps_vector_temp, [tau],
                              0, 3, filter_bool=False)

n_eps = 2 ** 15 + 1  # This number of elements is required for the romb method of integration used below
eps_vector = np.linspace(lim_T[0], lim_S[0], n_eps)

n_tf = 300
tf_vec = np.linspace(0.1, 25, n_tf, endpoint=True)

n_error = 50+1
error_vector = np.linspace(-1, 1, n_error, endpoint=True) * 1e-2

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[0, 0] = 1  # Initially the only state populated is the triplet (in our basis the first state)

partial_hamiltonian = np.zeros([n_eps, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = 1

parameters = [eps_vector, u, ET, tau, l1, l2]  # List with the parameters of the system
energies, states = compute_eigensystem(parameters, hamiltonian_2QD_1HH_Lowest)
factors, c_tilde = compute_adiabatic_parameter(eps_vector, states, energies, 1, hbar=hbar_muev_ns, partial_Hamiltonian=partial_hamiltonian)
s, eps_sol = compute_parameters_interpolation(eps_vector, factors, c_tilde, method_1=False)

n_t = 10 ** 3
args = []
for i in range(0, n_tf):
	time = np.linspace(0, tf_vec[i], n_t)
	for j in range(0, n_error):
		temp = [i * n_error + j, time, density0, [eps_sol, u, ET, tau, l1, l2, error_vector[j]], hamiltonian_2QD_1HH_Lowest,
		        {'normalization': tf_vec[i], 'hbar': hbar_muev_ns}]
		args.append(temp)

if __name__ == '__main__':
	results_list = []
	pbar = tqdm(total=n_tf * n_error, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')
	start = timer.perf_counter()
	with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
		results = executor.map(solve_system_unpack, args, chunksize=4)
		for result in results:
			results_list.append(result)
			pbar.update()

	pbar.refresh()
	finish = timer.perf_counter()

	results = sort_solution(results_list)

	density_matrix = []
	probabilities = []
	for temp in results:
		density_matrix.append(temp[0])
		probabilities.append(temp[1])

	population_middle = np.zeros([n_tf, n_error])
	fidelity = np.zeros([n_tf, n_error])
	for i in range(0, n_tf):
		for j in range(0, n_error):
			index = i * n_error + j
			temp = probabilities[index]
			population_middle[i, j] = np.max(temp[:, 2])
			fidelity[i, j] = temp[-1, 1]

	fig, ax = plt.subplots()
	pos = ax.imshow(fidelity.transpose(), origin='lower', cmap='jet', aspect='auto',
	                extent=[tf_vec[0], tf_vec[-1], error_vector[0], error_vector[-1]], interpolation='quadric', vmin=0, vmax=1)
	cbar = fig.colorbar(pos, ax=ax)

	total_time = finish - start

	if total_time > 60 * 60:
		units = '(hr)'
		total_time /= (60 * 60)
	elif total_time > 60:
		units = '(min)'
		total_time /= 60
	else:
		units = ' (s)'

	file_name = 'FAQUAD_DQD_2HH_Systematic_Error'
	message_telegram('DONETE: {} {}x{}. Total time: {:.2f} {}'.format(file_name, n_tf, n_error, total_time, units))
	save_figure(fig, file_name, overwrite=True, extension='png', dic='data/')
	image_telegram('data/' + file_name + '.png')
	save_data(file_name, [results, error_vector, tf_vec, ['results', 'Gamma_vec', 'tf_vec']])
