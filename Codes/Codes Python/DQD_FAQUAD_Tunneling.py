import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest
from general_functions import (solve_system_unpack, sort_solution, save_data, compute_adiabatic_parameter, compute_parameters_interpolation,
	compute_eigensystem, compute_limits)
from telegram_bot import message_telegram, image_telegram
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from plotting_functions import save_figure
import time as timer
from scipy.constants import h, e

workers = 8


def compute_parameters(pack):
	index_parallel, limit_lower, limit_upper = pack
	counter = n_tf * index_parallel
	eps_vector_test = np.linspace(limit_lower[index_parallel], limit_upper[index_parallel], n_eps)
	parameters_temp = [eps_vector_test, u, ET, tau_vec[index_parallel], l1_vector[index_parallel], l2_vector[index_parallel]]
	energies, states = compute_eigensystem(parameters_temp, hamiltonian_2QD_1HH_Lowest)
	factors, c_tilde = compute_adiabatic_parameter(eps_vector_test, states, energies, initial_state=1)
	s, eps_sol = compute_parameters_interpolation(eps_vector_test, factors, c_tilde)
	parameters_temp[0] = eps_sol
	args_parallel = []
	for k in range(n_tf):
		time = np.linspace(0, tf_vec[k], 10**3, endpoint=True)
		args_parallel.append([counter, time, density0, parameters_temp, hamiltonian_2QD_1HH_Lowest])
		args_parallel[-1].append({'normalization': tf_vec[k], 'atol': 1e-8, 'rtol': 1e-6, 'hbar': hbar_muev_ns})
		counter += 1
	return [index_parallel, args_parallel]


hbar_muev_ns = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman splitting (ueV)
u = 2000  # Intradot interaction (ueV)

lim = 10 ** 4
eps_vector_temp = np.linspace(-lim, lim, 2 ** 15) * ET - u

n_tau = 5
tau_vec = np.linspace(0.1, 4, n_tau, endpoint=True)
n_tf = 30
tf_vec = np.linspace(0.1, 60, n_tf, endpoint=True)

l2_vector = tau_vec * 0.4
l1_vector = l2_vector / 100

parameters = [0, u, ET, 0, 0, 0]

limit1 = 0.999
limit2 = 0.999
state_1 = 0
state_2 = 1
adiabatic_state = 1

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[0, 0] = 1  # Initially the only state populated is the triplet (in our basis the first state)

n_eps = 2 ** 15 + 1

partial_hamiltonian = np.zeros([n_eps, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = 1

if __name__ == '__main__':
	start = timer.perf_counter()
	lim_T, lim_S = compute_limits(hamiltonian_2QD_1HH_Lowest, parameters, limit1, limit2, state_1, state_2, adiabatic_state, eps_vector_temp,
	                              [tau_vec, l1_vector, l2_vector], 0, [3, 4, 5], filter_bool=False, window=51)

	results_list = []

	args = []
	for i in range(n_tau):
		args.append([i, lim_T, lim_S])

	with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
		results = executor.map(compute_parameters, args, chunksize=4)

		for result in results:
			results_list.append(result)

	args_temp = sort_solution(results_list)

	args = []
	for i in args_temp:
		for j in i:
			args.append(j)

	print('\nParameters computed')

	results_list = []

	pbar = tqdm(total=n_tf * n_tau, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')

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

	population_middle = np.zeros([n_tau, n_tf])
	fidelity = np.zeros([n_tau, n_tf])
	for i in range(0, n_tau):
		for j in range(0, n_tf):
			index = i * n_tf + j
			temp = probabilities[index]
			population_middle[i, j] = np.max(temp[:, 2])
			fidelity[i, j] = temp[-1, 1]

	fig, ax = plt.subplots()
	pos = ax.imshow(fidelity, origin='lower', cmap='jet', aspect='auto', extent=[tf_vec[0], tf_vec[-1], tau_vec[0], tau_vec[-1]], vmin=0, vmax=1)
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

	file_name = 'FAQUAD_DQD_2HH_Tunnelling'
	message_telegram('DONETE: {} {}x{}. Total time: {:.2f} {}'.format(file_name, n_tau, n_tf, total_time, units))
	save_figure(fig, file_name, overwrite=True, extension='png', dic='data/')
	image_telegram('data/' + file_name + '.png')
	save_data(file_name, [results, tau_vec, tf_vec, ['results', 'tau_vec', 'tf_vec']])
