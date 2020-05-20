import numpy as np
from hamiltonians import hamiltonian_two_sites
from general_functions import (compute_eigensystem, compute_adiabatic_parameter, compute_parameters_interpolation, compute_period,
	solve_system_unpack, sort_solution)
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.constants import h, e
from tqdm import tqdm
import time as timer
from multiprocessing import Pool

workers = 8

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]

J = 10
U = 22.3 * J

n_Delta = 2 ** 15 + 1  # This number of elements is required for the romb method of integration used below
limit_Delta = 66.7
Delta_vector = np.linspace(limit_Delta, -limit_Delta, n_Delta, endpoint=True) * J

parameters = [Delta_vector, U, J]  # List with the parameters of the system

labels = [r'$(\epsilon+u)/E_Z$', r'$E_n/J$']  # Labels of the figure

# Compute and plot the eigenenergies of the system
energies, states = compute_eigensystem(parameters, hamiltonian_two_sites, plot=False)

partial_hamiltonian = np.zeros([n_Delta, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = -1
partial_hamiltonian[:, 0, 0] = 1

factors, c_tilde = compute_adiabatic_parameter(Delta_vector, states, energies, 0, hbar=hbar, partial_Hamiltonian=partial_hamiltonian)

# Solve the EDO to obtain the dependency of eps with the parameter s
s, Delta_sol = compute_parameters_interpolation(Delta_vector, factors, c_tilde, method_1=False)  # parameters = [Delta_sol, U, J]

T = compute_period(Delta_sol, hamiltonian_two_sites, parameters, hbar, index=0, state=0)

n_tf = 400
tf_vec = np.linspace(0.1 * hbar / J, 26 * hbar / J, n_tf)

n_t = 10 ** 3

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[2, 2] = 1  # Initially the only state populated is the triplet (in our basis the first state)

args = []
for i in range(0, n_tf):
	time = np.linspace(0, tf_vec[i], n_t)  # Time vector in which compute the solution of the population
	temp = [i, time, density0, [Delta_sol, U, J], hamiltonian_two_sites,
	        {'normalization': tf_vec[i], 'atol': 1e-8, 'rtol': 1e-6, 'hbar': hbar}]  # List of parameters and default parameters as a dic
	args.append(temp)

if __name__ == '__main__':
	start = timer.perf_counter()
	results_list = []  # Empty list in which save the async results
	pbar = tqdm(total=n_tf, desc='Processing', ncols=90, bar_format='{l_bar}{bar}{r_bar}')

	with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
		results = executor.map(solve_system_unpack, args)
		for result in results:
			results_list.append(result)
			pbar.update()

	# pool = Pool()
	# for i, result in enumerate(pool.imap_unordered(solve_system_unpack, args), 1):  # Iterate over all the desired parameters
	# 	results_list.append(result)  # Save the result
	# 	pbar.update()  # Update the progress bar
	# pool.terminate()  # Terminate the pool

	# for i in range(0, n_tf):
	# 	result = solve_system_unpack(args[i])
	# 	results_list.append(result)
	# 	pbar.update()

	final = timer.perf_counter()

	print('\nThe computation took {} s'.format(final - start))

	results_sort = sort_solution(results_list)  # Sort the async results

	fidelity = np.zeros(n_tf)
	for i in range(0, n_tf):
		fidelity[i] = results_sort[i][-1, 0]  # Extract the data from the final time (i), last time computed (-1), and the state S(1,1) (1)

	maximum = False
	counter = 0
	while not maximum:
		if fidelity[counter] > fidelity[counter + 1]:
			index_max = counter
			maximum = True
		else:
			counter += 1

	plt.plot(tf_vec * J / hbar, fidelity)
	for i in range(0, 5):
		plt.vlines((tf_vec[index_max] + T * i) * J / hbar, 0, 1, linestyle='--')
