import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest, hamiltonian_stochastic_detunning
from general_functions import (compute_eigensystem, compute_adiabatic_parameter, compute_parameters_interpolation, solve_system_unpack,
	stochastic_noise, sort_solution)
from scipy.constants import h, e
import concurrent.futures
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman splitting (ueV)
tau = 0.25  # Spin-conserving (ueV)
l2 = tau * 0.4  # Spin-flip tunneling (ueV)
l1 = l2 / 100  # Spin-flip tunneling (ueV)
u = 2000  # Intradot interaction (ueV)
Gamma = 1e-3  # Dephasing parameters (1/ns)

# Create the vector for eps
n_eps = 2 ** 15 + 1  # This number of elements is required for the romb method of integration used below
limit_eps = 3
eps_vector = np.linspace(-limit_eps, limit_eps, n_eps) * ET - u

parameters = [eps_vector, u, ET, tau, l1, l2]  # List with the parameters of the system

# Compute and plot the eigenenergies of the system
energies, states = compute_eigensystem(parameters, hamiltonian_2QD_1HH_Lowest)

partial_hamiltonian = np.zeros([n_eps, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = 1

# Compute the factors (\sum...) and the c_tilde parameters
factors, c_tilde = compute_adiabatic_parameter(eps_vector, states, energies, 1, hbar=hbar, partial_Hamiltonian=partial_hamiltonian)

# Solve the EDO to obtain the dependency of eps with the parameters
s, eps_sol = compute_parameters_interpolation(eps_vector, factors, c_tilde, method_1=False)

n_tf = 100
tf_vec = np.linspace(0.1, 100, n_tf)

n_t = 10 ** 3

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[0, 0] = 1  # Initially the only state populated is the triplet (in our basis the first state)

# Create the list of lists with all the parameters that will be used in the parallel computation
args = []  # Empty list in which save the sorter list of parameters
for i in range(0, n_tf):  # Iterate over all the final times
	time = np.linspace(0, tf_vec[i], n_t)  # Time vector in which compute the solution of the population
	temp = [i, time, density0, [eps_sol, u, ET, tau, l1, l2], hamiltonian_2QD_1HH_Lowest,
	        {'normalization': tf_vec[i], 'hbar': hbar, 'decoherence_fun': stochastic_noise,
	         'decoherence_param': [hamiltonian_stochastic_detunning, Gamma, [eps_sol]]}]  # List of parameters and default parameters as a dic
	args.append(temp)  # Append the list

workers = 8

if __name__ == '__main__':
	pbar = tqdm(total=n_tf, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')
	results_list = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
		results = executor.map(solve_system_unpack, args, chunksize=4)

		for result in results:
			results_list.append(result)
			pbar.update()
	pbar.refresh()

	results = sort_solution(results_list)

	density_matrix = []
	probabilities = []

	for temp in results:
		density_matrix.append(temp[0])
		probabilities.append(temp[1])

	fidelity = np.zeros(n_tf)
	for i in range(0, n_tf):
		fidelity[i] = probabilities[i][-1, 1]  # Extract the data from the final time (i), last time computed (-1), and the state S(1,1) (1)

	plt.plot(tf_vec, fidelity)
