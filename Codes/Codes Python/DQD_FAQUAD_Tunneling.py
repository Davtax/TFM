import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest
from general_functions import (solve_system_unpack, sort_solution, save_data, message_telegram, compute_adiabatic_parameter,
	compute_parameters_interpolation, compute_eigensystem)
import concurrent.futures
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar as Bar
from plotting_functions import save_figure
from scipy.interpolate import interp1d

hbar = 6.582 * 10 ** (-1)  # Hbar (ueV*ns)
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman spliting (ueV)
u = 2000  # Intradot interaction (ueV)

time_step = 1e-3

N = 2 ** 15 + 1
limit = 50
eps_vector = np.linspace(-limit, limit, N) * ET - u

n_tau = 10
n_tf = 10
tau_vec = np.linspace(0.1, 5, n_tau)
tf_vec = np.linspace(0.1, 100, n_tf)

l2_vector = tau_vec * 0.4
l1_vector = l2_vector / 100

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[0, 0] = 1  # Initially the only state populated is the triplet (in our basis the first state)

args = []
counter = 0
for i in range(n_tau):
	parameters = [eps_vector, u, ET, tau_vec[i], l1_vector[i], l2_vector[i]]

	energies, states = compute_eigensystem(parameters, hamiltonian_2QD_1HH_Lowest)

	factors, c_tilde = compute_adiabatic_parameter(eps_vector, states, energies, initial_state=1)

	s, eps_sol = compute_parameters_interpolation(eps_vector, factors, c_tilde)
	index_max = np.where((eps_sol + u) / ET > limit)[0][0]
	s_mod = np.linspace(0, 1, index_max + 1)
	eps_sol = interp1d(s_mod, eps_sol[:index_max + 1], kind='quadratic')

	for j in range(n_tf):
		time = np.arange(0, tf_vec[i], time_step)
		args.append([counter, time, density0, parameters, hamiltonian_2QD_1HH_Lowest])
		counter += 1
