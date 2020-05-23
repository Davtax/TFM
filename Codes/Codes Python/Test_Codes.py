import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest
from general_functions import (compute_eigensystem, compute_adiabatic_parameter, compute_parameters_interpolation, solve_system_unpack,
	decoherence_test)
from scipy.constants import h, e

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman spliting (ueV)
tau = 0.25  # Sping-conserving (ueV)
l2 = tau * 0.4  # Spin-flip tunneling (ueV)
l1 = l2 / 100  # Spin-flip tunneling (ueV)
u = 2000  # Intradot interaction (ueV)
Gamma = 2e-3  # Dephasing parameters (1/ns)

# Create the vector for eps
n_eps = 2 ** 15 + 1  # This number of elements is requiered for the romb method of integration used below
limit_eps = 3
limit_eps_up = 202
limit_eps_low = -333
eps_vector = np.linspace(-limit_eps, limit_eps, n_eps) * ET - u

parameters = [eps_vector, u, ET, tau, l1, l2]  # List with the parameters of the system

labels = [r'$(\epsilon+u)/E_Z$', r'$E(\epsilon)/E_Z$']  # Labels of the figure

# Compute and plot the eigenenergies of the system
energies, states, fig, ax = compute_eigensystem(parameters, hamiltonian_2QD_1HH_Lowest, plot=True, x_vector=(eps_vector + u) / ET, normalization=ET,
                                                labels=labels)

partial_hamiltonian = np.zeros([n_eps, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = 1

# Compute the factors (\sum...) and the c_tilde parameters
factors, c_tilde = compute_adiabatic_parameter(eps_vector, states, energies, 1, hbar=hbar, partial_Hamiltonian=partial_hamiltonian)
print('c_tilde = {}'.format(c_tilde))

# Solve the EDO to obtain the dependency of eps with the paramer s
s, eps_sol = compute_parameters_interpolation(eps_vector, factors, c_tilde, method_1=False)

# Array for the values for the total times for the protocol that we will use
n_tf = 600
tf_vec = np.linspace(0.1, 100, n_tf)

n_t = 10 ** 3

density0 = np.zeros([3, 3], dtype=complex)  # Initialize the variable to save the density matrix
density0[0, 0] = 1  # Initially the only state populated is the triplet (in our basis the first state)

# Create the list of lists with all the parameters that will be used in the parallel computation
args = []  # Empty list in which save the sorter list of parameters
for i in range(0, n_tf):  # Iterate over all the final times
	time = np.linspace(0, tf_vec[i], n_t)  # Time vector in which compute the solution of the population
	temp = [i, time, density0, [eps_sol, u, ET, tau, l1, l2], hamiltonian_2QD_1HH_Lowest,
	        {'normalization': tf_vec[i], 'atol': 1e-8, 'rtol': 1e-6, 'decoherence_fun': decoherence_test,
	         'decoherence_param': [Gamma]}]  # List of parameters and default parameters as a dic
	args.append(temp)  # Append the list

result = solve_system_unpack(args[0])
