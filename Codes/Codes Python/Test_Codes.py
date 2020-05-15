import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest
import matplotlib.pyplot as plt
from general_functions import compute_limits

hbar = 6.582 * 10 ** (-1)  # Hbar (ueV*ns)
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman spliting (ueV)
u = 2000  # Intradot interaction (ueV)

n_eps = 100
limit = 200
eps_vector = np.linspace(-limit, limit, n_eps) * ET - u

n_tau = 100
min_tau = 0.1
max_tau = 5
tau_vec = np.linspace(min_tau, max_tau, n_tau)
l2_vector = tau_vec * 0.4
l1_vector = l2_vector / 100

parameters = [0, u, ET, 0, 0, 0]

limit1 = 0.99
limit2 = 0.99

state_1 = 0
state_2 = 1
adiabatic_state = 1

lim_T, lim_S = compute_limits(hamiltonian_2QD_1HH_Lowest, parameters, limit1, limit2, state_1, state_2, adiabatic_state, eps_vector,
                              [tau_vec, l1_vector, l2_vector], 0, [3, 4, 5], filter_bool=True)

plt.plot((lim_T + u) / ET, tau_vec)
plt.plot((lim_S + u) / ET, tau_vec)
