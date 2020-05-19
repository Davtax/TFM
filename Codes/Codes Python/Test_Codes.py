import numpy as np
from hamiltonians import hamiltonian_2QD_1HH_Lowest
from general_functions import (compute_eigensystem, compute_adiabatic_parameter, compute_parameters_interpolation, compute_period,
	solve_system_unpack, sort_solution)
from plotting_functions import modify_plot, save_figure
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm.notebook import tqdm
from scipy.constants import h, e
from scipy.interpolate import interp1d
from scipy.integrate import romb, odeint

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]
g = 1.35  # g-factor fo the GaAs
muB = 57.883  # Bohr magneton (ueV/T)
B = 0.015  # Magnetic field applied (T)
ET = g * muB * B  # Zeeman spliting (ueV)
tau = 0.25  # Sping-conserving (ueV)
l2 = tau * 0.4  # Spin-flip tunneling (ueV)
l1 = l2 / 100  # Spin-flip tunneling (ueV)
u = 2000  # Intradot interaction (ueV)

n_eps = 2 ** 15 + 1  # This number of elements is requiered for the romb method of integration used below
limit_eps = 4
eps_vector = np.linspace(-limit_eps, limit_eps, n_eps) * ET - u

parameters = [eps_vector, u, ET, tau, l1, l2]  # List with the parameters of the system

labels = [r'$(\epsilon+u)/E_Z$', r'$E(\epsilon)/E_Z$']  # Labels of the figure

# Compute and plot the eigenenergies of the system

energies, states = compute_eigensystem(parameters, hamiltonian_2QD_1HH_Lowest, plot=False)

# Compute the factors (\sum...) and the c_tilde parameters
partial_hamiltonian = np.zeros([n_eps, 3, 3], dtype=complex)
partial_hamiltonian[:, 2, 2] = 1

# Compute the factors (\sum...) and the c_tilde parameters
factors1, c_tilde1 = compute_adiabatic_parameter(eps_vector, states, energies, 1, hbar=hbar)

factors2, c_tilde2 = compute_adiabatic_parameter(eps_vector, states, energies, 1, hbar=hbar, partial_Hamiltonian=partial_hamiltonian)

x_vec = eps_vector

nt = len(x_vec)  # The number of elements is the total number of x_vec


def factor_interpolation1(x):  # Interpolation for the odeint method
	return interp1d(x_vec, 1 / np.sum(factors1, axis=1), kind='quadratic', fill_value="extrapolate")(x)


def model1(y, _):  # EDO to be solved
	return c_tilde1 / hbar * factor_interpolation1(y)


def factor_interpolation2(x):  # Interpolation for the odeint method
	return interp1d(x_vec, 1 / np.sum(factors2, axis=1), kind='quadratic', fill_value="extrapolate")(x)


def model2(y, _):  # EDO to be solved
	return c_tilde2 / hbar * factor_interpolation2(y)


# Rescaled time parameter s=t/tF, the end point is a bit larger than 1 since there are numerical errors, and the desired fina
# reached exactly at s=1
s = np.linspace(0, 1.01, nt, endpoint=True)

x_sol = odeint(model2, x_vec[0], s)[:, 0]

# s, eps_sol = compute_parameters_interpolation(eps_vector, factors2, c_tilde2)
