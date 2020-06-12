import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import hamiltonian_3QD_1HH
from general_functions import solve_system_unpack_qutip, sort_solution, save_data, generalized_Pauli_Matrices
from plotting_functions import save_figure
from scipy.constants import h, e
from scipy.misc import derivative
import qutip as qt
from tqdm import tqdm
import sys
from telegram_bot import message_telegram, image_telegram
import concurrent.futures
import time as timer

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9

n_tf = 157

tf_vector = np.linspace(0.001, 50, n_tf, endpoint=True)

# Parameters of the detunings for CTAP. THe definitions are shown in the
# Detunings for the dots

dx_derivative = 1e-3  # Step for the derivatives
alpha0 = 5  # Parameter that controls the amplitude of the pulses

# Auxiliar parameters
chi = lambda t, tf: np.pi * t / (2 * tf) - 1 / 3 * np.sin(2 * np.pi * t / tf) + 1 / 24 * np.sin(4 * np.pi * t / tf)
eta = lambda t, tf: np.arctan(derivative(chi, t, dx=dx_derivative, args=[tf]) / alpha0)

# Modified pulses
tau12_STA = lambda t, tf: (derivative(eta, t, dx=dx_derivative, args=[tf]) * np.cos(chi(t, tf)) + derivative(chi, t, dx=dx_derivative, args=[tf]) / (
		np.tan(eta(t, tf)) + 1e-16) * np.sin(chi(t, tf))) / np.sqrt(2) * hbar
tau23_STA = lambda t, tf: (derivative(chi, t, dx=dx_derivative, args=[tf]) * np.cos(chi(t, tf)) / (np.tan(eta(t, tf)) + 1e-16) - derivative(eta, t,
                                                                                                                                            dx=dx_derivative,
                                                                                                                                            args=[
	                                                                                                                                            tf]) * np.sin(
	chi(t, tf))) / np.sqrt(2) * hbar

phase = np.pi / 2

prop = 1

pulse1 = tau12_STA
pulse2 = tau23_STA

max_pulse = np.zeros(n_tf)

for i in range(n_tf):
	time = np.linspace(0, tf_vector[i], 10 ** 4, endpoint=True)  # Time vector in which compute the solution of the population

	pulse1_vec = pulse1(time, time[-1])
	pulse2_vec = pulse2(time, time[-1])

	max_temp = np.max((pulse1_vec, pulse2_vec))

	max_pulse[i] = max_temp

plt.figure()
plt.plot(tf_vector, max_pulse)
