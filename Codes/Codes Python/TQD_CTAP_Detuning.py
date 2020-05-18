import numpy as np
from hamiltonians import hamiltonian_3QD_1HH
from general_functions import solve_system_unpack, sort_solution, save_data
from telegram_bot import message_telegram, image_telegram
from scipy.constants import h, e
from scipy.misc import derivative
import concurrent.futures
import time as timer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from plotting_functions import save_figure

h_eV = h / e  # Value for the Plank's constant [eV * s]
ET = (100 * np.pi)  # Zeeman splitting (ueV)
l1 = 0  # Spin-flip tunneling (ueV)
l2 = 0  # Spin-flip tunneling (ueV)
Omega0 = (100 * np.pi)  # * h_eV * 10 ** 6 * 10 ** 6  # Value for the tunelling parameter [mueV]
tf = 20 * (2 * np.pi) / Omega0  # ((100 * np.pi) * 10 ** 6) * 10 ** 9  # Final time [ns]

# Parameters of the detunnings
tau = tf / 6
sigma = tau

Omega12 = lambda t: Omega0 * np.exp(-(t - tf / 2 - tau) ** 2 / sigma ** 2)
Omega23 = lambda t: Omega0 * np.exp(-(t - tf / 2 + tau) ** 2 / sigma ** 2)

theta = lambda t: np.arctan(Omega12(t) / Omega23(t))
Omegaa = lambda t: derivative(theta, t, dx=1e-5)
phi = lambda t: np.arctan(Omegaa(t) / Omega12(t))
Omega12_tilda = lambda t: np.sqrt(Omega12(t) ** 2 + Omegaa(t) ** 2)
Omega23_tilda = lambda t: Omega23(t) - derivative(phi, t, dx=1e-5)

dx_derivative = 1e-3
alpha0 = 400
chi = lambda t: np.pi * t / (2 * tf) - 1 / 3 * np.sin(2 * np.pi * t / tf) + 1 / 24 * np.sin(4 * np.pi * t / tf)
eta = lambda t: np.arctan(derivative(chi, t, dx=dx_derivative) / alpha0)


def Omega12_tilde_SA(t):
	return derivative(eta, t, dx=dx_derivative) * np.cos(chi(t)) + derivative(chi, t, dx=dx_derivative) / (np.tan(eta(t)) + 1e-16) * np.sin(chi(t))


def Omega23_tilde_SA(t):
	return derivative(chi, t, dx=dx_derivative) * np.cos(chi(t)) / (np.tan(eta(t)) + 1e-16) - derivative(eta, t, dx=dx_derivative) * np.sin(chi(t))


def Omega12f_tilde_SA(t):
	return Omega12_tilde_SA(t) * 0.4


def Omega23f_tilde_SA(t):
	return Omega23_tilde_SA(t) * 0.4


factor_mueV = h_eV * 10 ** 6 * 10 ** 6
factor_ns = (2 * np.pi) / ((100 * np.pi) * 10 ** 6) * 10 ** 9

time_step = 1e-3
time = np.arange(0, tf + time_step, time_step)  # Time vector in which compute the solution of the population

Omega_max = np.max([Omega12_tilde_SA(time), Omega23_tilde_SA(time)])

detunning13 = np.linspace(-1, 1, 100) * Omega_max
detunning2 = np.linspace(-1, 1, 100) * Omega_max

density0 = np.zeros([6, 6], dtype=complex)  # Variable in which save the initial density matrix
density0[0, 0] = 1  # The system initialize in the state |↑,0,0>

args = []
i = 0
for e13 in detunning13:
	for e2 in detunning2:
		parameters = [-e13, e2, e13, 0, Omega12_tilde_SA, Omega23_tilde_SA, Omega12f_tilde_SA, Omega23f_tilde_SA]
		args.append([i, time, density0, parameters, hamiltonian_3QD_1HH])
		args.append({'hbar': 1})
		i += 1

result_list = []

if __name__ == '__main__':
	pbar = tqdm(total=len(detunning13) * len(detunning2), desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')

	start = timer.perf_counter()

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = executor.map(solve_system_unpack, args)

		for result in results:
			result_list.append(result)
			pbar.update()

	finish = timer.perf_counter()

	result = sort_solution(result_list)

	fidelity = np.zeros([len(detunning13), len(detunning2)])

	for i in range(0, len(detunning13)):
		for j in range(0, len(detunning2)):
			index = i * len(detunning2) + j
			temp = result[index]
			fidelity[i, j] = temp[-1, 4]
	fig, ax = plt.subplots()
	pos = ax.imshow(fidelity.transpose(), cmap='jet', aspect='auto',
	                extent=[detunning13[0] / Omega_max, detunning13[-1] / Omega_max, detunning2[0] / Omega_max, detunning2[-1] / Omega_max])
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

	message_telegram('DONETE: {}x{}. Total time: {:.2f} {}'.format(len(detunning13), len(detunning2), total_time, units))

	file_name = 'CTAP_TQD_1HH_Test'
	save_figure(fig, file_name, overwrite=True, extension='png', dic='data/')
	image_telegram('data/' + file_name + '.png')
	save_data(file_name, [fidelity, detunning13, detunning2, Omega_max, ['fidelity', 'detunning13', 'detunning2', 'Omega_max']])
