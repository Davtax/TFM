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

scheme = 'CTAP'

sleep = False

hbar = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9
ET = 0  # Zeeman splitting (ueV)

n_tf = 64
n_dephasing = 64
n_total = n_tf * n_dephasing

tf_vector = np.linspace(0.001, 50, n_tf, endpoint=True)
gamma_vector = np.linspace(0, 2, n_dephasing, endpoint=True)

Gamma_vector, Tf_vector = np.meshgrid(gamma_vector, tf_vector)
Gamma_vector = Gamma_vector.flatten()
Tf_vector = Tf_vector.flatten()

workers = 8

chunksize = 2 ** 9

e1 = 0
e2 = 0
e3 = 0

# Parameters of the detunings for CTAP. THe definitions are shown in the
# Detunings for the dots

dx_derivative = 1e-3  # Step for the derivatives
alpha0 = 5  # Parameter that controls the amplitude of the pulses

tau0 = 2.3
tau12 = lambda t, tf: tau0 * np.exp(-(t - tf / 2 - tf / 6) ** 2 / (tf / 6) ** 2)
tau23 = lambda t, tf: tau0 * np.exp(-(t - tf / 2 + tf / 6) ** 2 / (tf / 6) ** 2)

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

if scheme == 'CTAP':
	pulse1 = tau12
	pulse2 = tau23
elif scheme == 'STA':
	pulse1 = tau12_STA
	pulse2 = tau23_STA

psi0 = qt.basis(6, 0)

matrices = generalized_Pauli_Matrices(6)

c_ops = []
for i in range(0, len(matrices)):
	c_ops.append(qt.Qobj(matrices[i]))

options = qt.Options(atol=1e-15, rtol=1e-13)

H0 = hamiltonian_3QD_1HH(*[e1, e2, e3, ET, 0, 0, 0, 0, 0]) / hbar
H0 = qt.Qobj(H0)
H1 = hamiltonian_3QD_1HH(*[0, 0, 0, 0, 1, 0, prop, 0, phase]) / hbar
H1 = qt.Qobj(H1)
H2 = hamiltonian_3QD_1HH(*[0, 0, 0, 0, 0, 1, 0, prop, phase]) / hbar
H2 = qt.Qobj(H2)

if __name__ == '__main__':
	if sleep:
		print('Sleeping (zzzzzz.....)')
		timer.sleep(60 * 60)
		print('Let\'s start to work')

	results_list = []
	pbar = tqdm(total=n_total, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')

	start = timer.perf_counter()
	for i in range(0, np.int(np.ceil(n_total / chunksize))):
		initial = i * chunksize
		final = initial + chunksize
		if final >= n_total:
			final = n_total

		args = []  # Empty list in which save the sorter list of parameters
		for j in range(initial, final):  # Iterate over all the final times

			time = np.linspace(0, Tf_vector[j], 10 ** 4, endpoint=True)  # Time vector in which compute the solution of the population

			tau12_interpolated = qt.Cubic_Spline(time[0], time[-1], pulse1(time, time[-1]))
			tau23_interpolated = qt.Cubic_Spline(time[0], time[-1], pulse2(time, time[-1]))

			H = [H0, [H1, tau12_interpolated], [H2, tau23_interpolated]]

			c_ops_factor = []
			for k in range(len(c_ops)):
				c_ops_factor.append(c_ops[k] * Gamma_vector[j] / 2)

			temp = [j, H, psi0, time,
			        {'dim': 6, 'only_final': True, 'c_ops': c_ops_factor, 'options': options}]  # List of parameters and default parameters as a dic
			args.append(temp)  # Append the list

		with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
			results = executor.map(solve_system_unpack_qutip, args, chunksize=4)
			for result in results:
				results_list.append(result)
				pbar.update()

	pbar.refresh()
	pbar.close()
	finish = timer.perf_counter()

	results = sort_solution(results_list)

	probabilities = []
	for temp in results:
		probabilities.append(np.abs(temp) ** 2)

	fidelity = np.zeros([n_tf, n_dephasing])
	for i in range(n_tf):
		for j in range(n_dephasing):
			index = i * n_dephasing + j
			temp = probabilities[index]
			fidelity[i, j] = temp[5]

	fig, ax = plt.subplots()
	data = fidelity.transpose()
	pos = ax.imshow(data, origin='lower', cmap='jet', aspect='auto', extent=[tf_vector[0], tf_vector[-1], gamma_vector[0], gamma_vector[-1]],
	                interpolation='none')
	cbar = fig.colorbar(pos, ax=ax)
	cbar.set_label(r'$\mathcal{F}$')
	ax.set_xlabel(r'$t_F\; [ns]$')
	ax.set_ylabel(r'$\gamma\; $')

	total_time = finish - start

	if total_time > 60 * 60:
		units = '(hr)'
		total_time /= (60 * 60)
	elif total_time > 60:
		units = '(min)'
		total_time /= 60
	else:
		units = ' (s)'

	file_name = scheme + '_TQD_Dephasing_SF'
	message_telegram('DONETE: {} {}x{}. Total time: {:.2f} {}'.format(file_name, n_tf, n_dephasing, total_time, units))
	save_figure(fig, file_name, overwrite=True, extension='png', dic='data/')
	image_telegram('data/' + file_name + '.png')
	save_data(file_name, [results, tf_vector, gamma_vector, ['results', 'tf_vector', 'gamma_vector']], overwrite=True)
