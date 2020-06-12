import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import hamiltonian_3QD_1HH
from general_functions import solve_system_unpack_qutip, sort_solution, save_data
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
ET = 0  # Zeeman splitting (ueV)
tau0 = 1.3  # Value for the tunneling parameter
tf = 10  # Final time

e2 = 0
e3 = 0

# Parameters of the detunings for CTAP. THe definitions are shown in the
# Detunings for the dots
e1 = 0
tau = tf / 6  # Be careful, this is the parameter for the Gaussian shape tunnelling, not the tunnelling itself
sigma = tau

time = np.linspace(0, tf, 10 ** 3, endpoint=True)  # Time vector in which compute the solution of the population

dx_derivative = 1e-3  # Step for the derivatives
alpha0 = 5  # Parameter that controls the amplitude of the pulses

# Auxiliar parameters
chi = lambda t: np.pi * t / (2 * tf) - 1 / 3 * np.sin(2 * np.pi * t / tf) + 1 / 24 * np.sin(4 * np.pi * t / tf)
eta = lambda t: np.arctan(derivative(chi, t, dx=dx_derivative) / alpha0)

# Modified pulses
tau12_tilde_STA = lambda t: (derivative(eta, t, dx=dx_derivative) * np.cos(chi(t)) + derivative(chi, t, dx=dx_derivative) / (
		np.tan(eta(t)) + 1e-16) * np.sin(chi(t))) / np.sqrt(2) * hbar
tau23_tilde_STA = lambda t: (derivative(chi, t, dx=dx_derivative) * np.cos(chi(t)) / (np.tan(eta(t)) + 1e-16) - derivative(eta, t,
                                                                                                                           dx=dx_derivative) * np.sin(
	chi(t))) / np.sqrt(2) * hbar

workers = 8

n_phase = 256
n_prop = 256
n_total = n_phase * n_prop

chunksize = 500

phase = np.pi / 2
delta_phase = np.linspace(-1, 1, n_phase, endpoint=True) * np.pi/2

prop = 1
delta_prop = np.linspace(-1, 1, n_prop, endpoint=True) * 0.001

Delta_phase, Delta_prop = np.meshgrid(delta_phase, delta_prop)
Delta_phase = Delta_phase.flatten()
Delta_prop = Delta_prop.flatten()

pulse1 = tau12_tilde_STA
pulse2 = tau23_tilde_STA

tau12_interpolated = qt.Cubic_Spline(time[0], time[-1], pulse1(time))
tau23_interpolated = qt.Cubic_Spline(time[0], time[-1], pulse2(time))

psi0 = qt.basis(6, 0)

if __name__ == '__main__':
	results_list = []
	pbar1 = tqdm(total=n_total, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')

	start = timer.perf_counter()
	for i in range(0, np.int(np.ceil(n_total / chunksize))):
		initial = i * chunksize
		final = initial + chunksize
		if final >= n_total:
			final = n_total

		args = []  # Empty list in which save the sorter list of parameters
		for j in range(initial, final):  # Iterate over all the final times

			H0 = hamiltonian_3QD_1HH(*[e1, e2, e3, ET, 0, 0, 0, 0, 0]) / hbar
			H0 = qt.Qobj(H0)
			H1 = hamiltonian_3QD_1HH(*[0, 0, 0, 0, 1, 0, prop + Delta_prop[j], 0, phase + Delta_phase[j]]) / hbar
			H1 = qt.Qobj(H1)
			H2 = hamiltonian_3QD_1HH(*[0, 0, 0, 0, 0, 1, 0, prop + Delta_prop[j], phase + Delta_phase[j]]) / hbar
			H2 = qt.Qobj(H2)

			H = [H0, [H1, tau12_interpolated], [H2, tau23_interpolated]]

			temp = [j, H, psi0, time, {'dim': 6, 'only_final': True}]  # List of parameters and default parameters as a dic
			args.append(temp)  # Append the list

		with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
			results = executor.map(solve_system_unpack_qutip, args, chunksize=4)
			for result in results:
				results_list.append(result)
				pbar1.update()

	pbar1.refresh()
	finish = timer.perf_counter()

	results = sort_solution(results_list)

	probabilities = []
	for temp in results:
		probabilities.append(np.abs(temp) ** 2)

	fidelity = np.zeros([n_prop, n_phase])
	for i in range(n_prop):
		for j in range(n_phase):
			index = i * n_phase + j
			temp = probabilities[index]
			fidelity[i, j] = temp[5]

	fig, ax = plt.subplots()
	data = np.log10(1 - fidelity).transpose()
	pos = ax.imshow(data, origin='lower', cmap='jet_r', aspect='auto', extent=[delta_prop[0], delta_prop[-1], delta_phase[0], delta_phase[-1]],
	                interpolation='spline16', vmin=-10)
	cbar = fig.colorbar(pos, ax=ax, extend='min')
	cbar.set_label(r'$\log_{10}(1-\mathcal{F})$')
	ax.set_xlabel(r'$\Delta x$')
	ax.set_ylabel(r'$\Delta \theta$')

	# contours = ax.contour(delta_prop, delta_phase, data, np.arange(-11, 0, 1), colors='black', linestyles='solid')
	# ax.clabel(contours, inline=False)

	total_time = finish - start

	if total_time > 60 * 60:
		units = '(hr)'
		total_time /= (60 * 60)
	elif total_time > 60:
		units = '(min)'
		total_time /= 60
	else:
		units = ' (s)'

	file_name = 'STA_TQD_Deviation_SF'
	message_telegram('DONETE: {} {}x{}. Total time: {:.2f} {}'.format(file_name, n_prop, n_phase, total_time, units))
	save_figure(fig, file_name, overwrite=True, extension='png', dic='data/')
	image_telegram('data/' + file_name + '.png')
	save_data(file_name, [results, delta_prop, delta_phase, ['results', 'delta_prop', 'delta_phase']], overwrite=True)
