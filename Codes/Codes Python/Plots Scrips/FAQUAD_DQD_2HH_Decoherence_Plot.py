import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import modify_plot, save_figure

styles = ['science']
prefix = './stylelib/'
sufix = '.mplstyle'

for i in range(len(styles)):
	styles[i] = prefix + styles[i] + sufix

plt.style.use(styles)

# Path the to file that we want to load
folder = 'data/'
file = 'FAQUAD_DQD_2HH_Decoherence'
extension = '.npy'
file_dic = folder + file + extension

data = np.load(file_dic, allow_pickle=True)  # Load the data, allow_pickle enable to load a list

results = data[0]
Gamma_vec = data[1]
tf_vec = data[2]

n_tf = len(tf_vec)
n_Gamma = len(Gamma_vec)

population_middle = np.zeros([n_tf, n_Gamma])
fidelity = np.zeros([n_tf, n_Gamma])
for i in range(0, n_tf):
	for j in range(0, n_Gamma):
		index = i * n_Gamma + j
		temp = results[index]
		population_middle[i, j] = np.max(temp[:, 2])
		fidelity[i, j] = temp[-1, 1]

save = True

ticks = 20
labels = 22
text = 22

fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=[14, 5])
fig.subplots_adjust(wspace=0.5)

pos1 = ax1.imshow(fidelity.transpose(), origin='lower', cmap='jet', aspect='auto',
                  extent=[tf_vec[0], tf_vec[-1], np.abs(Gamma_vec[0]), np.abs(Gamma_vec[-1])], interpolation='quadric', vmin=0, vmax=1)
cbar1 = fig.colorbar(pos1, ax=ax1)
cbar1.set_label(r'$\mathcal{F}$', fontsize=labels, labelpad=10)
cbar1.ax.tick_params(labelsize=ticks)

ax1.set_xlabel(r'$t_f\; [ns]$')
ax1.set_ylabel(r'$\Gamma\; [ns^{-1}]$', labelpad=10)

pos2 = ax2.imshow(population_middle.transpose(), origin='lower', cmap='jet', aspect='auto',
                  extent=[tf_vec[0], tf_vec[-1], np.abs(Gamma_vec[0]), np.abs(Gamma_vec[-1])], interpolation='quadric')
cbar2 = fig.colorbar(pos2, ax=ax2)
cbar2.set_label(r'$\max(P2(t))$', fontsize=labels, labelpad=10)
cbar2.ax.tick_params(labelsize=ticks)

ax2.set_xlabel(r'$t_f\; [ns]$')
ax2.set_ylabel(r'$\Gamma\; [ns^{-1}]$', labelpad=10);

if save:
	ax1.text(-4.5, 0.11, 'a)', {'fontsize': text})
	ax2.text(-4.5, 0.11, 'b)', {'fontsize': text})

	modify_plot(ax1, tick_direction='inout', x_ticks_vector=np.arange(0, 25, 5), label_size=labels, tick_label_size=ticks)
	ax1.tick_params(axis='x')

	modify_plot(ax2, tick_direction='inout', x_ticks_vector=np.arange(0, 25, 5), label_size=labels, tick_label_size=ticks)

	plt.tight_layout()

	save_figure(fig, 'FAQUAD_DQD_Decoherence', overwrite=save, extension='pdf')