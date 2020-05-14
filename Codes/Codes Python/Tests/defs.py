import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def model(y, t):
	return np.abs(y ** 2 + l ** 2) ** 1.5 * 2 * c1 / l


g = 1.35
muB = 9.27 * 10 ** (-24)
B = 0.015
ET = -g * muB * B
hbar = 1.055 * 10 ** (-34)
l = 0.1 * 1.6 * 10 ** (-19) * 10 ** (-6) / (hbar / 2)
tau = 5 * 1.6 * 10 ** (-19) * 10 ** (-6)
u = 3000 * 1.6 * 10 ** (-19) * 10 ** (-6)

Es = lambda eps: (u + eps - np.sqrt(8 * tau ** 2 + (u + eps) ** 2)) / 2
Delta = lambda eps: ((Es(eps) - ET) / 2) / (hbar / 2)

eps0 = -3050 * 1.6 * 10 ** (-19) * 10 ** (-6)
epstf = -2800 * 1.6 * 10 ** (-19) * 10 ** (-6)
N = 10000000

deltas = np.linspace(Delta(eps0), Delta(epstf), N, endpoint=True)
c1 = np.sum(1 / np.abs((l ** 2 + deltas ** 2) ** 1.5 / l)) * np.abs((deltas[0] - deltas[1])) / 2

n = 100
nt = 1000
DF = np.zeros([nt, 3])
Eps = np.zeros([nt])

s = np.linspace(0, 1, nt)


def resolucion_eps():
	delta_sol = odeint(model, Delta(eps0), s)[:, 0]
	# Deps = -(2 * tau ** 2 - (delta_sol * hbar + ET) ** 2) / (delta_sol * hbar + ET) - u
	Deps = ET - delta_sol * hbar - 2 * tau ** 2 / (ET - delta_sol * hbar) - u

	return delta_sol, Deps


def resolucion_dinamica_desempaquetar(paquete):
	tf, delta_sol = paquete
	return resolucion_dinamica(tf, delta_sol)


def resolucion_dinamica(tf, Delta_sol):
	t0 = np.linspace(0, tf, nt, endpoint=True)
	fDelta = interp1d(t0, Delta_sol, kind='cubic', fill_value="extrapolate")

	def rhs(t, h):
		# return [-1j * (h[0] * ET + l * h[1]) / 2, -1j * (h[1] * (u + feps(t) - np.sqrt(8 * tau ** 2 + (u + feps(t)) ** 2)) / 2 + l * h[0]) / 2]
		return [-1j * (h[0] * fDelta(t) + l * h[1]) / 2, -1j * (-h[1] * fDelta(t) / 2 + l * h[0]) / 2]

	res = solve_ivp(rhs, (0, tf), [1 + 0j, 0j], t_eval=t0)

	h1, h2 = np.abs(res.y) ** 2
	DF[:, 0] = t0
	DF[:, 1] = h1
	DF[:, 2] = h2

	return DF


def resolucion_density_matriz(tf, eps):
	t0 = np.linspace(0, tf, nt, endpoint=True)
	feps = interp1d(t0, eps, kind='cubic', fill_value="extrapolate")

	def Es(t):
		return (u + feps(t) - np.sqrt(8 * tau ** 2 + (u + feps(t)) ** 2)) / 2

	def rhs(t, h):
		# return [-1j * l * (h[2] - h[1]), -1j * (l * (h[3] - h[0]) + 2 * fDelta(t) * h[1]),
		#        -1j * (l * (h[0] - h[3]) - 2 * fDelta(t) * h[2]), -1j * l * (h[1] - h[2])]
		return [-1j * l * (h[2] - h[1]), -1j * (l * (h[3] - h[0]) + ET * h[1] - Es(t) * h[1]),
		        -1j * (l * (h[0] - h[3]) - ET * h[2] + Es(t) * h[2]), -1j * l * (h[1] - h[2])]

	res = solve_ivp(rhs, (0, tf), [1 + 0j, 0j, 0j, 0j], t_eval=t0)

	return res
