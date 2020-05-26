"""
In this file is all the functions that create the Hamiltonians for all the system in which I'm interested.
"""
import numpy as np


def hamiltonian_3QD_1HH(e1, e2, e3, EZ, tN12, tN23, tF12, tF23):
	"""
	Creation of the Hamiltonian in a matrix form representing the system of 3 Quantum Dots populated with only 1 Heavy Hole. The parameters for the
	detuning of each dot is controlled individually, as well as the tunnellings. The magnetic field is common for the hole QD array. The basis used is
	[(↑,0,0), (↓,0,0), (0,↑,0), (0,↓,0), (0,0,↑), (0,0,↓)]
	:param e1: (float) Value for the gate voltage for the left (first) quantum dot
	:param e2: (float) Value for the gate voltage for the center (second) quantum dot
	:param e3: (float) Value for the gate voltage for the right (third) quantum dot
	:param EZ: (float) Value for the Zeeman splitting: EZ= mu * g * B
	:param tN12: (float) Value for the gate voltage for the spin conserving tunneling between the dots 1 and 2
	:param tN23: (float) Value for the gate voltage for the spin conserving tunneling between the dots 2 and 3
	:param tF12: (float) Value for the gate voltage for the spin flip tunneling between the dots 1 and 2
	:param tF23: (float) Value for the gate voltage for the spin flip tunneling between the dots 2 and 3
	:return: (numpy.array) Matrix representing the Hamiltonian for the system which the given parameters.
	"""

	matrix = np.zeros([6, 6], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	# Fill of the elements of the Hamiltonian, row by row
	matrix[0, :] = [e1 + EZ / 2, 0, -tN12, -1j * tF12, 0, 0]
	matrix[1, :] = [0, e1 - EZ / 2, -1j * tF12, -tN12, 0, 0]
	matrix[2, :] = [-tN12, 1j * tF12, e2 + EZ / 2, 0, -tN23, 1j * tF23]
	matrix[3, :] = [1j * tF12, -tN12, 0, e2 - EZ / 2, 1j * tF23, -tN23]
	matrix[4, :] = [0, 0, -tN23, -1j * tF23, e3 + EZ / 2, 0]
	matrix[5, :] = [0, 0, -1j * tF23, -tN23, 0, e3 - EZ / 2]

	return matrix


def hamiltonian_3QD_1e(e1, e2, e3, EZ, tN12, tN23):
	"""
	Creation of the Hamiltonian in a matrix form representing the system of 3 Quantum Dots populated with only 1 electron. The parameters for the
	detuning of each dot is controlled individually, as well as the tunnellings. The magnetic field is common for the hole QD array. The basis used is
	[(1,0,0), (0,1,0), (0,0,1)]
	:param e1: (float) Value for the gate voltage for the left (first) quantum dot
	:param e2: (float) Value for the gate voltage for the center (second) quantum dot
	:param e3: (float) Value for the gate voltage for the right (third) quantum dot
	:param EZ: (float) Value for the Zeeman splitting: EZ= mu * g * B
	:param tN12: (float) Value for the gate voltage for the spin conserving tunneling between the dots 1 and 2
	:param tN23: (float) Value for the gate voltage for the spin conserving tunneling between the dots 2 and 3
	:return: (numpy.array) Matrix representing the Hamiltonian for the system which the given parameters.
	"""

	matrix = np.zeros([3, 3], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	# Fill of the elements of the Hamiltonian, row by row
	matrix[0, :] = [e1 + EZ / 2, -tN12, 0]
	matrix[1, :] = [-tN12, e2 + EZ / 2, -tN23]
	matrix[2, :] = [0, -tN23, e3 + EZ / 2]

	return matrix


def hamiltonian_2QD_1HH_Lowest(epsilon, u, EZ, tau, l1, l2):
	"""
	Creation of the Hamiltonian for the three lowest energy states for the system of a double quantum dot array with 2 HH. The basis used is:
	[|T_-(1,1)>, |S(1,1)>, |S(0,2)>]
	:param epsilon: (float) Value for the detuning between the two quantum dots
	:param u: (float) Interdot interchange energy
	:param EZ: (float) Zeeman Splitting energy EZ= mu * g * B
	:param tau: (float) Spin-conserving flip
	:param l1:  (float) SOC between |T_-(1,1)> and |S(1,1)>
	:param l2: (float) SOC between |T_-(1,1)> and |S(0,2)>
	:return: (numpy.array) Matrix representing the Hamiltonian.
	"""
	matrix = np.zeros([3, 3], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	matrix[0, :] = [-EZ, l1, l2]
	matrix[1, :] = [l1, 0, np.sqrt(2) * tau]
	matrix[2, :] = [l2, np.sqrt(2) * tau, epsilon + u]
	return matrix


def hamiltonian_2QD_1HH_All(epsilon, u, EZ, tau, l1, l2):
	"""
	Creation of the Hamiltonian for all the states, except the double occupation singlet in the left quantum dot, in a system of a double quantum dot
	populated with 2 heavy holes. The basis used is: [(↑,↑), (↑,↓), (↓,↑), (↓,↓), (0,↑↓)]
	:param epsilon: (float) Value for the detuning between the two quantum dots
	:param u: (float) Interdot interchange energy
	:param EZ: (float) Zeeman Splitting energy EZ= mu * g * B
	:param tau: (float) Spin-conserving flip
	:param l1:  (float) SOC between |T_-(1,1)> and |S(1,1)>
	:param l2: (float) SOC between |T_-(1,1)> and |S(0,2)>
	:return: (numpy.array) Matrix representing the Hamiltonian
	:return: (numpy.array) Matrix representing the Hamiltonian.
	"""
	matrix = np.zeros([5, 5], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	matrix[0, :] = [EZ, 0, 0, 0, 0]
	matrix[1, :] = [0, 0, 0, 0, -tau]
	matrix[2, :] = [0, 0, 0, 0, tau]
	matrix[3, :] = [0, 0, 0, -EZ, 0]
	matrix[4, :] = [0, -tau, tau, 0, u + epsilon]
	return matrix


def hamiltonian_two_sites(delta, u, j):
	matrix = np.zeros([3, 3], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	matrix[0, :] = [u + delta, -np.sqrt(2) * j, 0]
	matrix[1, :] = [-np.sqrt(2) * j, 0, -np.sqrt(2) * j]
	matrix[2, :] = [0, -np.sqrt(2) * j, u - delta]

	return matrix


def hamiltonian_stochastic_detunning(epsilon):
	matrix = np.zeros([3, 3], dtype=complex)  # Create a matrix with the correct dimensions and complex elements

	matrix[2, 2] = epsilon
	return matrix
