"""
In this file I will write all the functions that I commonly use in the Master's thesis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import romb, odeint
from scipy.interpolate import interp1d
from scipy.constants import h, e
import os
import telepot


hbar_muev_ns = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]


def question_overwrite(name):
	"""
	Make a question if you want to overwrite a certain file that already exist in the directory. There is only two possible answers y -> yes (true) or
	n -> no (False). If the answer is non of this two the question is repeated until a good answer is given.
	:param name: (Str) Name of the file to overwrite
	:return: (Bool) Answer given by the user
	"""

	temp = input('Do you want to overwrite the file ({})?  [y]/n'.format(name))  # Ask for and answer by keyword input

	if temp == 'y' or temp == '':
		return True
	elif temp == 'n':
		return False
	else:  # If the answer is not correct
		print('I didnt understand your answer.')
		return question_overwrite(name)  # The function will repeat until a correct answer if provided


def check_iterable_parameters(parameters):
	"""
	Function to check which parameter has more than one value, and the dimension of this one
	:param parameters: (list) List with the parameters
	:return: (numpy.array) Array with the number of elements > 1 each parameter has
	"""
	number_elements = np.zeros(len(parameters))  # Variable to save the number of elements in each parameter

	for i in range(0, len(parameters)):  # Iterate over all parameters
		number_elements[i] = np.size(parameters[i])  # Obtain the number of element in the parameter

	return number_elements.astype(int)


def create_hypermatrix(parameters, hamiltonian, join=None):
	"""
	Sometime I want a hypermatrix in which each "slice" represent a certain Hamiltonian with a given parameters, this is the function that make it.
	:param parameters: (list) List with all the parameters (they must be in the correct order for the given Hamiltonian.
	:param hamiltonian: (function) function that compute the hamiltonian that we what
	:param join: (list) List with the indices indicating the parameters that run equally. FUNCTION IS PROGRESS.
	:return: (numpy.matrix) Matrix with dimensions (n x D x D) where n is the number of time we want to compute the Hamiltonian, and D is the
	dimension of the Hamiltonian.
	"""

	n = check_iterable_parameters(parameters)  # Extract the number of elements
	n_wo_ones = n[np.where(n != 1)]

	# Compute all the possible combinations for the parameters, the indexing is important in order to obtain a correct result when reshaping the matrix
	if join is None:
		temp = np.meshgrid(*parameters, indexing='ij')
	else:
		parameters_temp = []
		factor_join = []
		for i in range(0, len(parameters)):
			if i not in join or i == join[0]:
				parameters_temp.append(parameters[i])  # TODO Change all this mess with different parameters joined
				factor_join.append(1)
		temp = np.meshgrid(*parameters_temp, indexing='ij')

	if join is not None:  # If various parameters goes in a group
		for i in range(1, len(parameters)):  # Iterate over the repeated indices
			n[join[i]] = 1  # Put the multiplicity to one
	total = np.product(n)  # Obtain the total number of combinations of parameters we have to compute

	runnings = np.zeros(len(parameters)).tolist()  # Compute a test parameter to extract the dimension of the Hamiltonian
	dim = np.shape(hamiltonian(*runnings))[0]

	hypermatrix = np.zeros(np.append(n_wo_ones, (dim, dim)), dtype=complex).flatten()  # Hypermatrix to save the different hamiltonians

	for i in range(0, total):  # Iterate over all the possible combinations
		runnings = []  # List in which the combinations will be saved
		for j in range(0, len(parameters)):  # Iterate over all the parameters
			runnings.append(temp[j].flatten()[i])  # Extract the value for the parameter j in the combination i
		hypermatrix[dim ** 2 * i:dim ** 2 * (i + 1)] = hamiltonian(*runnings).flatten()  # Compute the Hamiltonian for the combination i

	hypermatrix = hypermatrix.reshape(np.append(n_wo_ones, (dim, dim)))  # Reshape the hypermatrix

	return hypermatrix


def check_callable(parameters):
	"""
	Function to check in a list of parameters which of them are interpolation functions or lambda function and which are just numbers
	:param parameters: (list) List of parameters, some of them are interpolation functions and others are numbers
	:return: (list) List with booleans representing if the corresponding parameters is an interpolation function or not
	"""
	temp = []

	for i in parameters:  # Iterate over all the parameters
		# Check if is an interpolation function, or a lambda function
		if callable(i):
			temp.append(True)
		else:
			temp.append(False)

	return temp


def extract_interpolation(x, parameters, which, normalization=1):
	"""
	Function to extract the values of the interpolated and not interpolated parameters
	:param x: (float) Value representing the independent variable of the interpolation
	:param parameters: (list) List of parameters, some of them are interpolation functions and others are numbers
	:param which: (list) List of booleans representing which of the parameters are interpolation functions
	:param normalization: (float) Normalization, if needed, for the interpolation
	:return: (list) List with the values of the parameters at the given value of x
	"""
	temp = []

	for i in range(0, len(parameters)):  # Iterate over all the parameters
		if which[i]:  # If the parameters is an interpolation function
			# Compute the value at the given coordinate x. The  * 1 factor is just a trick to obtain a float and not a ndarray: ()
			temp.append(parameters[i](x / normalization) * 1)
		else:
			temp.append(parameters[i])

	return temp


def density_matrix_equation(t, y, dim, parameters, hamiltonian, which, normalization, hbar):
	"""
	Function to give the numerical value for the EDO that represent the evolution of the density matrix at a given time
	:param t: (float) Time at which we want to evaluate the EDO.
	:param y:  (numpy.array) Array with the flattened density matrix
	:param dim: (int) Dimension of the density matrix
	:param parameters: (list) List of parameters to evaluate, some of them are interpolation functions
	:param hamiltonian: (function) Function that compute the corresponding Hamiltonian in which we are interested
	:param which: (list) List of booleans representing which of the parameters are interpolation functions
	:param normalization: (float) Normalization (if needed) for the interpolation
	:param hbar: (float) Value for the h bar constant
	:return: (numpy.array) Solution of the EDO after flattening it
	"""
	rho = np.reshape(y, [dim, dim])  # Un-flatten the density matrix

	decoherence = 0  # For the moment I will not include decoherence

	# Extract the values for the interpolation parameters
	parameters_interpolated = extract_interpolation(t, parameters, which, normalization=normalization)

	ham_matrix = hamiltonian(*parameters_interpolated)  # Compute the hamiltonian with the interpolated parameters

	drhodt = -1j * (np.matmul(ham_matrix, rho) - np.matmul(rho, ham_matrix)) / hbar + decoherence  # Compute the EDO
	return drhodt.flatten()  # Flatten the solution to make it an array


def solve_system(time, density0, parameters, hamiltonian, full=False, prob=False, hbar=hbar_muev_ns, normalization=1, method='RK45', t_eval=False,
                 atol=1e-6, rtol=1e-3):
	"""
	Function to numerically solve the time evolution of a given density matrix.
	:param time: (numpy.array) Array with the times at which we want to compute the evolution
	:param density0: (numpy.matrix) Matrix with the density matrix at the initial time
	:param parameters: (list) List of parameters for the Hamiltonian. Some can be interpolation functions
	:param hamiltonian: (function) Function that compute the Hamiltonian we want
	:param full: (Bool) If the user want the full result of the EDO solver
	:param prob: (Bool) If the user want the probabilities of the states for the different times
	:param hbar: (float) Value for the h bar constant
	:param normalization: (float) Normalization for the argument of the interpolated functions
	:param method: (str) Method to solve the system
	:param t_eval: (Bool) If the user want solve_ivp automatically choose the time at which solve the EDO
	:param atol: (float) Absolute maximum error allowed
	:param rtol: (float) Relative maximum error allowed
	:return: Solution given by the function scipy.integrate.solve_ivp. If we want to extract the solution for the density matrix at each value of t
	we must make sol.y
	"""
	dim = np.shape(density0)[0]  # Compute the dimension of the density matrix
	which = check_callable(parameters)  # Check and save which of the parameters are callable functions

	if t_eval:
		t_eval_array = None
	else:
		t_eval_array = time

	# Solve the system at the given time, with the correct initial conditions
	sol = solve_ivp(density_matrix_equation, (time[0], time[-1]), density0.flatten(), t_eval=t_eval_array,
	                args=[dim, parameters, hamiltonian, which, normalization, hbar], method=method, atol=atol, rtol=rtol)

	if not full:  # If the user only want the result of the EDO
		time = sol.t
		sol = sol.y.reshape([dim, dim, len(time)])
		if prob:
			sol = [sol, np.abs(np.diagonal(sol)), time]

	return sol


def solve_system_unpack(pack):
	return [pack[0], solve_system(*pack[1:], prob=True, hbar=1)[1]]


def sort_solution(data):
	n = len(data)

	sorted_sol = [None] * n

	for i in range(n):
		index = data[i][0]
		temp = data[i][1]
		sorted_sol[index] = temp

	return sorted_sol


def compute_eigensystem(parameters, hamiltonian, hypermatrix=None, plot=False, x_vector=None, title=None, colors=None, legend=None, normalization=1,
                        labels=None, ax=None):
	"""
	Function to compute the eigenenergies and eigenstates of a given hypermatrix. If the user wants the eigenenergies are plotted
	:param parameters: (list) List of parameters for the hamiltonian
	:param hamiltonian: (build-in function) Function pointing at the hamiltonian in which we are interested
	:param hypermatrix: (numpy.matrix) The hypermatrix can be load directly
	:param plot: (Bool) If the user want to plot the eigenenergies
	:param x_vector: (numpy.array) Array with the independent parameter
	:param title: (str) Name for the title of the figure
	:param colors: (list) List with the colours for the different lines
	:param legend: (list) List with the names of each line for the legend
	:param normalization: (float) Value for the normalization in the Y axis
	:param labels: (list) List the with labels of the X and Y axis respectively
	:param ax: (matplotlib.axes) Axes in which plot the result
	:return: The result gives always the energies and the states. If the function plot the result then we add to the return the figure (if needed) and
	the axis.
	"""
	if hypermatrix is None:
		hypermatrix = create_hypermatrix(parameters, hamiltonian)
	eigensystem = np.linalg.eigh(hypermatrix)  # Compute the eigensystem of the given matrix
	energies = eigensystem[0]  # Extract the energies
	states = eigensystem[1]  # Extract the states

	dim = np.shape(energies)[1]  # Obtain the dimension of the original Hamiltonian

	ret = [energies, states]  # Save the variables to be returned

	if plot:  # If the user want to plot the result

		if legend is None:  # If a legend is not provided
			plot_legend = False
			legend = str(np.zeros(dim))  # Save what ever crap, it will not be plotted
		else:
			plot_legend = True

		if ax is None:  # If an axis is not provided
			fig = plt.figure()
			ax = fig.add_subplot()
			ret.append(fig)

		for i in range(0, dim):  # Iterate over all the energies
			if colors is not None:  # If colours are provided
				ax.plot(x_vector, energies[:, i] / normalization, label=legend[i], c=colors[i])
			else:
				ax.plot(x_vector, energies[:, i] / normalization, label=legend[i])

		if plot_legend:  # If a legend is provided
			ax.legend()

		ax.set_ylim(np.min(energies) / normalization, np.max(energies) / normalization)  # Resize the limit of the figure
		ax.set_xlim(x_vector[0], x_vector[-1])

		if labels is not None:  # If labels are provided
			ax.set_xlabel(labels[0])
			ax.set_ylabel(labels[1])

		if title is not None:  # If title is provided
			ax.set_title(title)

		ret.append(ax)  # Include the axis to the return

	return ret


def plot_probabilities(x_vector, prob, legend=None, labels=None, ax=None, title=None, limit=0):
	"""
	Function to plot the evolution of the probabilities in the time.
	:param x_vector: (numpy.array) Vector for the x-axis in the plot, usually time
	:param prob: (numpy.matrix) Matrix with the probabilities in time. The dimension is (n x D) where n is the number of time step and D the states
	:param legend: (list) List this the names to put in the legend
	:param labels: (list) List with the names for the labels of the plot
	:param ax:  (matplotlib.axes) Axis in which plot the data
	:param title: (str) Name for the title of the axis
	:param limit: (float) Limit for the maximum population to plot the state
	:return: (list) List with the figure, and the axis where the function has plotted the data.
	"""
	ret = []  # Empty list where the figure, and the axis will be saved
	states = np.shape(prob)[1]  # Number of states

	if legend is None:  # If a legend is not provided
		plot_legend = False
		legend = str(np.zeros(states))  # Save what ever crap, it will not be plotted
	else:
		plot_legend = True

	if ax is None:  # If an axis is not provided
		fig = plt.figure()
		ax = fig.add_subplot()
		ret.append(fig)

	for i in range(states):  # Iterate over all the states
		if np.max(prob[:, i]) > limit:  # If the occupation reaches the limit
			plt.plot(x_vector, prob[:, i], label=legend[i])  # The evolution is plotted

	if plot_legend:  # If a legend is provided
		ax.legend()

	ax.set_ylim(-0.05, 1.05)  # Resize the limit of the figure
	ax.set_xlim(x_vector[0], x_vector[-1])

	if labels is not None:  # If labels are provided
		ax.set_xlabel(labels[0])
		ax.set_ylabel(labels[1])

	if title is not None:  # If title is provided
		ax.set_title(title)

	ret.append(ax)  # Include the axis to the return

	return ret


def compute_adiabatic_parameter(x_vec, states, energies, initial_state, hbar=hbar_muev_ns):
	"""
	Compute the factors need for the FAQUAD and the value of tilde{c}.
	:param x_vec: (numpy.array) Vectors with the values of the parameters we are interested to change with a total of N values
	:param states: (numpy.matrix) Matrix with the D instant eigenstates of the system. The dimension is [N x D x D]
	:param energies: (numpy.matrix) Matrix with the D instant eigenenergies of the system. The dimension is [N x D]
	:param initial_state: (int) Index for the initial state in which we begin the protocol
	:param hbar: (float) Value for hbar
	:return: (list) List with the factors computed and the value of c_tilde.
	"""
	n, dim = np.shape(energies)  # Extract the number of steps for the independent variable, and the number of states

	derivatives = np.zeros([n, dim, dim], dtype=complex)  # Matrix to save the dim coordinates for the eigenstates

	for i in range(0, dim):  # Iterate over all the states
		for j in range(0, dim):  # Iterate over all the coordinates
			derivatives[:, i, j] = np.gradient(states[:, i, j], x_vec)  # Compute the numerical derivative

	counter = 0  # Temp variable to save the numbers of factors computed
	factors = np.zeros([n, dim - 1])  # Matrix to save the factors
	for i in range(0, dim):  # Iterate over all the states
		if i != initial_state:  # If the state is not the initial one
			# Compute the factor, this includes a scalar product
			factors[:, counter] = np.abs(np.sum(np.conjugate(states[:, initial_state, :]) * derivatives[:, i, :], axis=1) / (
					energies[:, initial_state] - energies[:, i] + 10 ** (-16)))
			counter += 1

	# Compute the c_tilda factor, that include a summation over all the states and an integration
	c_tilde = hbar * np.sum(romb(factors, dx=np.abs(x_vec[0] - x_vec[1]), axis=0))

	return factors, c_tilde


def compute_parameters_interpolation(x_vector, factors, c_tilde, nt=1000, hbar=hbar_muev_ns):
	"""
	Function to solve the ODE which gives the result for the parameters in terms of hte adimensional variable s=[0,1] for the FAQUAD protocol
	:param x_vector: (numpy.array) Vector with the values of the independent variable
	:param factors: (numpy.matrix) Matrix with the factors of the FAQUAD protocol
	:param c_tilde: (float) Value for the rescaled adiabatic parameter
	:param nt: Number of steps for the time variable
	:param hbar: Value for h bar
	:return: Vector of times and the parameter
	"""

	def model(y, t):  # EDO to be solved
		return c_tilde / hbar * factor_interpolation(y)

	def factor_interpolation(x):  # Interpolation for the odeint method
		return interp1d(x_vector, 1 / np.sum(factors, axis=1), kind='quadratic', fill_value="extrapolate")(x)

	s = np.linspace(0, 1, nt, endpoint=True)  # Rescaled time parameter s=t/tF
	x_sol = odeint(model, x_vector[0], s)[:, 0]  # Solve numerically the values of the parameter in terms of s

	return s, x_sol


def save_data(name, data, overwrite=None, index=0, ask=True):
	file_dic = 'data/' + name

	if index != 0:
		file_dic += ' (' + str(index) + ')'

	np.save(file_dic + '_temp', data)

	if overwrite is None:  # If the user does not give a preference for the overwriting
		if os.path.isfile(file_dic + '.npy'):  # If the file exists in the folder
			if ask:
				overwrite = question_overwrite(file_dic + '.npy')  # The function will ask if the user want to overwrite the file
			else:
				overwrite = False
		else:
			overwrite = True  # If the file does not exist, them the figure will be saved
	if overwrite:  # Depending on the answer of the user
		np.save(file_dic, data)
	else:
		save_data(name, data, index=index + 1, ask=False)

	os.remove(file_dic + '_temp.npy')


def message_telegram(text):
	bot = telepot.Bot('990722479:AAFes17zw8t4S9oSH8-2B_W4StoODQBxnlU')
	bot.sendMessage(909417112, text)

	return()
