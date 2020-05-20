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
from scipy.signal import savgol_filter, medfilt

hbar_muev_ns = ((h / e) / (2 * np.pi)) * 10 ** 6 * 10 ** 9  # Value for the reduced Plank's constant [ueV * ns]


def question_overwrite(name):
	"""
	Make a question if you want to overwrite a certain file that already exists in the directory. There is only two possible answers y -> yes (true) or
	n -> no (False). If the answer is non of this two the question is repeated until a good answer is given.
	:param name: (Str) Name of the file to overwrite
	:return: (Bool) Answer given by the user
	"""

	temp = input('Do you want to overwrite the file ({})?  [y]/n: '.format(name))  # Ask for an answer by keyword input

	if temp == 'y' or temp == '':
		return True
	elif temp == 'n':
		return False
	else:  # If the answer is not correct
		print('I didn\'t understand your answer.')
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

	runnings = range(len(parameters))  # Compute a test parameter to extract the dimension of the Hamiltonian
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
	"""
	These functions are used for the parallel computing, where we need to call the function with just one variable. Here we extract the index of the
	process and unpack the parameters given to solve the system. By default, the value for hbar is 1, and the absolute and relative errors are
	10^{-8} and 10^{-6} respectively.
	:param pack: (list) List with the following parameters: [index of the parallel computation, time, density0, parameters, hamiltonian]. To pass the
	values of default parameters, all of them must be in the sixth element of the list as a dictionary, e.g {'hbar': 1, 'atol': 1e-8}.
	:return: (list) list with the index of the computation al the solution of the system.
	"""

	if len(pack) > 5:
		extra_param = pack[-1]
	else:
		extra_param = {}
	return [pack[0], solve_system(*pack[1:-1], prob=True, **extra_param)[1]]


def sort_solution(data):
	"""
	Function to sort the data obtained for a parallel computation
	:param data: (list) List in which each entry represents one solution of the parallel computation. The elements are also list which contains in
	the first element the index and in the second one the result of the computation.
	:return: (list) List with the data sorted
	"""
	n = len(data)  # Extract the number of computation done

	sorted_sol = [None] * n  # Empty list with the correct number of elements

	for i in range(n):  # Iterate over all the elements
		index = data[i][0]  # Obtain the index of the result
		temp = data[i][1]  # Obtain the result
		sorted_sol[index] = temp  # Save the result in the correct element

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


# TODO: Comentar lo nuevo que he metido tanto en esta función como en la siguiente
def compute_adiabatic_parameter(x_vec, states, energies, initial_state, hbar=hbar_muev_ns, partial_Hamiltonian=None):
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

	if partial_Hamiltonian is None:
		method_1 = True
	else:
		method_1 = False

	if method_1:
		derivatives = np.zeros([n, dim, dim], dtype=complex)  # Matrix to save the dim coordinates for the eigenstates
		for i in range(0, dim):  # Iterate over all the states
			for j in range(0, dim):  # Iterate over all the coordinates
				derivatives[:, j, i] = np.gradient(states[:, j, i], x_vec)  # Compute the numerical derivative

	counter = 0  # Temp variable to save the number of factors computed
	factors = np.zeros([n, dim - 1])  # Matrix to save the factors
	for i in range(0, dim):  # Iterate over all the states
		if i != initial_state:  # If the state is not the initial one
			# Compute the factor, this includes a scalar product
			if method_1:
				factors[:, counter] = np.abs(
					np.sum(np.conjugate(states[:, :, initial_state]) * derivatives[:, :, i], axis=1) / (energies[:, initial_state] - energies[:, i]))
			else:
				for k in range(0, n):
					factors[k, counter] = np.abs(
						np.matmul(np.matmul(np.conjugate(states[k, :, i]), partial_Hamiltonian[k, :, :]), states[k, :, initial_state]) / (
								energies[k, i] - energies[k, initial_state]) ** 2)
			counter += 1

	if method_1:
		for i in range(0, dim - 1):
			factors[:, i] = medfilt(factors[:, i], 5)

	# Compute the c_tilda factor, that include a summation over all the states and an integration
	c_tilde = hbar * np.sum(romb(factors, dx=np.abs(x_vec[0] - x_vec[1]), axis=0))

	return factors, c_tilde


def compute_parameters_interpolation(x_vec, factors, c_tilde, nt=None, hbar=hbar_muev_ns, method_1=True):
	"""
	Function to solve the ODE which gives the result for the parameters in terms of the adimensional variable s=[0,1] for the FAQUAD protocol
	:param x_vec: (numpy.array) Vector with the values of the independent variable
	:param factors: (numpy.matrix) Matrix with the factors of the FAQUAD protocol
	:param c_tilde: (float) Value for the rescaled adiabatic parameter
	:param nt: Number of steps for the time variable
	:param hbar: Value for h bar
	:return: Vector of times and the parameter
	"""
	sig = np.sign(x_vec[1] - x_vec[0])

	if nt is None:  # If the number of elements for the time is not given
		nt = len(x_vec)  # The number of elements is the total number of x_vec

	def factor_interpolation(x):  # Interpolation for the odeint method
		return interp1d(x_vec, 1 / np.sum(factors, axis=1), kind='quadratic', fill_value="extrapolate")(x)

	def model(y, _):  # EDO to be solved
		return sig * c_tilde / hbar * factor_interpolation(y)

	# Rescaled time parameter s=t/tF, the end point is a bit larger than 1 since there are numerical errors, and the desired final x_vec is not
	# reached exactly at s=1
	if method_1:
		s_max = 1.01
	else:
		s_max = 1
	s = np.linspace(0, s_max, nt, endpoint=True)

	counter = 0
	reached = False  # Variable that controls if the final value of x_vec is reached
	while not reached:  # While the final value is not reached
		x_sol = odeint(model, x_vec[0], s)[:, 0]  # Solve numerically the values of the parameter in terms of s
		counter += 1
		if np.any(sig * x_sol > sig * x_vec[-1]):  # If the final value is reached
			index_max = np.where(sig * x_sol > sig * x_vec[-1])[0][0]  # Save the first index in which the final value is obtained
			reached = True  # Exit the while loop
		elif not method_1:
			print('The parameter may not reached the maximum value, verify it.\nConsider use the method 1.')
			index_max = len(s) - 1
			reached = True
		else:  # If the final value is not yet reached
			s *= 1.1  # Increase the values for s to give more time to reach the value
			reached = False  # Continue in the loop (this line is not necessary since the variable is still = False

		if counter > 20:
			print('The limit value has not been reached.')
			return ()

	s = np.linspace(0, 1, index_max + 1)  # Compute a new vector that goes exactly up to the unity
	x_sol = interp1d(s, x_sol[:index_max + 1], kind='quadratic')  # Interpolate and contract the data to reach the final value at s=1
	# We must use index_max + 1 in order to get also the value that fulfill the condition x_vex > limit

	return s, x_sol


def save_data(name, data, overwrite=None, index=0, ask=True):
	"""
	Function to save the data in a numpy file. All the files are saved in the folder "data", a sub-folder of "Codes Python". This function has a
	protection for not overwriting and save a temp file if the overwriting question is not asked.
	:param name: (str) String with the name of the file in which save the data
	:param data: (numpy.array or list) Data to be saved
	:param overwrite: (bool) Condition to overwrite or not the data. If a value is not given then the function will ask by default
	:param index: (int) Index to include in the file name is the previous onw is already occupied
	:param ask: (bool) Condition than controls if the question to overwrite is done.
	"""
	file_dic = 'data/' + name  # Directory in which save the data

	if index != 0:  # If an index is specified
		file_dic += ' (' + str(index) + ')'  # Include the index in the same

	np.save(file_dic + '_temp', data)  # Save a temp file to prevent an error during the question

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
	else:  # If the user does not want to over write, a copy is saved
		# The copy will include the typical (1) at the end of the name, if this already exist then (2) without asking. If the file also exist then (3)
		# and so on until an empty number is reached.
		save_data(name, data, index=index + 1, ask=False)

	os.remove(file_dic + '_temp.npy')  # If the file is correctly saved the temp file is removed


def compute_limits(hamiltonian, parameters, limit_1, limit_2, state_1, state_2, instant_state, x_vec, y_vec, index_x, index_y, window=None, pol=3,
                   filter_bool=False):
	"""
	Function to compute the limits at which some given states are populated in a given instant state. The limits will be find at the critical values
	of x_vec, and repeated for each value in y_vec. The data obtained is then filtered to obtain a smooth result if wanted.
	:param hamiltonian: (function) Function pointing at the hamiltonian in which we are interested
	:param parameters: (list) List with the parameters for the hamiltonian. Those parameters that we will change can be set to 0
	:param limit_1: (list) Limit value to obtain for the first state
	:param limit_2: (list) Limit value to obtain for the second state
	:param state_1: (int) Index for the first state in the Hamiltonian basis
	:param state_2: (int) Index for the second state in the Hamiltonian basis
	:param instant_state: (int) Index for the instantaneous state in which we are interested. These states are numerated beginning for the lest
	energetic one
	:param x_vec: (numpy.array) Array with the values at which we want to find the limit
	:param y_vec: (list or numpy.array) Values for the parameters that will change. If more than one parameter is changed at the same time then they
	must be passed as a sorted list
	:param index_x: (ind) Index in the parameters list pointing at the parameter at which find the critical value
	:param index_y: (ind or list) Index(s) in the parameters list pointing at the parameter(s) that changes()
	:param window: (int) Number of points to make the filter at each step
	:param pol: (int) Grade of the polynomial used for the filter
	:param filter_bool: (bool) If the filter is applied to the obtained data
	:return: (list) List with two elements, each of them is a numpy.array with the data of the limits (x_vector) in terms of the other parameter
	(y_vector) at which the population (limit_i) is reached (state_i).
	"""

	if type(index_y) is list:  # If the y_vector point at more than one parameter
		ny = len(y_vec[0])  # Extract the number of elements in each parameter
		islist = True  # Save if is a list
	else:  # If the y_vector is just one parameter
		ny = len(y_vec)  # Extract the number of elements in the parameter
		islist = False  # Save if is a list

	# Initialize the lists in which the critical values will be saved
	limit_1_vec = []
	limit_2_vec = []

	parameters[index_x] = x_vec  # Overwrite the data for the x_vector

	for i in range(ny):  # Iterate over all the elements of y_vector
		if islist:  # If y_vector is a list
			for j in range(0, len(index_y)):  # Iterate over all the parameters saved in y_vector
				parameters[index_y[j]] = y_vec[j][i]  # Save each parameter in the correct index of the list
		else:  # If is just one parameter
			parameters[index_y] = y_vec[i]  # Save the parameter in the correct index of the list
		_, states = compute_eigensystem(parameters, hamiltonian)  # Extract the eigenstates in terms of x_vector

		# Extract the population of each state in the instant_state in which we are interested
		population_1 = np.abs(states[:, state_1, instant_state]) ** 2
		population_2 = np.abs(states[:, state_2, instant_state]) ** 2

		# To compute the critical value we will use just brute force, and obtain which of the computed values are closes to the desired one.
		# Now we compute the difference to the desired limit
		temp1 = limit_1 - population_1
		temp2 = limit_2 - population_2

		# Check if for each state the given range in x_vector obtain the limit, that is, the differences are positive and negative at some point
		if np.any(temp1 > 0) * np.any(temp1 < 0) * np.any(temp2 > 0) * np.any(temp2 < 0):
			# Save the index at which the closer value to the desired limit is reached
			index1 = np.where(np.abs(temp1) == np.min(np.abs(temp1)))[0][0]
			index2 = np.where(np.abs(temp2) == np.min(np.abs(temp2)))[0][0]

			# Save the value of x_vector at which the value is obtained
			limit_1_vec.append(x_vec[index1])
			limit_2_vec.append(x_vec[index2])
		else:  # If the range is not large enough
			print('The x vector limits can not achieve the minimum value given')  # Print a message warning of the error
			return ()  # Return nothing

	# Convert the lists in arrays
	limit_1_vec = np.array(limit_1_vec)
	limit_2_vec = np.array(limit_2_vec)

	if filter_bool:  # If the user wants to apply the filter
		if window is None:  # If a window is not given
			window = np.int(len(x_vec) / 10)  # Compute a window more or less wise
			if not window % 2:  # If the window is even
				window -= 1  # Change its value to make it odd
		if pol > window:  # If the polynomial order is larger than the window
			pol = 2

		# Compute the Savitzky–Golay filter
		limit_1_vec = savgol_filter(limit_1_vec, window, pol)
		limit_2_vec = savgol_filter(limit_2_vec, window, pol)

	return limit_1_vec, limit_2_vec


# TODO: The value for the period is not good, revisit the function to fix it
def compute_period(x_sol, hamiltonian, parameters, hbar, index, state):
	"""
	Compute the characteristic period of the FAQUAD protocol
	:param x_sol: (list, scipy.interpolated) List (if more than one) with all the interpolated functions representing the independent variables
	:param hamiltonian: (function) Function pointing to the Hamiltonian in which are interested
	:param parameters: (list) List with the parameters of the system. The elements of the parameters that run can be set to 0
	:param hbar: (float) Value for the reduced Plank's constant
	:param index: (list) List of the index in the list parameters of the variables in x_sol
	:return: (float) Value for the period of the FAQUAD protocol.
	"""
	s = np.linspace(0, 1, 2 ** 15 + 1)  # Compute the s parameter with the correct number of element for doing the romb algorithm for the integral
	ns = len(s)  # Number of element in s

	x_sol_list = []  # Empty list to save the independent variables
	if type(x_sol) is list:  # If there are more than one independent variable
		for i in range(0, len(x_sol)):  # Iterate over all the variables
			x_sol_list.append(x_sol(s))  # Save all the variables in a list
	else:  # If there is only one independent variable
		x_sol_list = [x_sol(s)]  # Save the variable in a list
		index = [index]  # Make the index a list with only one element

	for i in range(0, len(index)):  # Iterate over all the independent variables to include in the list of parameters
		parameters[index[i]] = x_sol_list[i]  # Include the vec of independent variables

	# TODO: For the moment we can only have one independent variable
	h_matrix = create_hypermatrix(parameters, hamiltonian)  # Construct the hypermatrix of the hamiltonian
	energies = np.linalg.eigvalsh(h_matrix)  # Extract only the instant eigenenergies of the Hamiltonian

	n = np.shape(energies)[1]  # Extract the dimension of the Hamiltonian

	e_g = np.zeros([n - 1, ns])  # Array in which rows will be saved the gaps between the energies
	for i in range(0, n - 1):  # Iterate over all the gaps
		if i != state:
			e_g[i, :] = np.abs(energies[:, state] - energies[:, i])  # Compute the gaps, always a positive value

	phi = romb(np.sum(e_g, axis=0), dx=(s[1] - s[0])) / hbar  # Compute the integral of the gaps

	t = 2 * np.pi / phi  # Compute the period

	return t
