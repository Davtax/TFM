"""
In this file we will include all the functions that I made to customize the figures more automatically
"""
import os.path
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from general_functions import question_overwrite


def file_location(device):
	"""
	Give the location for the images used in the LaTeX file depending on which device we are working. The only possible devices is laptop or desktop.
	If non of these values is given to the function then it will ask for a correct device.
	:param device: (Str) Name of the device that we are using
	:return: (Str) Directory to the file of images used in the given device
	"""

	if device == 'laptop':
		return 'C:\\Users\\David\\OneDrive - Universidad Autonoma de Madrid\\Universidad\\Master\\TFM\\Notes\\LaTeX\\images\\'
	if device == 'desktop':
		return 'D:\\OneDrive - Universidad Autonoma de Madrid\\Universidad\\Master\\TFM\\Notes\\LaTeX\\images\\'
	else:  # If the device given is not correct
		temp = input('Please enter a correct device: laptop or desktop')  # The function will ask a device name by keyword input
		return file_location(temp)  # The function is repeated with the given input given to check the validity of the new answer


def save_figure(fig, name, device=None, overwrite=None, extension='eps', dic=None):
	"""
	Function to save a given figure. We must introduce the name with which we want to save the file, and the device in which we are working. If the
	file already exist them we will be asked if we want to overwrite it. We can also change the extension used to save the image.
	:param fig: (matplotlib.fig object) Figure that we want to save in the device
	:param name: (Str) Name for the file
	:param device: (Str) Device in which we are working. By default, an empty string is given we are going to be asked to introduce the device
	:param overwrite: (Bool) Overwrite the file if it already exists. By default, the value is True, but if the file exists the value will be asked
	:param extension: (Str) Extension used to save the figure.
	"""

	if dic is None:
		dic = file_location(device)  # We obtain the direction of the folder depending on which device we are working

	file_dic = dic + name + '.' + extension  # Complete the directory of the file including its extension
	file_dic_copy = dic + name + '.' + 'png'  # Complete the directory of the copy file in .png

	if overwrite is None:  # If the user does not give a preference for the overwriting
		if os.path.isfile(file_dic):  # If the file exists in the folder
			overwrite = question_overwrite(file_dic)  # The function will ask if the user want to overwrite the file
		else:
			overwrite = True  # If the file does not exist, them the figure will be saved

	if overwrite:  # Depending on the answer of the user
		fig.savefig(file_dic, format=extension,
		            bbox_inches="tight")  # Save the figure with the corresponding file direction and the correct extension
		if extension != 'png':
			fig.savefig(file_dic_copy, format='png',
			            bbox_inches="tight")  # Save the figure with the corresponding file direction and the correct extension

	return ()


def modify_plot(ax, fig=None, lines_width=2.5, label_size=20, tick_label_size=15, annotation_size=20, title_size=20, length_tick=6.5, legend_size=15,
                figsize=None, x_ticks_vector=None, y_ticks_vector=None, legend=False, tick_direction='in', lines_bool=True, lines_style=None,
                styles=True):
	"""
	Function to modify the parameters of a figure to make it better looking for put in a LaTeX document.
	:param ax: (matplotlib.axis) Axis in which we want to iterate
	:param fig: (matplotlib.figure) Figure if we want to change it's size
	:param lines_width: (float) With for the lines
	:param label_size:  (float) Size for the label title font
	:param tick_label_size: (float) Size for the tick label font
	:param annotation_size:  (float) Size for the text added to the plot
	:param title_size: (float) Size for the title font
	:param length_tick:  (float) Length of the major ticks
	:param legend_size: (float) Size for the legend font
	:param figsize: (tuple) Tuple with the dimension of the figure, [width,height] in inches
	:param x_ticks_vector: (numpy.array) Array with the major ticks for the x-axis
	:param y_ticks_vector:  (numpy.array) Array with the major ticks for the y-axis
	:param legend: (Bool) Boolean variable that control if exist a legend in the given axis
	:param tick_direction: (str) String specifying the direction for the ticks bars ('in', 'out' or 'inout')
	:param lines_bool: (Bool) Variable controlling if the function change the lines
	:param styles: (Bool) If the user wants to change the styles of the lines
	:param lines_style: (List) List with the line style
	"""
	# Modify the size of the labels
	ax.xaxis.label.set_size(label_size)
	ax.yaxis.label.set_size(label_size)

	ax.title.set_size(title_size)  # Modify the font size of the title

	# Modify the size, length and directions of the two ticks
	ax.tick_params(axis='both', which='major', labelsize=tick_label_size, direction=tick_direction, length=length_tick)

	if fig is not None:  # If a figure is given
		# Change the size of the figure
		fig.set_figwidth(figsize[0])
		fig.set_figheight(figsize[1])

	if lines_bool:  # If the function can change the lines
		lines = ax.lines  # Extract the lines

		for line in lines:  # Iterate over the lines
			line.set_linewidth(lines_width)  # Change the width

		if lines_style is None:  # If a line style is not given
			lines_style = ['-', '--', '-.', ':']  # Obtain the default line styles
		if styles:
			for i in range(0, len(lines)):  # Iterate over all the lines
				lines[i].set_linestyle(lines_style[i % len(lines_style)])  # Change the line style

	if x_ticks_vector is not None:  # If a vector with xticks is provided
		ax.xaxis.set_ticks(x_ticks_vector)  # Change the xticks

	if y_ticks_vector is not None:  # If a vector with yticks is provided
		ax.yaxis.set_ticks(y_ticks_vector)  # Change the yticks

	if legend:  # If exist a legend
		ax.legend(fontsize=legend_size)  # Change the legend font size


def zoomed_plot(fig, ax, pos, size, data, x_limit, y_limit, color='tab:blue', line_style='-', vertex=None):
	inset_ax = fig.add_axes([pos[0], pos[1], size[0], size[1]])
	inset_ax.plot(data[0], data[1], color=color, linestyle=line_style)

	inset_ax.set_xlim(x_limit)
	inset_ax.set_ylim(y_limit)

	inset_ax.xaxis.set_visible('False')
	inset_ax.yaxis.set_visible('False')

	if vertex is not None:
		mark_inset(ax, inset_ax, loc1=vertex[0], loc2=vertex[1], fc="none", ec="0.5")

	return inset_ax
