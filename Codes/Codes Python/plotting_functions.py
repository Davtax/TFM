"""
In this file we will include all the functions that I made to customize the figures more automatically
"""
import os.path
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from general_functions import question_overwrite


def save_figure(fig, name, overwrite=None, extension='eps', dic='notes'):
	"""
	Function to save a given figure. We must introduce the name with which we want to save the file, and the device in which we are working. If the
	file already exist them we will be asked if we want to overwrite it. We can also change the extension used to save the image.
	:param fig: (matplotlib.fig object) Figure that we want to save in the device
	:param name: (Str) Name for the file
	:param overwrite: (Bool) Overwrite the file if it already exists. By default, the value is True, but if the file exists the value will be asked
	:param extension: (Str) Extension used to save the figure.
	:param dic: (str) Directory of the folder in with save teh image. If no value is given, the default is TMF/Notes/LaTeX/images.
	"""

	if dic is 'notes':
		dic = '../../Notes/LaTeX/images/'  # We obtain the direction of the folder depending on which device we are working
	elif dic is 'thesis':
		dic = '../../Thesis/Figures/'

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
		print('Figure saved')


def modify_plot(ax, fig=None, lines_width=2.5, label_size=20, tick_label_size=15, title_size=20, length_tick=6.5, legend_size=15, figsize=None,
                x_ticks_vector=None, y_ticks_vector=None, legend=False, tick_direction='in', lines_bool=True, lines_style=None, styles=True,
                colors_bool=True, colors_list=None):
	"""
	Function to modify the parameters of a figure to make it better looking for put in a LaTeX document.
	:param ax: (matplotlib.axis) Axis in which we want to iterate
	:param fig: (matplotlib.figure) Figure if we want to change it's size
	:param lines_width: (float) With for the lines
	:param label_size:  (float) Size for the label title font
	:param tick_label_size: (float) Size for the tick label font
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
	ax.tick_params(axis='both', which='minor', direction=tick_direction, length=length_tick / 2)

	if fig is not None:  # If a figure is given
		# Change the size of the figure
		fig.set_figwidth(figsize[0])
		fig.set_figheight(figsize[1])

	if lines_bool:  # If the function can change the lines
		lines = ax.lines  # Extract the lines

		for i, line in enumerate(lines):  # Iterate over the lines
			line.set_linewidth(lines_width)  # Change the width

		if colors_bool:
			for i, line in enumerate(lines):
				if colors_list is None:
					if len(lines) > 1:
						line.set_color(cycle_color(i))
					else:
						line.set_color(cycle_color(7))
				else:
					line.set_color(colors_list[i])

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

	ax.autoscale(tight=True)


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


def cycle_color(i):
	color_list = ['007BE0', 'E00000', 'F99E00', '00B253', '4123AD', '563100', 'EA2B00', '00293D', '2CDD00', '00F9AE']

	return '#' + color_list[i % len(color_list)]
