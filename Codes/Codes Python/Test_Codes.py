
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import modify_plot, save_figure

# Path the to file that we want to load
folder = 'data/'
file = 'FAQUAD_DQD_1HH'
extension = '.npy'
file_dic = folder + file + extension

data = np.load(file_dic, allow_pickle=True)  # Load the data, allow_pickle enable to load a list
print(data[-1])

