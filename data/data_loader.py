
'''Utlities for loading matlab data. This module takes the raw matlab folder containing the battery datasets and extracts the necessary data'''

import os

# %%
from scipy.io import loadmat


# %%
def load_data(file_path):
    '''load a matlab(.mat) battery file.
    Input: file_path
    Output: dictionary produced by loadmat'''
    return loadmat(file_path)


# %%
def load_all_batteries(folder):
    '''load every matlab file from the folder
    Input: folder path with each file in the folder saved by the battery name or ID
    Output: dict{battery_id: matlab_dictionary}'''
    batteries = {}
    for file in os.listdir(folder):
        if file.endswith('.mat'):
            battery_name = os.path.splitext(file)[0]
            batteries[battery_name] = load_data(os.path.join(folder, file))
    return batteries

# %%



