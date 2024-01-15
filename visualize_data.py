import mne
import h5py
import numpy as np 
import os

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('\\')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def scale_matrix(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    scaled_matrix = (matrix - min_value) / (max_value - min_value)
    return scaled_matrix

file_name = "C:/Users/lazar/OneDrive/Υπολογιστής/AI MSc/Pattern Recognition & Deep Learning/group__assignment/Final Project Data/Cross/train/rest_113922_5.h5"

with h5py.File(file_name, 'r') as f:
    dataset_name = get_dataset_name(file_name)
    data = f.get(os.path.basename(dataset_name))[()]

data = scale_matrix(data)
#data = np.reshape(matrix, (matrix.shape[0], -1)).T
ch_names = ['Ch' + str(i + 1) for i in range(data.shape[0])]
ch_types = ['eeg' for i in range(data.shape[0])]
sfreq = 2034
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

raw = mne.io.RawArray(data, info)

print(raw.info)
print('Duration:', raw.times[-1], 'seconds')
raw.plot()
print('x')