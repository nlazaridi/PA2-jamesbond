import os
import shutil
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import h5py

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def time_wise_min_max_scaling(matrix, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_matrix = scaler.fit_transform(matrix.T).T  # Transpose for time-wise scaling
    return scaled_matrix

def time_wise_z_score_scaling(matrix):
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix.T).T  # Transpose for time-wise scaling
    return scaled_matrix
    
def copy_and_modify_folder(original_folder, new_folder, scale_method):
    # Copy the folder structure
    shutil.copytree(original_folder, new_folder)

    # Traverse the new folder and modify each file
    for foldername, subfolders, filenames in os.walk(new_folder):

        # Exclude folders starting with a dot
        subfolders[:] = [folder for folder in subfolders if not folder.startswith('.')]
    
        for filename in filenames:
            if not filename.startswith('.'):
                file_path = os.path.join(foldername, filename)
                print(file_path)
    
                # Modify the file (replace this part with your own modification logic)
                with h5py.File(file_path,'r') as f:
                    dataset_name = get_dataset_name(file_path)
                    matrix = f.get(dataset_name)[()]
                    print(type(matrix))
                    print(matrix.shape)
    
                # Generating a sample signal (replace this with your actual signal)
                # For example, a sine wave with a higher sampling rate
                fs_original = 2034  # Original sampling rate (in Hz)
                t = np.arange(0, 1, 1/fs_original)  # Time array
                #original_signal = np.sin(2 * np.pi * 5 * t)  # Example signal (5 Hz sine wave)
                original_signal = matrix
                
                # Downsampling
                desired_fs = 200  # Desired sampling rate (in Hz)
                factor = fs_original // desired_fs  # Downsampling factor
                fs_downsampled = fs_original / factor  # New sampling rate
                
                # Applying low-pass filter before downsampling
                nyquist = 0.5 * fs_original
                cutoff = 0.9 * desired_fs  # Adjust cutoff frequency as needed
                b, a = signal.butter(8, cutoff / nyquist, 'low')  # Creating a low-pass Butterworth filter
                filtered_signal = signal.filtfilt(b, a, original_signal)
                print(filtered_signal.shape)
                
                downsampled_signal = filtered_signal[:,::factor]  # Downsampling by selecting every 'factor' sample
    
                if scale_method == 1:
                    # Apply time-wise min-max scaling
                    scaled_matrix = time_wise_min_max_scaling(downsampled_signal)
                else:
                    # Apply time-wise Z-score scaling
                    scaled_matrix = time_wise_z_score_scaling(downsampled_signal)

                file_path = os.path.join(foldername, filename)

                os.remove(file_path)
                
                with h5py.File(file_path, 'w') as f:
                    dataset_name = get_dataset_name(file_path)
                    # Modify the dataset with the new matrix
                    # Create a new dataset with the modified shape and data
                    new_dataset_name = dataset_name
                    new_dataset = f.create_dataset(new_dataset_name, data=scaled_matrix)
            
            '''    
            # Plotting the original and downsampled signals
            plt.figure(figsize=(10, 6))
            
            plt.subplot(2, 1, 1)
            plt.title("Original Signal")
            plt.plot(t, original_signal)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            
            plt.subplot(2, 1, 2)
            plt.title("Downsampled Signal")
            t_downsampled = np.arange(0, len(downsampled_signal)) * (1 / desired_fs)
            plt.plot(t_downsampled, downsampled_signal)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            plt.show()
            '''
            

if __name__ == "__main__":
    original_folder = "/Users/iacopoermacora/Desktop/Final Project data"
    new_folder_a = "Final Project data min_max_scaling"
    new_folder_b = "Final Project data score_scaling"

    copy_and_modify_folder(original_folder, new_folder_a, 1)
    copy_and_modify_folder(original_folder, new_folder_b, 0)