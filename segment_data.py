import os
import h5py
import numpy as np
import shutil

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def create_segments_and_save(input_folder, output_folder, window_size, overlap_percentage):
    # Copy the folder structure
    shutil.copytree(input_folder, output_folder)

    # Traverse the new folder and modify each file
    for foldername, subfolders, filenames in os.walk(output_folder):

        # Exclude folders starting with a dot
        subfolders[:] = [folder for folder in subfolders if not folder.startswith('.')]
    
        for filename in filenames:
            if not filename.startswith('.'):
                file_path = os.path.join(foldername, filename)
                with h5py.File(file_path, 'r') as f:
                    dataset_name = get_dataset_name(file_path)
                    data = f.get(dataset_name)[()]

                    # Calculate overlap in samples
                    overlap_samples = int(window_size * overlap_percentage / 100)

                    # Create segments
                    for i in range(0, data.shape[1] - window_size + 1, overlap_samples):
                        segment_data = data[:, i:i + window_size]

                        # Save segment to new H5 file
                        output_filename_prefix = os.path.splitext(filename)[0]
                        output_filename = f"{output_filename_prefix}_segment_{i // overlap_samples}.h5"
                        output_filepath = os.path.join(foldername, output_filename)

                        with h5py.File(output_filepath, 'w') as output_h5file:
                            dataset_name = get_dataset_name(output_filepath)
                            new_dataset_name = dataset_name
                            new_dataset = output_h5file.create_dataset(new_dataset_name, data=segment_data)  # Adjust dataset names if needed

                # Delete the original file after processing
                os.remove(file_path)       

# Example usage
input_folder = "/Users/iacopoermacora/Final Project data global_min_max_scaling"
output_folder = "/Users/iacopoermacora/Final Project data global_min_max_scaling segmented"
window_size = 160  # 0.8 seconds
overlap_percentage = 40  # 40% overlap

create_segments_and_save(input_folder, output_folder, window_size, overlap_percentage)
