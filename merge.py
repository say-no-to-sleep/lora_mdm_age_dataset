import ezc3d
import numpy as np

# --- Configuration ---
# 1. Add your sequential C3D filenames here
file_list = [
    "./van_criekinge_unprocessed/able_bodied/SUBJ01/SUBJ1 (0).c3d", 
    "./van_criekinge_unprocessed/able_bodied/SUBJ01/SUBJ1 (1).c3d", 
    "./van_criekinge_unprocessed/able_bodied/SUBJ01/SUBJ1 (2).c3d", 
    "./van_criekinge_unprocessed/able_bodied/SUBJ01/SUBJ1 (3).c3d"
]
output_filename = "concatenated_trial.c3d"

# --- Main Logic ---
if not file_list:
    print("Your file list is empty. Please add filenames to continue.")
else:
    c3d_objects = [ezc3d.c3d(f) for f in file_list]

    # Extract point and analog data from each file
    point_data_list = [c['data']['points'] for c in c3d_objects]
    analog_data_list = [c['data']['analogs'] for c in c3d_objects]

    # Concatenate the data along the "frames" axis (axis 2)
    concatenated_points = np.concatenate(point_data_list, axis=2)
    concatenated_analogs = np.concatenate(analog_data_list, axis=2)

    # Create a new c3d object using the first file as a template for parameters
    c_new = ezc3d.c3d()
    c_new['parameters'] = c3d_objects[0]['parameters']
    
    # Add the concatenated data to the new object
    c_new['data']['points'] = concatenated_points
    c_new['data']['analogs'] = concatenated_analogs

    # Save the new, combined C3D file
    c_new.write(output_filename)
    
    print(f"Success! Merged {len(file_list)} files into '{output_filename}'.")