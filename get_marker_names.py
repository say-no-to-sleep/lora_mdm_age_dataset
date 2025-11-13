import ezc3d
import sys
import os

filename = "./c3d_files/SUBJ100 (0).c3d"

if not os.path.exists(filename):
    print(f"Error: File not found.")
    print(f"I looked for '{filename}' in this directory:")
    print(f"{os.getcwd()}")
    print("\nPlease make sure the file is in the same folder as the script,")
    print("or change the 'filename' variable to the full file path.")
    sys.exit() # Stop the script

try:
    c = ezc3d.c3d(filename)

    marker_labels = c['parameters']['POINT']['LABELS']['value']
    
    # Also get the number of markers (a good sanity check)
    num_markers = c['parameters']['POINT']['USED']['value'][0]

    # 3. Print the results in a clean list
    print(f"--- Inspection Results for: {filename} ---")
    print(f"Total markers found: {num_markers}\n")
    
    print("Marker Names (in order):")
    # We use enumerate to get a clean, numbered list (e.g., "0: L_HEEL")
    for i, name in enumerate(marker_labels):
        print(f"  {i}: {name}")

    # Save results to a text file
    output_filename = filename.replace('.c3d', '_marker_labels.txt')
    with open(output_filename, 'w') as f:
        f.write(f"--- Inspection Results for: {filename} ---\n")
        f.write(f"Total markers found: {num_markers}\n\n")
        f.write("Marker Names (in order):\n")
        for i, name in enumerate(marker_labels):
            f.write(f"  {i}: {name}\n")
    print(f"\nMarker names have also been saved to '{output_filename}'.")


except KeyError:
    print(f"Error: Could not find the standard 'POINT' or 'LABELS' parameters.")
    print("This might be a corrupt or non-standard C3D file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
