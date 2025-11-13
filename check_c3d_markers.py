import c3d
import sys
from pathlib import Path

# --- Configuration ---

# 1. SET THIS to the path of your raw C3D files
# This is the directory containing SUBJ01, SUBJ02, etc.
BASE_DATA_DIR = "./van_criekinge_unprocessed_1/able_bodied"

# 2. This is the set of markers your pipeline is looking for.
# I copied this directly from your '2_fit_smpl_markers.py' script.
ANATOMICAL_MARKERS = {
    # Head / upper trunk
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN",
    # Pelvis triad
    "LASI", "RASI", "SACR",
    # Shoulders / elbows / wrists
    "LSHO", "RSHO",
    "LELB", "RELB",
    "LWRA", "LWRB",
    "RWRA", "RWRB",
    # Lower limbs
    "LKNE", "RKNE",
    "LANK", "RANK",
    "LHEE",  "RHEE",
    "LTOE",   "RTOE",
}

# --- Main Script ---

def main():
    data_dir = Path(BASE_DATA_DIR)
    if not data_dir.is_dir():
        print(f"Error: Directory not found. Please check the BASE_DATA_DIR path.")
        print(f"I looked for: {data_dir.resolve()}")
        sys.exit(1)

    print(f"Scanning for .c3d files in: {data_dir}\n")
    
    # Use rglob to find all .c3d files in all subdirectories
    c3d_files = sorted(data_dir.rglob("*.c3d"))
    if not c3d_files:
        print("No .c3d files found. Check your path.")
        sys.exit(0)

    print(f"Found {len(c3d_files)} total .c3d files. Now checking...")
    
    problem_files = []

    for filepath in c3d_files:
        try:
            with open(filepath, 'rb') as f:
                reader = c3d.Reader(f)
                
                # Get the raw marker labels
                raw_labels = [s.strip() for s in reader.point_labels if s.strip()]
                if not raw_labels:
                    # Fallback if labels are empty
                    raw_labels = [f"marker_{i:02d}" for i in range(reader.header.point_count)]

                # Clean the labels, same as your 1_dataset_prep.py script
                cleaned_labels = []
                for m in raw_labels:
                    cleaned = m.strip().upper()
                    if ":" in cleaned:
                        cleaned = cleaned.split(":")[-1]
                    cleaned_labels.append(cleaned)
                
                # This is the check from 2_fit_smpl_markers.py
                # Count how many of our cleaned markers are in the "known" set
                found_anatomical_count = 0
                for label in cleaned_labels:
                    if label in ANATOMICAL_MARKERS:
                        found_anatomical_count += 1
                
                # This is the failure condition
                if found_anatomical_count < 12:
                    problem_files.append((filepath, found_anatomical_count))

        except Exception as e:
            print(f"\n--- ERROR reading file: {filepath} ---")
            print(f"   {e}")
            print("   This file may be corrupt. Skipping.\n")
    
    # --- Print Report ---
    print("\n" + "="*40)
    print("         Scan Complete.          ")
    print("="*40)

    if not problem_files:
        print("\nGood news! All files passed the check (> 12 anatomical markers).")
    else:
        print(f"\nFound {len(problem_files)} problematic files:")
        print("These files have < 12 recognizable anatomical markers")
        print("and will fail '2_fit_smpl_markers.py':\n")
        
        for path, count in problem_files:
            # Get the relative path to make it cleaner
            try:
                rel_path = path.relative_to(data_dir)
            except ValueError:
                rel_path = path
            print(f"  [ {count:2d} markers ]  {rel_path}")
            
    print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    main()