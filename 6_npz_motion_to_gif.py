#!/usr/bin/env python3
"""
Loads a HumanML3D-style NPZ file (from 3_export_humanml3d.py) 
and reconstructs the 3D motion as a GIF.

Use this to visualize the difference between the root-centered data
and the fully reconstructed data.

Example:

# 1. To see the "moonwalk" (what the model was learning from):
python reconstruct_gif.py --input data/humanml3d/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --output moonwalk.gif

# 2. To see the *correct* motion with the trajectory:
python reconstruct_gif.py --input data/humanml3d/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --output correct_walk.gif --use_trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
from tqdm import tqdm

# Define the kinematic chain (skeleton) for the 22-joint HumanML3D skeleton
# This is hardcoded to avoid external dependencies.
# Joint names: 0:pelvis, 1:L_hip, 2:R_hip, 3:spine1, 4:L_knee, 5:R_knee, 6:spine2, 
# 7:L_ankle, 8:R_ankle, 9:spine3, 10:L_foot, 11:R_foot, 12:neck, 13:L_collar, 
# 14:R_collar, 15:head, 16:L_shoulder, 17:R_shoulder, 18:L_elbow, 19:R_elbow, 
# 20:L_wrist, 21:R_wrist
KINEMATIC_CHAIN = [
    (0, 1), (1, 4), (4, 7), (7, 10),  # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),  # Right Leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
    (9, 13), (13, 16), (16, 18), (18, 20), # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21)  # Right Arm
]

def load_data(npz_path, use_trajectory):
    """Loads motion data and optionally reconstructs it with the pelvis trajectory."""
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error: Could not load file {npz_path}")
        print(e)
        sys.exit(1)
        
    if 'joints' not in data:
        print(f"Error: 'joints' array not found in {npz_path}")
        sys.exit(1)
        
    root_centered_pose = data['joints']
    
    if use_trajectory:
        if 'pelvis_traj' not in data:
            print(f"Error: --use_trajectory was specified, but 'pelvis_traj' array not found.")
            sys.exit(1)
        
        pelvis_trajectory = data['pelvis_traj']
        
        # Ensure lengths match
        if root_centered_pose.shape[0] != pelvis_trajectory.shape[0]:
            print(f"Warning: Mismatch in frame count. Trimming to shortest.")
            min_len = min(root_centered_pose.shape[0], pelvis_trajectory.shape[0])
            root_centered_pose = root_centered_pose[:min_len]
            pelvis_trajectory = pelvis_trajectory[:min_len]

        # Reconstruct by adding the trajectory back to the root-centered pose
        # (T, 22, 3) + (T, 1, 3) -> (T, 22, 3)
        motion_data = root_centered_pose + pelvis_trajectory[:, None, :]
        print("Reconstructed motion with pelvis trajectory.")
    else:
        motion_data = root_centered_pose
        print("Using root-centered 'joints' data (moonwalk).")
        
    return motion_data

def setup_plot_limits(ax, motion_data):
    """Calculates stable axis limits to prevent camera wobble."""
    # Data is (T, J, D) where D is (X, Y, Z)
    # We plot (X, Z, Y) because Y is UP in data, but Z is UP in mplot3d
    X = motion_data[..., 0]
    Y_up = motion_data[..., 1]
    Z_fwd = motion_data[..., 2]

    # Find the min/max of all data
    max_range = np.array([
        X.max() - X.min(),
        Y_up.max() - Y_up.min(),
        Z_fwd.max() - Z_fwd.min()
    ]).max() / 2.0
    
    # Center the plot
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y_up.max() + Y_up.min()) * 0.5
    mid_z = (Z_fwd.max() + Z_fwd.min()) * 0.5
    
    # Set the limits
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range) # Plot Z (fwd) on Y-axis
    ax.set_zlim(mid_y - max_range, mid_y + max_range) # Plot Y (up) on Z-axis

    ax.set_xlabel("X")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Up)")
    
    return (mid_x, mid_y, mid_z, max_range)

def main():
    parser = argparse.ArgumentParser(description="Reconstruct HumanML3D NPZ file as a GIF.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .npz file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .gif file")
    parser.add_argument("--use_trajectory", action="store_true", help="Add the 'pelvis_traj' to the 'joints' data.")
    parser.add_argument("--downsample", type=int, default=2, help="Plot every Nth frame to speed up GIF (default: 2)")
    args = parser.parse_args()

    # 1. Load and (optionally) reconstruct the data
    motion_data = load_data(args.input, args.use_trajectory)
    
    # Downsample frames
    motion_data = motion_data[::args.downsample]
    total_frames = motion_data.shape[0]

    # 2. Setup the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=-135) # Set a nice viewing angle
    
    # Calculate fixed limits so the plot doesn't wobble
    plot_limits = setup_plot_limits(ax, motion_data)

    # 3. Define the animation update function
    pbar = tqdm(total=total_frames, desc="Rendering GIF")
    
    def update(frame_index):
        ax.clear()
        
        # Reset limits and labels
        mid_x, mid_y, mid_z, max_range = plot_limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (Forward)")
        ax.set_zlabel("Y (Up)")
        
        # Get the pose for this frame
        frame_pose = motion_data[frame_index]
        
        # Swap Y and Z for plotting (Y is UP in data, Z is UP in mplot3d)
        points = frame_pose[:, [0, 2, 1]] # (X, Z_fwd, Y_up)

        # Plot the joints
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', marker='.')

        # Plot the bones
        for parent, child in KINEMATIC_CHAIN:
            p1 = points[parent]
            p2 = points[child]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=1.5)
            
        ax.set_title(f'Frame {frame_index * args.downsample} (Displaying 1/{args.downsample})')
        pbar.update(1)

    # 4. Create and save the animation
    print(f"Creating animation with {total_frames} frames (downsampled 1/{args.downsample})...")
    
    # 20 FPS = 50ms interval
    # We adjust the interval based on the original downsample to keep apparent speed
    interval = 50 * args.downsample 
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=False)
    
    print(f"Saving GIF to {args.output}... (This may take a moment)")
    anim.save(args.output, writer='pillow', fps=(1000 // interval))
    pbar.close()
    plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    main()