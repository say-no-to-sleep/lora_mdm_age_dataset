#!/usr/bin/env python3
"""
Loads a 263-dim HumanML3D-style FEATURE vector (.npy file) from 
4_motion_process.py, reconstructs the 3D motion, and saves it as a GIF.

This script MUST be run from the root of your LoRA-MDM project,
so it can import the 'utils' package.

Example (using your paths):

# 1. To see the OLD (buggy) motion:
python visualize_features.py \
    --input van_criekinge/train/motions/SUBJ01/SUBJ1_1_humanml3d_22joints.npy \
    --mean van_criekinge/Mean.npy \
    --std van_criekinge/Std.npy \
    --output old_walk.gif

# 2. To see the NEW (fixed) motion:
python visualize_features.py \
    --input Comp_v6_KLD01/train/motions/SUBJ01/SUBJ1_1_humanml3d_22joints.npy \
    --mean Comp_v6_KLD01/meta/mean.npy \
    --std Comp_v6_KLD01/meta/std.npy \
    --output new_walk.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import os
from tqdm import tqdm
import torch

# --- Imports from the LoRA-MDM/utils directory ---
# This requires the script to be run from a place where 'utils' is importable
try:
    from utils.quaternion import qrot, qinv
except ImportError:
    print("Error: Could not import quaternion utils.")
    print("Please run this script from the root of the LoRA-MDM project.")
    sys.exit(1)

# --- Copied functions from 4_motion_process.py ---
# We copy these to make the script self-contained and avoid import errors.

def recover_root_rot_pos(data):
    """
    Recovers root rotation (quat) and position from 4-dim root feature.
    Input: (B, T, D) tensor, where D >= 4
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    """
    Reconstructs 3D joint positions from the HumanML3D feature vector.
    This function specifically uses the RIC (Root Invariant Coordinates) data.
    Input: (B, T, 263) tensor
    Output: (B, T, 22, 3) tensor
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # Get the ric_data (local joint positions) from the feature vector
    # 4 (root) + (joints_num - 1) * 3
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3)) # (B, T, 21, 3)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2) # (B, T, 22, 3)

    return positions

# --- Animation Code (from previous script) ---

# Kinematic chain for the 22-joint skeleton (for plotting)
KINEMATIC_CHAIN = [
    (0, 1), (1, 4), (4, 7), (7, 10),  # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),  # Right Leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
    (9, 13), (13, 16), (16, 18), (18, 20), # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21)  # Right Arm
]

def setup_plot_limits(ax, motion_data):
    """Calculates stable axis limits to prevent camera wobble."""
    X = motion_data[..., 0]
    Y_up = motion_data[..., 1]
    Z_fwd = motion_data[..., 2]

    max_range = np.array([
        X.max() - X.min(),
        Y_up.max() - Y_up.min(),
        Z_fwd.max() - Z_fwd.min()
    ]).max() / 2.0
    
    # Handle the case of a single point or no motion
    if max_range < 0.1:
        max_range = 1.0 
    
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y_up.max() + Y_up.min()) * 0.5
    mid_z = (Z_fwd.max() + Z_fwd.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range) # Plot Z (fwd) on Y-axis
    ax.set_zlim(mid_y - max_range, mid_y + max_range) # Plot Y (up) on Z-axis

    ax.set_xlabel("X")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Up)")
    
    return (mid_x, mid_y, mid_z, max_range)

def load_and_reconstruct(args):
    """Loads features, un-normalizes, and reconstructs 3D motion."""
    print(f"Loading features from: {args.input}")
    try:
        features = np.load(args.input).astype(np.float32)
        mean = np.load(args.mean).astype(np.float32)
        std = np.load(args.std).astype(np.float32)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
        
    if features.shape[1] != 263 or mean.shape[0] != 263 or std.shape[0] != 263:
        print(f"Error: Feature/Mean/Std shape mismatch.")
        print(f"Features: {features.shape}, Mean: {mean.shape}, Std: {std.shape}")
        sys.exit(1)
        
    print("Un-normalizing features...")
    std[std < 1e-8] = 1e-8  # Avoid division by zero
    features_unnorm = features * std + mean
    
    # Pad features: recover_from_ric expects (T, D)
    # The feature vector is (T-1, D). We pad the first frame.
    print("Padding first frame for reconstruction...")
    features_padded = np.vstack([features_unnorm[0:1,:], features_unnorm])
    
    # Convert to tensor and add batch dim
    features_tensor = torch.from_numpy(features_padded).float().unsqueeze(0) # (1, T, 263)
    
    print("Reconstructing 3D motion...")
    # Reconstruct
    with torch.no_grad():
        reconstructed_xyz_tensor = recover_from_ric(features_tensor, joints_num=22)
    
    # Convert back to numpy
    motion_data = reconstructed_xyz_tensor.squeeze(0).cpu().numpy() # (T, 22, 3)
    print(f"Reconstruction complete. Motion shape: {motion_data.shape}")
    
    return motion_data

def main():
    parser = argparse.ArgumentParser(description="Reconstruct HumanML3D feature file (.npy) as a GIF.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input feature .npy file")
    parser.add_argument("--mean", type=str, required=True, help="Path to the Mean.npy file")
    parser.add_argument("--std", type=str, required=True, help="Path to the Std.npy file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .gif file")
    parser.add_argument("--downsample", type=int, default=2, help="Plot every Nth frame to speed up GIF (default: 2)")
    args = parser.parse_args()

    # 1. Load and reconstruct the data
    motion_data = load_and_reconstruct(args)
    
    # Downsample frames for faster GIF
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