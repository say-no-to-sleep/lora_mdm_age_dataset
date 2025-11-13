import argparse
import numpy as np
import c3d
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Import kinematic chain from paramUtil (make sure it's accessible)
try:
    # Assuming paramUtil.py is in a 'utils' subdirectory relative to this script
    from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
except ImportError:
    # Fallback if structure is different - you might need to adjust this path
    print("Warning: Could not import kinematic_chain from utils.paramUtil. Using default.")
    # Define t2m_kinematic_chain directly if needed, copy from paramUtil.py
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def load_c3d_motion(c3d_path):
    """Loads C3D, converts to meters, returns T x N x 3 """
    positions_list = []
    try:
        with open(c3d_path, 'rb') as handle:
            reader = c3d.Reader(handle)
            fps = reader.header.frame_rate
            for i, points, analog in reader.read_frames():
                positions_list.append(points[:, :3]) # Take only XYZ
        positions_raw = np.array(positions_list, dtype=np.float32)
        # Handle potential NaNs/Infs
        positions_raw[np.isnan(positions_raw)] = 0.0
        positions_raw[np.isinf(positions_raw)] = 0.0
        # Convert mm to meters
        positions_m = positions_raw / 1000.0
        print(f"Loaded C3D: {positions_m.shape}, FPS: {fps}")
        return positions_m, fps
    except Exception as e:
        print(f"Error loading C3D {c3d_path}: {e}")
        return None, None

def load_reconstructed_motion(npz_path):
    """Loads reconstructed joints from NPZ"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        joints = data['joints'].astype(np.float32) # Shape: (T, 22, 3)
        fps = float(data.get('fps', 20.0))
        print(f"Loaded Reconstructed NPZ: {joints.shape}, FPS: {fps}")
        return joints, fps
    except Exception as e:
        print(f"Error loading reconstructed NPZ {npz_path}: {e}")
        return None, None

def plot_comparison_frame(ax, joints_rec, title, frame_idx):
    """Plots a single frame for comparison"""
    ax.clear()
    ax.view_init(elev=120, azim=-90)
    ax.dist = 7.5

    # Plot Reconstructed Motion (using kinematic chain)
    joints = joints_rec[frame_idx] # (22, 3)
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"] # Orange theme
    for i, (chain) in enumerate(kinematic_chain):
         if i < 5:
             linewidth = 4.0
             ax.plot(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=linewidth, color=colors[i%len(colors)])
         # else: # Don't plot hand chains if not needed
         #     linewidth = 2.0
         #     ax.plot(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=linewidth, color=colors[i%len(colors)])

    # Calculate dynamic plot limits based on this frame
    # (Could also calculate once based on whole sequence for stability)
    min_vals = joints.min(axis=0)
    max_vals = joints.max(axis=0)
    center = (min_vals + max_vals) / 2
    radius = np.max(max_vals - min_vals) * 0.6 # Make radius slightly larger
    radius = max(radius, 0.5) # Minimum radius

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title + f"\nFrame {frame_idx}")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def main_visualize():
    parser = argparse.ArgumentParser(description="Visualize original C3D vs reconstructed motion.")
    parser.add_argument("--c3d_path", type=str, required=True, help="Path to the original C3D file.")
    parser.add_argument("--rec_npz_path", type=str, required=True, help="Path to the reconstructed joints NPZ file.")
    parser.add_argument("--output_gif", type=str, required=True, help="Path to save the comparison GIF.")
    parser.add_argument("--num_frames", type=int, default=None, help="Limit number of frames to render (optional).")
    parser.add_argument("--gif_fps", type=int, default=20, help="FPS for the output GIF.")
    args = parser.parse_args()

    # --- Load Data ---
    motion_c3d_markers, fps_c3d = load_c3d_motion(args.c3d_path)
    motion_rec_joints, fps_rec = load_reconstructed_motion(args.rec_npz_path)

    if motion_c3d_markers is None or motion_rec_joints is None:
        print("Failed to load motion data. Exiting.")
        return

    # --- Prepare for Animation ---
    num_markers = motion_c3d_markers.shape[1]
    T_c3d = motion_c3d_markers.shape[0]
    T_rec = motion_rec_joints.shape[0]

    # Use the minimum length if different, or limit by args.num_frames
    max_T = min(T_c3d, T_rec)
    if args.num_frames is not None:
        max_T = min(max_T, args.num_frames)

    # Note: C3D has markers, Rec has joints. Visualizing markers directly is messy.
    # We will show the RECONSTRUCTED joints only for now.
    # To show C3D, you'd typically plot raw markers or fit SMPL to C3D markers first.
    print(f"Rendering {max_T} frames for comparison (Reconstructed Joints only).")

    # --- Create Animation ---
    fig = plt.figure(figsize=(8, 8)) # Single plot for reconstructed
    ax_rec = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        plot_comparison_frame(ax_rec, motion_rec_joints, "Reconstructed Motion", frame_idx)
        # Add frame text
        fig.suptitle(f"Frame {frame_idx}/{max_T-1}", y=0.05)


    ani = FuncAnimation(fig, update, frames=range(max_T), interval=1000/args.gif_fps, blit=False)

    # --- Save Animation ---
    print(f"Saving comparison GIF to: {args.output_gif}")
    os.makedirs(os.path.dirname(args.output_gif), exist_ok=True)
    writer = PillowWriter(fps=args.gif_fps)
    ani.save(args.output_gif, writer=writer)
    plt.close(fig) # Close the plot figure
    print("Done.")

if __name__ == "__main__":
    main_visualize()