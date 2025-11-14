import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
# Path to the directory that *contains* Comp_v6_KLD01
VC_ROOT = "."   # <-- change this
BASE = os.path.join(VC_ROOT, "Comp_v6_KLD01")

# Choose which split to inspect first ("train" is usually best)
SPLIT = "train"
# ----------------------------

def load_split_list(base, split):
    split_txt = os.path.join(base, f"{split}.txt")
    if not os.path.exists(split_txt):
        raise FileNotFoundError(f"Split file not found: {split_txt}")
    with open(split_txt) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

def load_motion_and_age(base, split, rel_name):
    """
    rel_name looks like: 'SUBJ03/SUBJ3_0_humanml3d_22joints'
    We expect:
      motions/<rel_name>.npy
      ages/<rel_name>.txt
    """
    motion_path = os.path.join(base, split, "motions", rel_name + ".npy")
    age_path    = os.path.join(base, split, "ages",    rel_name + ".txt")

    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Missing motion file: {motion_path}")
    if not os.path.exists(age_path):
        raise FileNotFoundError(f"Missing age file: {age_path}")

    motion = np.load(motion_path)         # shape [T, 263]
    with open(age_path) as f:
        age_str = f.read().strip()
    try:
        age = float(age_str)
    except ValueError:
        raise ValueError(f"Could not parse age '{age_str}' in {age_path}")

    return motion, age

def compute_clip_speed(motion):
    """
    motion: [T, 263] feature sequence, *unnormalized* (as saved by build_vc_dataset)
    Feature index 1: root vx (ground plane X)
    Feature index 2: root vz (ground plane Z)
    We compute mean |velocity| over time.
    """
    if motion.ndim != 2 or motion.shape[1] < 3:
        raise ValueError(f"Unexpected motion shape {motion.shape}, expected [T, 263]")

    vx = motion[:, 1]
    vz = motion[:, 2]
    speed = np.sqrt(vx ** 2 + vz ** 2)   # [T]
    return float(speed.mean())

def main():
    # ---- Load list of clips for this split ----
    names = load_split_list(BASE, SPLIT)
    print(f"Found {len(names)} clips in split '{SPLIT}'")

    ages = []
    speeds = []
    bad_ages = 0

    skipped_zero = 0

    for rel in names:
        base_name = os.path.basename(rel)
        if "_0_" in base_name:
            skipped_zero += 1
            continue
        try:
            motion, age = load_motion_and_age(BASE, SPLIT, rel)
        except Exception as e:
            print(f"[WARN] Skipping {rel}: {e}")
            continue

        # Ignore unknown ages (e.g. -1)
        if age < 0:
            bad_ages += 1
            continue

        speed = compute_clip_speed(motion)

        ages.append(age)
        speeds.append(speed)

    ages = np.array(ages, dtype=np.float32)
    speeds = np.array(speeds, dtype=np.float32)

    print(f"Using {len(ages)} clips with valid ages. Skipped {bad_ages} clips with age < 0.")

    # ---- Basic stats ----
    print(f"Age:   min={ages.min():.1f}, max={ages.max():.1f}, mean={ages.mean():.1f}, std={ages.std():.1f}")
    print(f"Speed: min={speeds.min():.3f}, max={speeds.max():.3f}, mean={speeds.mean():.3f}, std={speeds.std():.3f}")

    # ---- Scatter plot: age vs mean speed ----
    plt.figure(figsize=(7, 5))
    plt.scatter(ages, speeds, alpha=0.5, edgecolors="none")
    plt.xlabel("Age (years)")
    plt.ylabel("Mean planar speed (|v_xz|)")
    plt.title(f"Van Criekinge: Age vs Mean Walking Speed ({SPLIT} split)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()