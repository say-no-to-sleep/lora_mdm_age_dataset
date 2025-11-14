import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import f as scipy_f_dist
except ImportError:  # pragma: no cover - SciPy not always available
    scipy_f_dist = None

# ---------- CONFIG ----------
# Path to the directory that *contains* Comp_v6_KLD01
VC_ROOT = "."   # <-- change this
BASE = os.path.join(VC_ROOT, "Comp_v6_KLD01")
# Moving-average window (must be >= 1). Increase for a smoother curve.
SMOOTHING_WINDOW = 5
PIECEWISE_KNOT = 60.0
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 42
AGE_BINS = [
    ("20-39", 20.0, 40.0),
    ("40-59", 40.0, 60.0),
    ("60-79", 60.0, 80.0),
    ("80+", 80.0, np.inf),
]

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

def smooth_curve(values, window):
    """
    Simple moving-average smoothing that keeps the output length equal to input.
    """
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or len(values) <= 1:
        return values
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float32) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def run_linear_regression(ages, speeds):
    slope, intercept = np.polyfit(ages, speeds, 1)
    slope_decade = slope * 10.0
    print("\n[Linear regression]")
    print(f"  speed ≈ {intercept:.5f} + {slope:.6f} * age")
    print(f"  Δ speed per year:   {slope:.6f}")
    print(f"  Δ speed per decade: {slope_decade:.6f}")
    return slope, intercept


def run_piecewise_regression(ages, speeds, knot=PIECEWISE_KNOT):
    hinge = np.clip(ages - knot, a_min=0.0, a_max=None)
    X = np.column_stack([np.ones_like(ages), ages, hinge])
    coef, _, _, _ = np.linalg.lstsq(X, speeds, rcond=None)
    intercept, slope_base, slope_after_delta = coef
    slope_after = slope_base + slope_after_delta
    print(f"\n[Piecewise regression @ age {knot:g}]")
    print(f"  speed ≈ {intercept:.5f} + {slope_base:.6f} * age + {slope_after_delta:.6f} * max(0, age-{knot:g})")
    print(f"  slope before {knot:g}: {slope_base:.6f}")
    print(f"  slope after  {knot:g}: {slope_after:.6f}")
    return coef


def bootstrap_confidence_band(ages, speeds, grid_ages, window, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    if len(ages) < 2 or grid_ages.size == 0:
        return None

    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(samples):
        idx = rng.integers(0, len(ages), len(ages))
        ages_sample = ages[idx]
        speeds_sample = speeds[idx]

        age_speed_map = defaultdict(list)
        for age_val, speed_val in zip(ages_sample.tolist(), speeds_sample.tolist()):
            age_speed_map[float(age_val)].append(float(speed_val))

        sorted_ages = np.array(sorted(age_speed_map.keys()), dtype=np.float32)
        if sorted_ages.size < 2:
            continue
        avg_speed = np.array([np.mean(age_speed_map[a]) for a in sorted_ages], dtype=np.float32)
        smooth_speed = smooth_curve(avg_speed, window)
        curve = np.interp(
            grid_ages,
            sorted_ages,
            smooth_speed,
            left=smooth_speed[0],
            right=smooth_speed[-1],
        )
        curves.append(curve)

    if not curves:
        return None

    curves = np.stack(curves, axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    return lower, upper


def summarize_age_bins(ages, speeds, bins=AGE_BINS):
    summary = []
    groups = []
    for label, low, high in bins:
        if np.isfinite(high):
            mask = (ages >= low) & (ages < high)
        else:
            mask = ages >= low
        group = speeds[mask]
        count = int(group.size)
        mean = float(group.mean()) if count else float("nan")
        std = float(group.std(ddof=1)) if count > 1 else (0.0 if count == 1 else float("nan"))
        summary.append((label, count, mean, std))
        if count:
            groups.append(group)
    return summary, groups


def run_one_way_anova(groups):
    groups = [np.asarray(g, dtype=np.float32) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None

    n_total = sum(len(g) for g in groups)
    k = len(groups)
    if n_total <= k:
        return None

    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    if not np.isfinite(ms_within) or ms_within <= 0:
        return {
            "F": np.nan,
            "df_between": df_between,
            "df_within": df_within,
            "p_value": None,
        }

    f_stat = ms_between / ms_within
    if scipy_f_dist is not None and np.isfinite(f_stat):
        p_value = float(scipy_f_dist.sf(f_stat, df_between, df_within))
    else:
        p_value = None
    return {
        "F": float(f_stat),
        "df_between": df_between,
        "df_within": df_within,
        "p_value": p_value,
    }

def main():
    # ---- Load list of clips for this split ----
    names = load_split_list(BASE, SPLIT)
    print(f"Found {len(names)} clips in split '{SPLIT}'")

    ages = []
    speeds = []
    clip_names = []
    bad_ages = 0

    skipped_zero = 0

    for rel in names:
        base_name = os.path.basename(rel)
        if "_0_" in base_name or "SUBJ101_humanml3d_22joints" in base_name:
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
        clip_names.append(rel)

    if not ages:
        print("No valid clips found with known ages.")
        return
    
    ages = np.array(ages, dtype=np.float32)
    speeds = np.array(speeds, dtype=np.float32)

    print(f"Using {len(ages)} clips with valid ages. Skipped {bad_ages} clips with age < 0.")

    # ---- Basic stats ----
    print(f"Age:   min={ages.min():.1f}, max={ages.max():.1f}, mean={ages.mean():.1f}, std={ages.std():.1f}")
    print(f"Speed: min={speeds.min():.3f}, max={speeds.max():.3f}, mean={speeds.mean():.3f}, std={speeds.std():.3f}")

    run_linear_regression(ages, speeds)
    run_piecewise_regression(ages, speeds, PIECEWISE_KNOT)

    summary, anova_groups = summarize_age_bins(ages, speeds)
    print("\n[Age-bin summary]")
    for label, count, mean, std in summary:
        if count == 0:
            print(f"  {label:>6}: n=0")
        else:
            print(f"  {label:>6}: n={count:3d}, mean={mean:.4f}, std={std:.4f}")

    anova_result = run_one_way_anova(anova_groups)
    if anova_result:
        p_str = (
            f"{anova_result['p_value']:.4g}"
            if anova_result["p_value"] is not None
            else "SciPy not available"
        )
        print(
            f"\n[One-way ANOVA] F={anova_result['F']:.3f} "
            f"(df_between={anova_result['df_between']}, df_within={anova_result['df_within']}), "
            f"p={p_str}"
        )
    else:
        print("\n[One-way ANOVA] Not enough data per bin to compute statistics.")

    # ---- Scatter plot: age vs mean speed ----
    age_speed_map = defaultdict(list)
    for age_val, speed_val in zip(ages.tolist(), speeds.tolist()):
        age_speed_map[float(age_val)].append(float(speed_val))

    avg_age_sorted = np.array(sorted(age_speed_map.keys()), dtype=np.float32)
    avg_speed_sorted = np.array([np.mean(age_speed_map[a]) for a in avg_age_sorted], dtype=np.float32)
    smooth_avg_speed = smooth_curve(avg_speed_sorted, SMOOTHING_WINDOW)
    ci_bounds = bootstrap_confidence_band(
        ages,
        speeds,
        avg_age_sorted,
        SMOOTHING_WINDOW,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    fig, ax = plt.subplots(figsize = (7,5))
    scatter = ax.scatter(ages, speeds, alpha = 0.5, edgecolors = "none", label="Clips")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean planar speed |v_xz|")
    ax.set_title(f"Van Criekinge: Age vs Mean Speed ({SPLIT} split)")
    ax.set_ylim(bottom = 0)
    ax.grid(True)
    ax.plot(avg_age_sorted, avg_speed_sorted, color="0.7", linestyle="--", linewidth=1.5, marker="o", label="Average per age")
    ax.plot(avg_age_sorted, smooth_avg_speed, color="tab:red", linewidth=2.5, label=f"Smoothed (window={SMOOTHING_WINDOW})")
    if ci_bounds is not None:
        lower, upper = ci_bounds
        ax.fill_between(
            avg_age_sorted,
            lower,
            upper,
            color="tab:red",
            alpha=0.15,
            label="95% bootstrap CI",
        )
    ax.legend()
    fig.tight_layout()

    annot = ax.annotate(
        "",
        xy=(0,0),
        xytext=(10,10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    def update_annot(ind):
        idx = ind["ind"][0]
        pos = scatter.get_offsets()[idx]
        annot.xy = pos
        text = f"{clip_names[idx]}\nAge: {ages[idx]:.1f}\nSpeed: {speeds[idx]:.3f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.9)

    def on_hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)


    plt.show()

if __name__ == "__main__":
    main()
