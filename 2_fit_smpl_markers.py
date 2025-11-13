#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit SMPL to Van Criekinge C3D markers (no MoSh).
- Uses smplx SMPL layer
- Per-subject betas (shared), per-frame pose+trans
- Robust marker subset, small temporal smoothness
- Writes *_smpl_params.npz + fit report
"""
import os, json, math, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ------------------------- Marker subset & joint map -------------------------
# SMPL 24-joint order used here:
SMPL_JOINTS = [
    "pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2","L_ankle","R_ankle",
    "spine3","L_foot","R_foot","neck","L_collar","R_collar","head",
    "L_shoulder","R_shoulder","L_elbow","R_elbow","L_wrist","R_wrist","L_hand","R_hand"
]
J = {n:i for i,n in enumerate(SMPL_JOINTS)}

# Subset of anatomical markers found in your metadata (robust across subjects)
MARKER_TO_JOINT = {
    # Head / upper trunk
    "LFHD":J["head"], "RFHD":J["head"], "LBHD":J["head"], "RBHD":J["head"],
    "C7":J["neck"], "T10":J["spine2"], "CLAV":J["spine3"], "STRN":J["spine3"],

    # Pelvis triad
    "LASI":J["pelvis"], "RASI":J["pelvis"], "SACR":J["pelvis"],

    # Shoulders / elbows / wrists
    "LSHO":J["L_shoulder"], "RSHO":J["R_shoulder"],
    "LELB":J["L_elbow"], "RELB":J["R_elbow"],
    "LWRA":J["L_wrist"], "LWRB":J["L_wrist"],
    "RWRA":J["R_wrist"], "RWRB":J["R_wrist"],

    # Lower limbs
    "LKNE":J["L_knee"], "RKNE":J["R_knee"],
    "LANK":J["L_ankle"], "RANK":J["R_ankle"],

    "LHEE":J["L_ankle"],  "RHEE":J["R_ankle"],
    "LTOE":J["L_foot"],   "RTOE":J["R_foot"],
}

# ------------------------------ Utilities -----------------------------------
def rotmat_to_axis_angle(R):
    # R: (3,3)
    eps = 1e-8
    cos = (np.trace(R) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz], dtype=np.float64)
    axis /= (2.0*np.sin(angle) + eps)
    return (axis * angle).astype(np.float32)

def estimate_root_init(markers_np, used_names):
    """
    markers_np: (T,K,3) subset already (no NaNs after prep_markers)
    used_names: list[str] names aligned with K
    Returns: axis-angle (3,), transl_init (T,3)
    """
    name2i = {n.upper(): i for i, n in enumerate(used_names)}
    need = ["LASI","RASI","SACR","C7","CLAV","STRN"]
    if not all(n in name2i for n in need):
        # Fallback
        aa = np.zeros(3, dtype=np.float32)
        t0 = np.zeros((markers_np.shape[0],3), dtype=np.float32)
        return aa, t0

    LASI = markers_np[:, name2i["LASI"], :]
    RASI = markers_np[:, name2i["RASI"], :]
    SACR = markers_np[:, name2i["SACR"], :]
    C7   = markers_np[:, name2i["C7"],   :]
    CLAV = markers_np[:, name2i["CLAV"], :]
    STRN = markers_np[:, name2i["STRN"], :]

    pelvis = (LASI + RASI + SACR) / 3.0    # (T,3)
    upper  = (C7 + CLAV + STRN) / 3.0      # (T,3)

    # Pick first frame
    p0 = pelvis[0]; u0 = upper[0]
    up = u0 - p0; up /= (np.linalg.norm(up) + 1e-8)
    lr = LASI[0] - RASI[0]; lr /= (np.linalg.norm(lr) + 1e-8)  # left minus right
    fwd = np.cross(up, lr); fwd /= (np.linalg.norm(fwd) + 1e-8)
    lr  = np.cross(fwd, up); lr  /= (np.linalg.norm(lr) + 1e-8)

    # Columns: X=lr, Y=up, Z=fwd → world
    Rw = np.stack([lr, up, fwd], axis=1)          # 3x3
    aa = rotmat_to_axis_angle(Rw)                 # (3,)
    return aa.astype(np.float32), pelvis.astype(np.float32)

def huber(x, delta=0.01):
    a = x.abs()
    mask = (a <= delta).float()
    return 0.5*(a**2)*mask + delta*(a-0.5*delta)* (1.0-mask)

def linear_interp_nans(arr):  # (T,3)
    x = arr.copy()
    T = x.shape[0]
    for d in range(3):
        v = x[:,d]
        nans = np.isnan(v)
        if nans.any():
            idx = np.arange(T)
            v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
        x[:,d] = v
    return x

def decimate_indices(T, src_fps, dst_fps):
    if src_fps == dst_fps: return np.arange(T)
    step = src_fps / dst_fps
    return np.clip(np.round(np.arange(0, T, step)).astype(int), 0, T-1)

def sanitize(name: str):
    return name.replace(" ", "_").replace("(", "").replace(")", "")

# ------------------------------ Fitter --------------------------------------
class SMPLFitter(nn.Module):
    def __init__(self, smpl_layer, T, device="cpu", init_betas=None):
        super().__init__()
        self.smpl = smpl_layer
        self.device = device
        # Parameters
        self.global_orient = nn.Parameter(torch.zeros(T, 3, device=device))
        self.transl = nn.Parameter(torch.zeros(T, 3, device=device))
        self.body_pose = nn.Parameter(torch.zeros(T, 69, device=device))   # axis-angle 23*3
        betas_init = torch.zeros(10, device=device) if init_betas is None else torch.as_tensor(init_betas, device=device).float()
        self.betas = nn.Parameter(betas_init)

    def forward(self):
        # Returns joints (T,24,3)
        out = self.smpl(
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            betas=self.betas.unsqueeze(0).expand(self.global_orient.shape[0], -1),
            transl=self.transl,
            pose2rot=True
        )
        return out.joints  # (T, 24, 3)

def build_smpl(model_dir: str, gender: str, device: str):
    import smplx
    # smplx.create expects directory containing SMPL_*.pkl
    smpl = smplx.create(model_path=model_dir, model_type="smpl", gender=gender,
                        use_pca=False, num_betas=10).to(device)
    smpl.eval()
    for p in smpl.parameters(): p.requires_grad = False
    return smpl

def pick_gender(subject_meta):
    g = subject_meta.get("gender","neutral").lower()
    if g in ("male","female"): return g
    return "neutral"

def load_markers_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    M = d["marker_data"]        # (T, n_markers, 3) in meters
    names = [str(x) for x in d["marker_names"].tolist()]
    fps = float(d["frame_rate"])
    return M, names, fps

def subset_markers(M, names):
    keep = []
    targets = []
    used = []
    for i, nm in enumerate(names):
        key = nm.strip().upper()
        if key in MARKER_TO_JOINT:
            keep.append(i)
            targets.append(MARKER_TO_JOINT[key])
            used.append(key)
    if len(keep) < 12:
        raise RuntimeError("Too few anatomical markers found after subsetting.")
    return M[:, keep, :], used, np.array(targets, dtype=np.int64)

def prep_markers(M_sub, max_gap=4, drop_thresh=0.6):
    # Interp small gaps, drop frames with > (1-drop_thresh) missing
    T, K, _ = M_sub.shape
    valid = ~np.isnan(M_sub).any(axis=2)
    keep_frames = (valid.sum(axis=1) >= math.ceil(drop_thresh*K))
    M2 = M_sub[keep_frames].copy()
    # interpolate independently per marker
    for k in range(M2.shape[1]):
        M2[:,k,:] = linear_interp_nans(M2[:,k,:])
    return M2, keep_frames

def subject_cache_paths(out_root, subject_id):
    subj_dir = Path(out_root) / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir, subj_dir / "betas.npy"

def fit_sequence(smpl, markers_torch, marker_joint_idx, used_names, init_betas, device, iters=400,
                 w_marker=1.0, w_pose=1e-3, w_betas=5e-4, w_vel=1e-2, w_acc=1e-3):
    T, K, _ = markers_torch.shape
    fitter = SMPLFitter(smpl, T, device, init_betas=init_betas).to(device)

    mk_np = markers_torch.detach().cpu().numpy()     # (T,K,3)
    aa, pelvis_series = estimate_root_init(mk_np, used_names)  # (3,), (T,3)
    fitter.global_orient.data[:] = torch.from_numpy(np.repeat(aa[None, :], T, axis=0)).to(device)
    fitter.transl.data[:]        = torch.from_numpy(pelvis_series).to(device)

    opt = Adam([
        {"params":[fitter.global_orient, fitter.body_pose, fitter.transl], "lr":3e-2},
        {"params":[fitter.betas], "lr":1e-3}
    ])
    hub = nn.SmoothL1Loss(beta=0.01, reduction='none')  # Huber-like

    targets_idx = torch.as_tensor(marker_joint_idx, device=device, dtype=torch.long)  # (K,)
    for it in range(iters):
        opt.zero_grad()
        joints = fitter()                         # (T,24,3)
        # gather joint positions for each marker
        pick = joints[:, targets_idx, :]          # (T,K,3)
        names = np.array(used_names)                          # list[str] length K
        w = np.ones(len(names), dtype=np.float32)
        leg_keys = {"LKNE","RKNE","LANK","RANK","LHEE","RHEE","LTOE","RTOE"}
        w[[i for i, n in enumerate(names) if n in leg_keys]] = 1.5
        w_t = torch.from_numpy(w).to(device)[None, :, None]   # (1,K,1)

        per_elem = hub(pick, markers_torch)                   # (T,K,3), reduction='none'
        marker_loss = (per_elem * w_t).mean()

        # priors & smoothness
        pose_l2 = (fitter.body_pose**2).mean()
        betas_l2 = (fitter.betas**2).mean()
        vel = (fitter.transl[1:] - fitter.transl[:-1]).pow(2).mean()
        acc = (fitter.transl[2:] - 2*fitter.transl[1:-1] + fitter.transl[:-2]).pow(2).mean() if T>2 else torch.tensor(0., device=device)

        loss = w_marker*marker_loss + w_pose*pose_l2 + w_betas*betas_l2 + w_vel*vel + w_acc*acc
        loss.backward()
        opt.step()

        if (it+1) % 100 == 0 or it==0:
            print(f"  iter {it+1:03d}: total={loss.item():.6f}  marker={marker_loss.item():.6f}")

    with torch.no_grad():
        joints = fitter()
    result = {
        "poses": fitter.body_pose.detach().cpu().numpy(),         # (T,63)
        "global_orient": fitter.global_orient.detach().cpu().numpy(),  # (T,3)
        "trans": fitter.transl.detach().cpu().numpy(),            # (T,3)
        "betas": fitter.betas.detach().cpu().numpy(),             # (10,)
        "joints": joints.detach().cpu().numpy(),                  # (T,24,3)
        "final_marker_loss": float(marker_loss.detach().cpu().item())
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True, help=".../data/processed/van_criekinge")
    ap.add_argument("--models_dir", required=True, help=".../body_models/smpl (contains SMPL_*.pkl)")
    ap.add_argument("--out_dir", required=True, help="where to write smpl_fitted_smpl")
    ap.add_argument("--subject", default=None, help="e.g., SUBJ01 (default: all subjects under processed_dir)")
    ap.add_argument("--trial", default=None, help="e.g., 'SUBJ1 (0)' (optional: fit only this trial)")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = [args.subject] if args.subject else sorted([p.name for p in processed.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for subj in subjects:
        subj_dir = processed / subj
        npz_files = sorted(glob.glob(str(subj_dir / "*_markers_positions.npz")))
        if not npz_files:
            print(f"[{subj}] no *_markers_positions.npz files — skipping")
            continue
        # cache paths
        subj_out_dir, betas_path = subject_cache_paths(out_root, subj)
        # gender read from any metadata file
        meta_files = sorted(glob.glob(str(subj_dir / "*_metadata.json")))
        gender = "neutral"
        subj_meta = {}
        if meta_files:
            with open(meta_files[0], "r") as f:
                subj_meta = json.load(f)
            gender = pick_gender(subj_meta)
        print(f"[{subj}] gender={gender}")

        # build SMPL for gender (or neutral fallback)
        device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
        smpl = build_smpl(args.models_dir, gender, device)

        # Load/initialize betas
        init_betas = None
        if betas_path.exists():
            init_betas = np.load(betas_path)
            print(f"[{subj}] loaded cached betas {betas_path}")
        else:
            print(f"[{subj}] no cached betas; will optimize and save")

        # Optionally filter to one trial
        if args.trial:
            npz_files = [p for p in npz_files if Path(p).stem.startswith(args.trial)]
            if not npz_files:
                print(f"[{subj}] requested trial {args.trial} not found")
                continue

        # If no betas, fit them once on the first trial (or only trial)
        if init_betas is None:
            first_npz = npz_files[0]
            M, names, fps = load_markers_npz(first_npz)
            M_sub, used_names, tgt_idx = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub, max_gap=4, drop_thresh=0.6)
            T = M_sub.shape[0]
            if T > 220:   # limit for fast betas fit
                sel = decimate_indices(T, src_fps=fps, dst_fps=min(fps, 20))
                M_sub = M_sub[sel]
                T = M_sub.shape[0]
            markers_torch = torch.from_numpy(M_sub).float().to(device)
            result = fit_sequence(smpl, markers_torch, tgt_idx, used_names, init_betas=None, device=device, iters=min(args.iters*2, 800))
            np.save(betas_path, result["betas"])
            init_betas = result["betas"]
            print(f"[{subj}] saved betas → {betas_path}")

        # Fit each trial with cached betas
        for npz_path in npz_files:
            trial_name = Path(npz_path).stem.replace("_markers_positions","")
            trial_safe = sanitize(trial_name)
            out_trial = subj_out_dir / f"{trial_safe}_smpl_params.npz"
            report_path = subj_out_dir / f"{trial_safe}_smpl_metadata.json"

            if out_trial.exists():
                print(f"[{subj}] {trial_name}: already fitted → {out_trial.name}")
                continue

            M, names, fps = load_markers_npz(npz_path)
            M_sub, used_names, tgt_idx = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub, max_gap=4, drop_thresh=0.6)
            markers_torch = torch.from_numpy(M_sub).float().to(device)

            print(f"[{subj}] fitting {trial_name}: frames={M_sub.shape[0]} fps={fps} markers={len(used_names)}")
            result = fit_sequence(smpl, markers_torch, tgt_idx, used_names, init_betas=init_betas, device=device, iters=args.iters)

            # Pack SMPL params (HumanML3D expects body_pose+global_orient concatenated to 72D)
            poses72 = np.concatenate([result["global_orient"], result["poses"]], axis=1)  # (T,72)
            np.savez(out_trial,
                     poses=poses72, trans=result["trans"], betas=result["betas"],
                     gender=gender, subject_id=subj, trial_name=trial_name, fps=fps,
                     n_frames=result["trans"].shape[0], joints=result["joints"])

            report = {
                "subject_id": subj, "trial_name": trial_name, "gender": gender,
                "frames_kept": int(markers_torch.shape[0]), "fps": fps,
                "markers_used": used_names,
                "final_marker_loss": result["final_marker_loss"]
            }
            with open(report_path, "w") as f: json.dump(report, f, indent=2)
            print(f"[{subj}] wrote {out_trial.name} and {report_path.name}")

if __name__ == "__main__":
    main()
