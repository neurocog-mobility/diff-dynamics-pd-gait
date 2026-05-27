"""Figure — Model Walks: Body-graph film strip + trajectory validation.

Usage:
    uv run python scripts/fig_model_walks.py --stride-index 0 --checkpoint results/model/checkpoints/model_sub01.pt
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.canonicalize import CANONICAL_JOINT_NAMES, canonicalize
from src.graph_ode import GraphGaitODE, N_JOINTS

OUT_DIR = Path("results/figures")

# ── Colour palette ───────────────────────────────────────────────────────
REF_NODE_COLOR = "#c0392b"
CONTRA_NODE_COLOR = "#2980b9"
PHASE_PCT = [0, 20, 40, 60, 80]

PANEL_B_JOINTS = ["Ref_Hip", "Ref_Knee", "Ref_Ank", "Ref_Sho"]
PANEL_B_TITLES = [
    "Reference hip",
    "Reference knee",
    "Reference ankle",
    "Reference shoulder",
]

SEG_TRUNK, SEG_SHOULDER_HALF, SEG_SHOULDER_RAISE = 0.65, 0.60, 0.25
SEG_HIP_DROP, SEG_HIP_HALF, SEG_THIGH, SEG_SHANK = 0.25, 0.42, 1.0, 1.0
SEG_ARM, SEG_FOOT = 0.65, 0.22

_J = {n: i for i, n in enumerate(CANONICAL_JOINT_NAMES)}


def forward_kinematics(angles, pelvis_tilt=13.0):
    a = np.array(angles, dtype=float)
    hip_r, knee_r, ank_r = a[_J["Ref_Hip"]], a[_J["Ref_Knee"]], a[_J["Ref_Ank"]]
    hip_c, knee_c, ank_c = (
        a[_J["Contra_Hip"]],
        a[_J["Contra_Knee"]],
        a[_J["Contra_Ank"]],
    )
    sho_r, sho_c, trunk_ang = a[_J["Ref_Sho"]], a[_J["Contra_Sho"]], a[_J["Trunk"]]

    px, py = 0.0, 0.0
    trunk_rad = np.radians(90 + trunk_ang * 0.3)
    tx, ty = px + SEG_TRUNK * np.cos(trunk_rad), py + SEG_TRUNK * np.sin(trunk_rad)
    perp = trunk_rad + np.pi / 2
    sho_cx, sho_cy = tx + SEG_SHOULDER_RAISE * np.cos(
        trunk_rad
    ), ty + SEG_SHOULDER_RAISE * np.sin(trunk_rad)
    ref_sx, ref_sy = sho_cx + SEG_SHOULDER_HALF * np.cos(
        perp
    ), sho_cy + SEG_SHOULDER_HALF * np.sin(perp)
    con_sx, con_sy = sho_cx - SEG_SHOULDER_HALF * np.cos(
        perp
    ), sho_cy - SEG_SHOULDER_HALF * np.sin(perp)

    def _leg(hip_ang, knee_ang, ank_ang, sign=1.0):
        h_rad = np.radians(-pelvis_tilt + hip_ang - 90)
        hx, hy = px + sign * SEG_HIP_HALF, py - SEG_HIP_DROP
        kx, ky = hx + SEG_THIGH * np.cos(h_rad), hy + SEG_THIGH * np.sin(h_rad)
        k_rad = np.radians(-pelvis_tilt + hip_ang - knee_ang - 90)
        ax, ay = kx + SEG_SHANK * np.cos(k_rad), ky + SEG_SHANK * np.sin(k_rad)
        f_rad = np.radians(-pelvis_tilt + hip_ang - knee_ang + ank_ang)
        fx, fy = ax + SEG_FOOT * np.cos(f_rad), ay + SEG_FOOT * np.sin(f_rad)
        return (hx, hy), (kx, ky), (ax, ay), (fx, fy)

    def _arm(sho_ang, sho_x, sho_y):
        arm_rad = np.radians(-sho_ang - 90)
        return (sho_x + SEG_ARM * np.cos(arm_rad), sho_y + SEG_ARM * np.sin(arm_rad))

    r_hip, r_knee, r_ank, r_foot = _leg(hip_r, knee_r, ank_r, sign=-1.0)
    c_hip, c_knee, c_ank, c_foot = _leg(hip_c, knee_c, ank_c, sign=+1.0)

    return {
        "Ref_Sho": (ref_sx, ref_sy),
        "Contra_Sho": (con_sx, con_sy),
        "Ref_Elbow": _arm(sho_r, ref_sx, ref_sy),
        "Contra_Elbow": _arm(sho_c, con_sx, con_sy),
        "Trunk": (tx, ty),
        "Pelvis": (px, py),
        "Ref_Hip": r_hip,
        "Contra_Hip": c_hip,
        "Ref_Knee": r_knee,
        "Contra_Knee": c_knee,
        "Ref_Ank": r_ank,
        "Contra_Ank": c_ank,
        "Ref_Foot": r_foot,
        "Contra_Foot": c_foot,
    }


_SKELETON_SEGMENTS = [
    ("Pelvis", "Trunk", "axial"),
    ("Ref_Sho", "Contra_Sho", "axial"),
    ("Trunk", "Ref_Sho", "ref"),
    ("Trunk", "Contra_Sho", "contra"),
    ("Ref_Sho", "Ref_Elbow", "ref"),
    ("Contra_Sho", "Contra_Elbow", "contra"),
    ("Ref_Hip", "Contra_Hip", "axial"),
    ("Pelvis", "Ref_Hip", "ref"),
    ("Ref_Hip", "Ref_Knee", "ref"),
    ("Ref_Knee", "Ref_Ank", "ref"),
    ("Ref_Ank", "Ref_Foot", "ref"),
    ("Pelvis", "Contra_Hip", "contra"),
    ("Contra_Hip", "Contra_Knee", "contra"),
    ("Contra_Knee", "Contra_Ank", "contra"),
    ("Contra_Ank", "Contra_Foot", "contra"),
]


def _panel_filmstrip(ax, truth_angles):
    T = truth_angles.shape[0]
    frame_pcts = np.linspace(0, 100, 50, endpoint=True)
    _KEY_PHASES = {20 * i: f"{20 * i}%" for i in range(6)}
    key_frame_map = {
        frame_pcts[np.argmin(np.abs(frame_pcts - k))]: k for k in _KEY_PHASES
    }

    for pct in frame_pcts:
        t_idx = int(round(pct / 100 * (T - 1)))
        pos = forward_kinematics(truth_angles[t_idx])
        matched = key_frame_map.get(pct, None)

        lw = 4.0 if matched is not None else 1.0
        al = 1.0 if matched is not None else 0.35 * np.cos(pct / 6.36) ** 4 + 0.05

        for ja, jb, side in _SKELETON_SEGMENTS:
            col = {"ref": REF_NODE_COLOR, "contra": CONTRA_NODE_COLOR, "axial": "0.15"}[
                side
            ]
            ax.plot(
                [pct + pos[ja][0] * 3.0, pct + pos[jb][0] * 3.0],
                [pos[ja][1], pos[jb][1]],
                color=col,
                lw=lw,
                alpha=al,
                solid_capstyle="round",
            )

        if matched is not None:
            for jname, (jx, jy) in pos.items():
                c = (
                    REF_NODE_COLOR
                    if "Ref_" in jname
                    else (CONTRA_NODE_COLOR if "Contra_" in jname else "0.15")
                )
                ax.plot(
                    pct + jx * 3.0,
                    jy,
                    "o",
                    color=c,
                    markersize=4,
                    markeredgecolor="k",
                    markeredgewidth=0.8,
                )
            ax.text(
                pct,
                1.05,
                _KEY_PHASES[matched],
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="0.25",
            )

    ax.set_ylabel("Model\nreconstructed\nkinematics")
    ax.legend(
        handles=[
            Patch(facecolor=REF_NODE_COLOR, edgecolor="none", label="Reference limb"),
            Patch(
                facecolor=CONTRA_NODE_COLOR,
                edgecolor="none",
                label="Contralateral limb",
            ),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.3),
        ncols=2,
    )
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_yticks([])
    ax.set_xlim(-10, 110)
    ax.set_ylim(-2.3, 1.5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def _panel_trajectories(
    axes, truth, predicted, pred_no_anat, pred_no_skip, joint_indices
):
    gait_pct = np.linspace(0, 100, truth.shape[0])
    lines = [
        ("0.10", 3.0, "solid", 1.00),
        ("0.10", 2.5, "--", 1.00),
        ("#e07b54", 1.4, (0, (4, 2)), 0.70),
        ("#27ae60", 1.4, (0, (1, 2)), 0.70),
    ]
    for i, (ax, j_idx, title) in enumerate(zip(axes, joint_indices, PANEL_B_TITLES)):
        y_true, y_pred = truth[:, j_idx], predicted[:, j_idx]
        r2 = 1 - np.nansum((y_true - y_pred) ** 2) / max(
            np.nansum((y_true - np.nanmean(y_true)) ** 2), 1e-10
        )

        for pct in PHASE_PCT:
            ax.axvline(pct, color="0.75", ls=":", lw=0.8)

        for (col, lw, ls, al), y in zip(
            lines, [y_true, y_pred, pred_no_anat[:, j_idx], pred_no_skip[:, j_idx]]
        ):
            ax.plot(gait_pct, y, color=col, lw=lw, ls=ls, alpha=al)
            ax.plot(
                [gait_pct[0], gait_pct[-1]], [y[0], y[-1]], "o", color=col, alpha=al
            )

        ax.set_xlim(-5, 105)
        ax.set_ylabel("Angle (°)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"{title},  R² = {r2:.3f}", loc="left", fontsize=10, fontstyle="italic"
        )

        if i < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Gait cycle (%)", fontsize=11)
            ax.set_xticks(PHASE_PCT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride-index", type=int, default=0, help="Index of stride to analyze"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/model/checkpoints/model_sub01.pt",
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    npz = np.load("data/processed/strides.npz", allow_pickle=False)

    total_strides = npz["data"].shape[0]
    if args.stride_index >= total_strides:
        print(
            f"Error: Stride index {args.stride_index} out of bounds (0 to {total_strides-1})"
        )
        return

    data_full = npz["data"][args.stride_index : args.stride_index + 1]
    side = (
        npz["side"][args.stride_index : args.stride_index + 1]
        if "side" in npz
        else np.zeros(1)
    )

    data_full = canonicalize(data_full, side)
    stride_full = data_full[0]

    # Downsample for model
    data_t = torch.from_numpy(data_full).permute(0, 2, 1).float()
    stride_25 = (
        F.interpolate(data_t, size=25, mode="linear", align_corners=True)
        .permute(0, 2, 1)
        .numpy()[0]
    )

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model = GraphGaitODE()
    model.load_state_dict(sd, strict=False)
    model.eval()

    X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
    if isinstance(X_mean, torch.Tensor):
        X_mean, X_std = X_mean.numpy(), X_std.numpy()

    X_norm = torch.from_numpy(
        ((stride_25 - X_mean.squeeze()) / X_std.squeeze()).astype(np.float32)
    ).unsqueeze(0)

    def _predict(ab_anat=False, ab_skip=False):
        with torch.no_grad():
            z, _ = model(
                X_norm,
                torch.linspace(0, 1, 25),
                ablate_anat=ab_anat,
                ablate_skip=ab_skip,
            )
        return z.numpy()[0] * X_std.squeeze() + X_mean.squeeze()

    pred = _predict()
    pred_no_anat = _predict(ab_anat=True)
    pred_no_skip = _predict(ab_skip=True)

    for jname in ("Ref_Sho", "Contra_Sho"):
        idx = _J[jname]
        stride_full[:, idx] *= -1
        stride_25[:, idx] *= -1
        pred[:, idx] *= -1
        pred_no_anat[:, idx] *= -1
        pred_no_skip[:, idx] *= -1

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        5,
        1,
        height_ratios=[1.5, 1, 1, 1, 1],
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.10,
        hspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0])
    axes_b = [fig.add_subplot(gs[i + 1], sharex=ax_a) for i in range(4)]

    _panel_filmstrip(ax_a, stride_full)
    _panel_trajectories(
        axes_b,
        stride_25,
        pred,
        pred_no_anat,
        pred_no_skip,
        [CANONICAL_JOINT_NAMES.index(j) for j in PANEL_B_JOINTS],
    )

    fig.legend(
        handles=[
            Line2D([0], [0], color="0.10", lw=3.0, label="Ground truth"),
            Line2D([0], [0], color="0.10", lw=2.5, ls="--", label="Full model"),
            Line2D(
                [0],
                [0],
                color="#e07b54",
                lw=1.4,
                ls=(0, (4, 2)),
                alpha=0.70,
                label="Anatomical edges suppressed",
            ),
            Line2D(
                [0],
                [0],
                color="#27ae60",
                lw=1.4,
                ls=(0, (1, 2)),
                alpha=0.70,
                label="Cross-chain edges suppressed",
            ),
        ],
        frameon=False,
        loc="lower right",
        ncols=4,
        bbox_to_anchor=(0.98, 0.0),
    )
    fig.align_ylabels()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"model_walks_stride{args.stride_index}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
