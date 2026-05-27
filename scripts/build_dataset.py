"""Generic Template for Building a Processed Stride Dataset.

This script demonstrates how to ingest your own kinematic data (e.g. from generic CSV files),
extract individual strides using heel-strike events, time-normalize them, canonicalize
the left/right limbs into reference/contralateral, and save the final `strides.npz`
expected by the Neural ODE model.

Expected CSV Format:
- Columns for the 10 joints (sagittal angles for limbs, transverse for trunk/pelvis):
  L_Sho, R_Sho, Trunk, Pelvis, L_Hip, R_Hip, L_Knee, R_Knee, L_Ank, R_Ank
- Boolean/binary columns for heel strikes:
  L_Heel_Strike, R_Heel_Strike
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


from src.canonicalize import canonicalize, CANONICAL_JOINT_NAMES

# The raw joint names expected in the CSV, in this exact order to match the canonicalizer.
JOINT_COLUMNS = [
    "L_Sho",
    "R_Sho",
    "Trunk",
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ank",
    "R_Ank",
]


def extract_strides_from_csv(csv_path: Path, stride_points: int = 101) -> list[dict]:
    """Extract time-normalized single strides from a single CSV file."""
    df = pd.read_csv(csv_path)

    # 1. Get raw kinematic data
    try:
        angles_array = df[JOINT_COLUMNS].values  # [T, 10]
    except KeyError as e:
        print(f"Missing columns in {csv_path}: {e}")
        return []

    # 2. Extract Heel Strike frames
    # Assumes columns 'L_Heel_Strike' and 'R_Heel_Strike' are 1 at the frame of contact.
    l_hs_frames = np.where(df.get("L_Heel_Strike", np.zeros(len(df))) == 1)[0]
    r_hs_frames = np.where(df.get("R_Heel_Strike", np.zeros(len(df))) == 1)[0]

    strides = []

    # Process both sides
    for side, hs_frames in [("L", l_hs_frames), ("R", r_hs_frames)]:
        sorted_frames = sorted(hs_frames)
        # Iterate over consecutive pairs of heel strikes
        for k in range(len(sorted_frames) - 1):
            f_start = sorted_frames[k]
            f_end = sorted_frames[k + 1]

            if f_end - f_start < 10:  # Skip impossibly short strides
                continue

            raw_segment = angles_array[f_start : f_end + 1, :].copy()

            # Interpolate any NaNs
            for j in range(raw_segment.shape[1]):
                col = raw_segment[:, j]
                nans = np.isnan(col)
                if nans.any() and not nans.all():
                    col[nans] = np.interp(
                        np.where(nans)[0], np.where(~nans)[0], col[~nans]
                    )
                    raw_segment[:, j] = col

            # Skip if still too many NaNs
            if np.isnan(raw_segment).sum() / raw_segment.size > 0.10:
                continue

            # Time-normalize to stride_points (0-100% of gait cycle)
            n_raw = raw_segment.shape[0]
            t_raw = np.linspace(0, 100, n_raw)
            t_norm = np.linspace(0, 100, stride_points)
            normalized = np.column_stack(
                [
                    np.interp(t_norm, t_raw, raw_segment[:, j_col])
                    for j_col in range(raw_segment.shape[1])
                ]
            )

            strides.append(
                {
                    "data": normalized,
                    "side": 1 if side == "R" else 0,  # 0=LHS, 1=RHS
                }
            )

    return strides


def main():
    print("Building generic stride dataset...")

    # Replace this with the path to your folder of generic CSVs
    csv_dir = Path("data/raw/csv_files")
    if not csv_dir.exists():
        print(
            f"Example directory {csv_dir} not found. Please create it and add your CSVs."
        )
        print("For now, creating an empty dummy strides.npz to demonstrate the format.")
        all_strides = []
    else:
        all_strides = []
        for csv_file in csv_dir.glob("*.csv"):
            all_strides.extend(extract_strides_from_csv(csv_file))

    # For demonstration, if no strides found, we create a dummy one
    if not all_strides:
        print("Injecting dummy stride data for demonstration purposes...")
        all_strides.append(
            {
                "data": np.zeros((101, 10), dtype=np.float32),
                "side": 0,
            }
        )

    # Stack lists into arrays
    n = len(all_strides)
    data = np.zeros((n, 101, 10), dtype=np.float32)
    sides = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(all_strides):
        data[i] = s["data"]
        sides[i] = s["side"]

    # ---------------------------------------------------------
    # anonicalize
    # This transforms the [L, R] data into [Reference, Contralateral]
    # ensuring the striking limb is always in the first columns.
    # ---------------------------------------------------------
    data_canonical = canonicalize(data, sides)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strides_custom.npz"

    np.savez_compressed(
        out_path,
        data=data_canonical,
        side=sides,
        joint_names=np.array(CANONICAL_JOINT_NAMES),
    )

    print(f"Saved {n} strides to {out_path}")
    print(f"Shape: {data_canonical.shape} (N x Time x Joints)")


if __name__ == "__main__":
    main()
