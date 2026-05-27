"""Canonical (Reference/Contralateral) joint representation for stride-level data.

10-node graph with midline joints (Trunk, Pelvis).

Canonicalizes strides so that the reference limb (the striking limb) is always
in the same column position, regardless of whether the stride is left or right.

LHS strides (side=0): L = Reference, R = Contra  → no permutation needed
RHS strides (side=1): R = Reference, L = Contra  → swap lateral pairs

Midline joints (Trunk, Pelvis) are invariant under Ref/Contra swaps.

Canonical node ordering (0–9):
  0=Ref_Sho, 1=Contra_Sho, 2=Trunk, 3=Pelvis,
  4=Ref_Hip, 5=Contra_Hip, 6=Ref_Knee, 7=Contra_Knee,
  8=Ref_Ank, 9=Contra_Ank
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_JOINT_NAMES = [
    "Ref_Sho", "Contra_Sho",
    "Trunk", "Pelvis",
    "Ref_Hip", "Contra_Hip",
    "Ref_Knee", "Contra_Knee",
    "Ref_Ank", "Contra_Ank",
]

# 11 edges in canonical space
CANONICAL_EDGE_LIST = [
    ("Ref_Sho",    "Contra_Sho"),   # 0:  bilateral shoulder
    ("Ref_Sho",    "Trunk"),        # 1:  ipsi shoulder-trunk
    ("Contra_Sho", "Trunk"),        # 2:  contra shoulder-trunk
    ("Trunk",      "Pelvis"),       # 3:  axial coupling
    ("Pelvis",     "Ref_Hip"),      # 4:  ipsi pelvis-hip
    ("Pelvis",     "Contra_Hip"),   # 5:  contra pelvis-hip
    ("Ref_Hip",    "Contra_Hip"),   # 6:  bilateral hip
    ("Ref_Hip",    "Ref_Knee"),     # 7:  ipsi hip-knee ref
    ("Ref_Knee",   "Ref_Ank"),      # 8:  ipsi knee-ankle ref
    ("Contra_Hip", "Contra_Knee"),  # 9:  ipsi hip-knee contra
    ("Contra_Knee","Contra_Ank"),   # 10: ipsi knee-ankle contra
]

CANONICAL_EDGE_NAMES = [f"{a}↔{b}" for a, b in CANONICAL_EDGE_LIST]

# 15 edges in canonical space (11 anatomical + 4 neuromotor skip links)
NEUROMOTOR_EDGE_LIST = CANONICAL_EDGE_LIST + [
    ("Ref_Knee",   "Contra_Knee"),  # 11: bilateral knee
    ("Ref_Ank",    "Contra_Ank"),   # 12: bilateral ankle
    ("Ref_Sho",    "Contra_Hip"),   # 13: diagonal synergy (posterior oblique sling)
    ("Contra_Sho", "Ref_Hip"),      # 14: diagonal synergy contra
]

NEUROMOTOR_EDGE_NAMES = [f"{a}↔{b}" for a, b in NEUROMOTOR_EDGE_LIST]

# Functional groups in canonical space (7 groups)
CANONICAL_FUNCTIONAL_GROUPS: dict[str, list[int]] = {
    "bilateral_shoulder":  [0],
    "shoulder_trunk":      [1, 2],
    "axial":               [3],
    "pelvis_hip":          [4, 5],
    "bilateral_hip":       [6],
    "ipsi_hip_knee":       [7, 9],
    "ipsi_knee_ankle":     [8, 10],
}

GROUP_NAMES = list(CANONICAL_FUNCTIONAL_GROUPS.keys())

# Edge index → functional group name
EDGE_TO_GROUP_NAME: dict[int, str] = {}
for _gname, _eidxs in CANONICAL_FUNCTIONAL_GROUPS.items():
    for _eidx in _eidxs:
        EDGE_TO_GROUP_NAME[_eidx] = _gname

# Column permutation for RHS strides: swap L↔R lateral pairs, keep midline
# Raw order:   L_Sho(0), R_Sho(1), Trunk(2), Pelvis(3),
#              L_Hip(4), R_Hip(5), L_Knee(6), R_Knee(7),
#              L_Ank(8), R_Ank(9)
# RHS: R=Ref → col 1→Ref_Sho, col 0→Contra_Sho, 2→Trunk, 3→Pelvis,
#              col 5→Ref_Hip, col 4→Contra_Hip, etc.
_RHS_PERM = [1, 0, 2, 3, 5, 4, 7, 6, 9, 8]


def canonicalize(data: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Permute joint columns for RHS strides so reference limb is always first.

    Parameters
    ----------
    data : [N, T, 10] float32 — stride kinematics, joints in JOINT_NAMES order
    side : [N] int — 0 = LHS stride, 1 = RHS stride

    Returns
    -------
    [N, T, 10] float32 — joints in CANONICAL_JOINT_NAMES order
    """
    out = data.copy()
    rhs = side == 1
    if rhs.any():
        out[rhs] = data[rhs][:, :, _RHS_PERM]
    return out
