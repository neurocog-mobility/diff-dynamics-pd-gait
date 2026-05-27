"""Analyze Stride: Phase-resolved coupling and Fiedler metric extraction.

Loads a single stride from the processed generic dataset, passes it through the
pre-trained Neural ODE to extract coupling strengths (K), computes the 
continuous phase-resolved effective coupling, and outputs the overall Fiedler 
value (algebraic connectivity, λ₂) for the stride.

Usage:
    uv run python scripts/analyze_stride.py --stride-index 0
"""

import argparse
from pathlib import Path
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from src.canonicalize import NEUROMOTOR_EDGE_NAMES, canonicalize
from src.graph_ode import GraphGaitODE, N_EDGES

OUT_DIR = Path("results/analysis")

JOINTS = [
    "Ref_Sho", "Contra_Sho", "Trunk", "Pelvis",
    "Ref_Hip", "Contra_Hip", "Ref_Knee", "Contra_Knee",
    "Ref_Ank", "Contra_Ank",
]

def build_graph(ec_dict):
    """Build NetworkX graph from effective coupling dictionary."""
    G = nx.Graph()
    G.add_nodes_from(JOINTS)
    for edge_name, w in ec_dict.items():
        u, v = edge_name.split("↔")
        if w > 0:
            G.add_edge(u, v, weight=w)
    return G

def compute_fiedler_value(ec_dict):
    """Compute the Fiedler value (λ₂) of the weighted graph."""
    G = build_graph(ec_dict)
    L = nx.laplacian_matrix(G, weight="weight").toarray()
    eigs = np.sort(np.real(np.linalg.eigvals(L)))
    return float(eigs[1]) if len(eigs) >= 2 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride-index", type=int, default=0, help="Index of stride to analyze")
    parser.add_argument("--checkpoint", type=str, default="results/model/checkpoints/model_sub01.pt", help="Path to checkpoint")
    parser.add_argument("--timepoints", type=int, default=25, help="Timepoints for ODE integration")
    args = parser.parse_args()

    # Load Data
    npz = np.load("data/processed/strides.npz", allow_pickle=False)
    
    total_strides = npz["data"].shape[0]
    if args.stride_index >= total_strides:
        print(f"Error: Stride index {args.stride_index} out of bounds (0 to {total_strides-1})")
        return
        
    data_full = npz["data"][args.stride_index:args.stride_index+1]
    side = npz["side"][args.stride_index:args.stride_index+1] if "side" in npz else np.zeros(1)
    
    data_full = canonicalize(data_full, side)

    # Downsample for model
    data_t = torch.from_numpy(data_full).permute(0, 2, 1).float()
    stride_25 = F.interpolate(data_t, size=args.timepoints, mode="linear", align_corners=True).permute(0, 2, 1).numpy()[0]

    # Load Model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    ode = GraphGaitODE()
    ode.load_state_dict(sd, strict=False)
    ode.eval()
    
    X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
    if isinstance(X_mean, torch.Tensor): 
        X_mean, X_std = X_mean.numpy(), X_std.numpy()

    X_norm = torch.from_numpy(((stride_25 - X_mean.squeeze()) / X_std.squeeze()).astype(np.float32)).unsqueeze(0)
    t_span = torch.linspace(0.0, 1.0, args.timepoints)

    with torch.no_grad():
        _, K = ode(X_norm, t_span)

    print(f"--- Coupling Analysis for Stride {args.stride_index} ---")
    
    K_vals = K[0].numpy()
    ec_dict = {NEUROMOTOR_EDGE_NAMES[i]: float(K_vals[i]) for i in range(N_EDGES)}
    
    print("\nStatic Edge Strengths (K):")
    for edge, val in ec_dict.items():
        print(f"  {edge:25s}: {val:.4f}")
        
    fiedler = compute_fiedler_value(ec_dict)
    print(f"\nFiedler Value (λ₂): {fiedler:.4f}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"stride_{args.stride_index}_coupling.csv"
    
    with open(out_file, "w") as f:
        f.write("edge,k\n")
        for edge, val in ec_dict.items():
            f.write(f"{edge},{val:.6f}\n")
            
    print(f"\nSaved coupling values to {out_file}")

if __name__ == "__main__":
    main()
