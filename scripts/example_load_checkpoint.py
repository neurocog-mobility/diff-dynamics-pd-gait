"""
Minimal Example: Loading a Checkpoint and Extracting Coupling

This script demonstrates how to:
1. Load a pre-trained LOSO (leave-one-subject-out) checkpoint.
2. Prepare a sample stride from the processed dataset.
3. Pass the stride through the model to obtain:
   - Stride-level coupling strengths (K)
   - Reconstructed continuous trajectories
   - Effective Locomotor Coupling (ELC) metrics over time

Usage:
    python scripts/example_load_checkpoint.py
"""

import numpy as np
import torch
import torchdiffeq
from pathlib import Path

from src.graph_ode import GraphGaitODE, N_JOINTS, N_ANAT_EDGES
from src.canonicalize import NEUROMOTOR_EDGE_NAMES

def main():
    # 1. Configuration
    checkpoint_path = Path("results/model/checkpoints/model_sub01.pt")
    data_path = Path("data/processed/strides.npz")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}.")
        print("Please run the training pipeline first or download the pretrained checkpoints.")
        return
    
    if not data_path.exists():
        print(f"Data not found at {data_path}.")
        print("Please run the preprocessing pipeline first.")
        return

    # 2. Load the model checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Checkpoints trained with DistributedDataParallel may have 'module.' prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    
    model = GraphGaitODE()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    X_mean = torch.tensor(ckpt["X_mean"], dtype=torch.float32)
    X_std = torch.tensor(ckpt["X_std"], dtype=torch.float32)

    # 3. Load a sample stride
    # For this example, we just take the first valid stride from the dataset
    npz = np.load(data_path, allow_pickle=False)
    data = npz["data"]
    
    # Take a single stride (shape: [timepoints=100, nodes=10])
    sample_stride_raw = data[0]
    
    # Z-score normalize the stride using training statistics
    X_mean_sq = X_mean.squeeze()
    X_std_sq = X_std.squeeze()
    sample_stride_norm = (torch.tensor(sample_stride_raw, dtype=torch.float32) - X_mean_sq) / X_std_sq
    
    # Add batch dimension: [B=1, T=101, N=10]
    X_input = sample_stride_norm.unsqueeze(0)

    # 4. Extract Stride-level Coupling (K)
    with torch.no_grad():
        K = model.encoder(X_input)  # Shape: [1, 15]
    
    print("\n--- Stride-level Coupling Strengths (K) ---")
    K_flat = K[0].numpy()
    for i, edge_name in enumerate(NEUROMOTOR_EDGE_NAMES):
        edge_type = "Anatomical" if i < N_ANAT_EDGES else "Cross-chain"
        print(f"{edge_name:<25} ({edge_type}): {K_flat[i]:.4f}")

    # 5. Integrate Trajectory
    # Set the estimated K into the ODE function
    model.ode_func.set_K(K)
    
    # Create the initial augmented state space
    d_state = model.d_state
    z0 = torch.zeros(1, N_JOINTS, d_state)
    z0[:, :, 0] = X_input[:, 0, :]  # Initialize with the physical joint angles at t=0
    z0_flat = z0.reshape(1, N_JOINTS * d_state)
    
    # Define integration time grid
    timepoints = X_input.shape[1]
    t_span = torch.linspace(0, 1, timepoints)

    with torch.no_grad():
        z_traj_flat = torchdiffeq.odeint(model.ode_func, z0_flat, t_span, method="midpoint")
        # Output shape: [T, B, N*d_state] -> [B, T, N, d_state]
        z_traj = z_traj_flat.permute(1, 0, 2).reshape(1, timepoints, N_JOINTS, d_state)

        # The reconstructed physical angles are the first dimension of the augmented state
        x_hat_norm = z_traj[:, :, :, 0]
        # Un-normalize back to physical units (degrees)
        x_hat = (x_hat_norm * X_std) + X_mean

    print("\n--- Trajectory Reconstruction ---")
    print(f"Original shape: {sample_stride_raw.shape}")
    print(f"Reconstructed shape: {x_hat[0].shape}")
    mse = torch.nn.functional.mse_loss(torch.tensor(sample_stride_raw, dtype=torch.float32), x_hat[0])
    print(f"Reconstruction MSE: {mse.item():.4f} degrees^2")
    
    print("\nExtraction complete! For continuous Effective Locomotor Coupling (ELC),")
    print("please see scripts/analysis/phase_coupling.py which extracts ||g(z(t))|| over the full cycle.")

if __name__ == "__main__":
    main()
