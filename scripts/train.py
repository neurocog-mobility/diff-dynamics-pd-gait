"""Model Training: Neuromotor Graph (15 Edges) Neural ODE with Periodicity Loss.

Trains the model on a generic stride dataset using 5-fold cross-validation.
Outputs to results/model/.

Usage:
    uv run python scripts/train.py --fold-limit 1  # smoke test
    uv run python scripts/train.py                 # full run
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("high")

from src.canonicalize import CANONICAL_JOINT_NAMES, canonicalize
from src.graph_ode import GraphGaitODE, N_EDGES, N_JOINTS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OUT_DIR = Path("results/model")
SEED = 42
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 128
LAMBDA_SPARSE = 1e-3
LAMBDA_PERIODIC = 0.1
LOG_EVERY = 1

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(timepoints: int = 25):
    """Load and downsample generic stride data. Returns: X [N, T, 10]"""
    npz = np.load("data/processed/strides.npz", allow_pickle=False)
    data = npz["data"]

    # Canonicalize if 'side' is present
    if "side" in npz:
        data = canonicalize(data, npz["side"].astype(int))

    if timepoints != 101:
        data_t = torch.from_numpy(data).permute(0, 2, 1).float()
        data_t = F.interpolate(
            data_t, size=timepoints, mode="linear", align_corners=True
        )
        data = data_t.permute(0, 2, 1).numpy()

    clean = np.isfinite(data).all(axis=(1, 2))
    n_dropped = (~clean).sum()
    if n_dropped:
        print(f"  [data] Dropping {n_dropped} strides with NaN/Inf")

    return data[clean]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphGaitODE_Train(GraphGaitODE):
    """GraphGaitODE that exposes z0 and z_final for periodicity loss."""

    def forward(self, stride, t_span=None, ablate_coupling=False):
        B, T, N = stride.shape
        ds = self.d_state

        K = self.encoder(stride)
        if ablate_coupling:
            K = torch.zeros_like(K)

        z0 = torch.zeros(B, N, ds, device=stride.device, dtype=stride.dtype)
        z0[:, :, 0] = stride[:, 0, :]
        z0_flat = z0.reshape(B, N * ds)

        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, T, device=stride.device)

        self.ode_func.set_K(K)
        odeint_fn = (
            torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint
        )
        z_pred_flat = odeint_fn(self.ode_func, z0_flat, t_span, method=self.solver)

        T_out = z_pred_flat.shape[0]
        z_pred = z_pred_flat.view(T_out, B, N, ds)[:, :, :, 0]

        # Store for periodicity loss (detach-free — stays in graph)
        self._z0_flat = z0_flat
        self._z_final_flat = z_pred_flat[-1]

        return z_pred.permute(1, 0, 2), K

    def periodicity_loss(self):
        """||z(T) - z(0)||² averaged over batch and state dims."""
        return ((self._z_final_flat - self._z0_flat) ** 2).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(
    X_tr,
    X_val,
    device,
    fold_label,
    args,
):
    """Train one fold with periodicity loss."""
    torch.manual_seed(SEED)

    X_mean = X_tr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    X_std = (X_tr.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    X_normed = (X_tr - X_mean) / X_std
    ws_var = X_normed.var(axis=1).mean(axis=0)
    jw = 1.0 / (ws_var + 1e-8)
    jw = np.clip(jw, None, 5.0 * np.median(jw))
    jw *= len(jw) / jw.sum()
    joint_weights = torch.from_numpy(jw.astype(np.float32)).to(device)

    def norm(arr):
        return torch.from_numpy(((arr - X_mean) / X_std).astype(np.float32))

    Xt = norm(X_tr).to(device)
    Xv = norm(X_val).to(device)

    T_obs = Xt.shape[1]
    integ_mult = getattr(args, "integration_mult", 4)
    T_dense = (T_obs - 1) * integ_mult + 1
    t_span_dense = torch.linspace(0.0, 1.0, T_dense, device=device)
    obs_indices = torch.arange(0, T_dense, integ_mult, device=device)

    solver_choice = getattr(args, "solver", "midpoint")
    model = GraphGaitODE_Train(solver=solver_choice).to(device)
    model.use_adjoint = args.use_adjoint

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=1e-5
    )
    ds = TensorDataset(Xt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = args.patience
    lam_p = args.lambda_periodic

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        n_batches = len(loader)

        for b_idx, (Xb,) in enumerate(loader, 1):
            z_pred_dense, K = model(Xb, t_span_dense)
            z_pred = z_pred_dense[:, obs_indices, :]

            recon_loss = ((z_pred - Xb) ** 2 * joint_weights).mean()
            sparse_loss = K[:, 11:].abs().mean()
            periodic_loss = model.periodicity_loss()

            loss = recon_loss + LAMBDA_SPARSE * sparse_loss + lam_p * periodic_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_losses.append(recon_loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            z_val_dense, K_val = model(Xv, t_span_dense)
            z_val = z_val_dense[:, obs_indices, :]
            val_loss = ((z_val - Xv) ** 2 * joint_weights).mean().item()

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1

        if epoch == 1 or epoch % args.log_every == 0 or improved or patience == 0:
            mark = "*" if improved else " "
            print(
                f"\r  {fold_label} ep{epoch:03d} "
                f"tr_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} "
                f"pat={patience:02d}{mark:<20}",
                end="",
                flush=True,
            )

        scheduler.step()
        if patience == 0:
            break

    print()
    model.load_state_dict(best_state)
    model.to(device)

    # R²
    model.eval()
    with torch.no_grad():
        z_val_dense, _ = model(Xv, t_span_dense)
        z_val = z_val_dense[:, obs_indices, :]
        ss_res = ((z_val - Xv) ** 2).sum().item()
        ss_tot = ((Xv - Xv.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot

    fold_log = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "val_r2": r2,
    }
    return model, X_mean, X_std, fold_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--use-adjoint", action="store_true")
    parser.add_argument("--integration-mult", type=int, default=1)
    parser.add_argument("--solver", type=str, default="midpoint")
    parser.add_argument("--timepoints", type=int, default=25)
    parser.add_argument("--lambda-periodic", type=float, default=LAMBDA_PERIODIC)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("=" * 60)
    print("Graph-Constrained Neural ODE — Neuromotor Graph (15 Edges)")
    print("=" * 60)
    print(f"Device: {device}")

    X = load_data(args.timepoints)
    print(f"Total Strides: {len(X)}")

    # Save config
    config = {
        "version": "neuromotor_graph",
        "lambda_periodic": args.lambda_periodic,
        "d_embed": 16,
        "d_state": 8,
        "d_hidden_self": 64,
        "d_hidden_couple": 64,
        "d_hidden_encoder": 32,
        "solver": args.solver,
        "timepoints": args.timepoints,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "lambda_sparse": LAMBDA_SPARSE,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "max_epochs": args.max_epochs,
        "seed": SEED,
    }
    with open(OUT_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- 5-Fold CV ---
    indices = np.random.permutation(len(X))
    fold_size = len(X) // 5

    val_r2s = []

    # We allow running just 1 fold for smoke testing
    n_folds = min(5, args.fold_limit)

    for fold_i in range(n_folds):
        fold_label = f"Fold {fold_i+1}/{n_folds}"
        print(f"\n{fold_label}")

        val_idx = indices[fold_i * fold_size : (fold_i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        model, X_mean, X_std, fold_log = train_fold(
            X[train_idx],
            X[val_idx],
            device,
            fold_label,
            args,
        )

        ckpt_dir = OUT_DIR / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "X_mean": X_mean,
                "X_std": X_std,
            },
            ckpt_dir / f"model_fold{fold_i+1}.pt",
        )

        val_r2s.append(fold_log["val_r2"])

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"Mean Val R²: {np.mean(val_r2s):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
