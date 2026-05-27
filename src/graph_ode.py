"""Graph-Constrained Neural ODE for gait coordination (Neuromotor Graph).

Implements a coupled dynamical system on the extended body graph (15 edges):

    dz_i/dt = f_self(z_i, τ_i)
              + Σ_{j∈anat(i)} K_anat_ij · g_anat(z_i, z_j, τ_i, τ_j)
              + Σ_{j∈skip(i)} K_skip_ij · g_skip(z_i, z_j, τ_i, τ_j)

Key architectural decisions:
    - **Separate coupling functions**: g_anat and g_skip are independent MLPs.
      This prevents the optimizer from routing anatomical coupling through
      skip edges (or vice versa), ensuring each pathway learns its own
      coupling dynamics.
    - **Asymmetric K constraints**: Anatomical K has a nonzero floor (ε=0.05)
      because physical connections always exist. Skip K can reach zero.
    - **Asymmetric sparsity** (in training script): L1 penalty on skip K only.

Components:
    f_self   — intrinsic joint dynamics (small MLP, joint-type conditioned)
    g_anat   — anatomical coupling function (11 edges, kinetic chain)
    g_skip   — neuromotor coupling function (4 skip edges, cross-body)
    K        — per-stride coupling strengths predicted by StrideEncoder

Each joint's state is augmented from ℝ¹ (observed angle) to ℝ^d_state.
The extra latent dimensions give the autonomous ODE the phase-space richness
needed to produce multi-peaked waveforms (e.g. ankle dorsi/plantarflexion).
Only the first state dimension is observed; the rest are latent.

Trained via self-supervised trajectory reconstruction. K is the primary
scientific output: coupling strength per edge on the body graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq

from src.canonicalize import NEUROMOTOR_EDGE_LIST, CANONICAL_JOINT_NAMES

# ---------------------------------------------------------------------------
# Graph topology as index pairs
# ---------------------------------------------------------------------------
_NAME_TO_IDX = {name: i for i, name in enumerate(CANONICAL_JOINT_NAMES)}
EDGE_INDICES: list[tuple[int, int]] = [
    (_NAME_TO_IDX[a], _NAME_TO_IDX[b]) for a, b in NEUROMOTOR_EDGE_LIST
]

N_JOINTS = len(CANONICAL_JOINT_NAMES)  # 10
N_EDGES = len(NEUROMOTOR_EDGE_LIST)    # 15
N_ANAT_EDGES = 11                      # first 11 = anatomical kinetic chain
N_SKIP_EDGES = N_EDGES - N_ANAT_EDGES  # last 4 = neuromotor skip edges

ANAT_EDGE_INDICES = EDGE_INDICES[:N_ANAT_EDGES]
SKIP_EDGE_INDICES = EDGE_INDICES[N_ANAT_EDGES:]

# K floor for anatomical edges (physical connections always exist)
K_ANAT_FLOOR = 0.05


# ---------------------------------------------------------------------------
# ODE dynamics
# ---------------------------------------------------------------------------
class CouplingODEFunc(nn.Module):
    """Autonomous coupled-oscillator dynamics on the body graph.

    Uses **separate coupling functions** for anatomical and skip edges:
        - g_anat: learned biomechanical coupling (11 edges, kinetic chain)
        - g_skip: learned neuromotor coupling (4 edges, cross-body)

    This separation prevents the optimizer from using skip edges as
    shortcuts for anatomical coupling, ensuring each pathway captures
    its own dynamics.

    Time *t* is received from the ODE solver but intentionally ignored —
    the dynamics depend only on the current state z and the coupling
    parameters K.
    """

    def __init__(
        self,
        f_self_z: nn.Module,
        f_self_e: nn.Module,
        f_self_out: nn.Module,
        g_anat_z: nn.Module,
        g_anat_e: nn.Module,
        g_anat_out: nn.Module,
        g_skip_z: nn.Module,
        g_skip_e: nn.Module,
        g_skip_out: nn.Module,
        joint_embed: nn.Embedding,
        d_state: int = 2,
    ):
        super().__init__()
        self.f_self_z = f_self_z
        self.f_self_e = f_self_e
        self.f_self_out = f_self_out

        # Anatomical coupling pathway
        self.g_anat_z = g_anat_z
        self.g_anat_e = g_anat_e
        self.g_anat_out = g_anat_out

        # Skip coupling pathway (separate network)
        self.g_skip_z = g_skip_z
        self.g_skip_e = g_skip_e
        self.g_skip_out = g_skip_out

        self.joint_embed = joint_embed
        self.d_state = d_state
        self._K: torch.Tensor | None = None

        # Per-joint learnable input gain (log-space for positivity).
        self.log_gain = nn.Parameter(torch.zeros(N_JOINTS, 1))  # [N, 1]

        # Precomputed static embedding projections
        self._f_embed_proj: torch.Tensor | None = None
        self._g_anat_embed_proj: torch.Tensor | None = None
        self._g_skip_embed_proj: torch.Tensor | None = None

        # --- Anatomical edge incidence matrices ---
        E_a, N = N_ANAT_EDGES, N_JOINTS
        D_dst_a = torch.zeros(E_a, N)
        D_src_a = torch.zeros(E_a, N)
        for e, (s, d) in enumerate(ANAT_EDGE_INDICES):
            D_dst_a[e, d] = 1.0
            D_src_a[e, s] = 1.0
        self.register_buffer("D_dst_anat", D_dst_a)
        self.register_buffer("D_src_anat", D_src_a)
        self.register_buffer(
            "src_nodes_anat",
            torch.tensor([s for s, _ in ANAT_EDGE_INDICES], dtype=torch.long),
        )
        self.register_buffer(
            "dst_nodes_anat",
            torch.tensor([d for _, d in ANAT_EDGE_INDICES], dtype=torch.long),
        )

        # --- Skip edge incidence matrices ---
        E_s = N_SKIP_EDGES
        D_dst_s = torch.zeros(E_s, N)
        D_src_s = torch.zeros(E_s, N)
        for e, (s, d) in enumerate(SKIP_EDGE_INDICES):
            D_dst_s[e, d] = 1.0
            D_src_s[e, s] = 1.0
        self.register_buffer("D_dst_skip", D_dst_s)
        self.register_buffer("D_src_skip", D_src_s)
        self.register_buffer(
            "src_nodes_skip",
            torch.tensor([s for s, _ in SKIP_EDGE_INDICES], dtype=torch.long),
        )
        self.register_buffer(
            "dst_nodes_skip",
            torch.tensor([d for _, d in SKIP_EDGE_INDICES], dtype=torch.long),
        )

    def set_K(self, K: torch.Tensor) -> None:
        """Store per-batch coupling and precompute static embedding projections."""
        self._K = K  # [B, N_EDGES]

        embeds = self.joint_embed.weight  # [N_JOINTS, d_embed]

        # Precompute intrinsic embedding projection -> [N_JOINTS, d_hidden_self]
        self._f_embed_proj = self.f_self_e(embeds)

        # Precompute anatomical coupling embedding -> [2*E_a, d_hidden_couple]
        tau_src_a = embeds[self.src_nodes_anat]
        tau_dst_a = embeds[self.dst_nodes_anat]
        tau_i_a = torch.cat([tau_src_a, tau_dst_a], dim=0)
        tau_j_a = torch.cat([tau_dst_a, tau_src_a], dim=0)
        tau_ij_a = torch.cat([tau_i_a, tau_j_a], dim=-1)
        self._g_anat_embed_proj = self.g_anat_e(tau_ij_a)

        # Precompute skip coupling embedding -> [2*E_s, d_hidden_couple]
        tau_src_s = embeds[self.src_nodes_skip]
        tau_dst_s = embeds[self.dst_nodes_skip]
        tau_i_s = torch.cat([tau_src_s, tau_dst_s], dim=0)
        tau_j_s = torch.cat([tau_dst_s, tau_src_s], dim=0)
        tau_ij_s = torch.cat([tau_i_s, tau_j_s], dim=-1)
        self._g_skip_embed_proj = self.g_skip_e(tau_ij_s)

    def _compute_coupling(
        self,
        z_g: torch.Tensor,
        K_sub: torch.Tensor,
        g_z: nn.Module,
        g_out: nn.Module,
        embed_proj: torch.Tensor,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        D_dst: torch.Tensor,
        D_src: torch.Tensor,
    ) -> torch.Tensor:
        """Compute coupling contribution for a subset of edges.

        Parameters
        ----------
        z_g : [B, N, d_state] — gain-scaled joint states
        K_sub : [B, E_sub] — coupling strengths for this edge subset
        g_z, g_out : coupling MLP components
        embed_proj : [2*E_sub, d_hidden] — precomputed embedding projection
        src_nodes, dst_nodes : [E_sub] — node indices for this edge subset
        D_dst, D_src : [E_sub, N] — incidence matrices for accumulation

        Returns
        -------
        coupling : [B, N, d_state] — accumulated coupling per node
        """
        E = D_dst.shape[0]

        # Gather edge endpoints
        z_src = z_g[:, src_nodes, :]  # [B, E, d_state]
        z_dst = z_g[:, dst_nodes, :]  # [B, E, d_state]

        # Both directions: [B, 2E, 2*d_state]
        z_i = torch.cat([z_src, z_dst], dim=1)
        z_j = torch.cat([z_dst, z_src], dim=1)
        z_ij = torch.cat([z_i, z_j], dim=-1)

        # State projection + embedding projection
        g_z_out = g_z(z_ij)
        g_in = g_z_out + embed_proj.unsqueeze(0)
        g_result = g_out(g_in)  # [B, 2E, d_state]

        g_sd = g_result[:, :E, :]  # src->dst
        g_ds = g_result[:, E:, :]  # dst->src

        # Weight by K and accumulate onto nodes
        weighted_sd = K_sub.unsqueeze(-1) * g_sd
        weighted_ds = K_sub.unsqueeze(-1) * g_ds
        coupling = (
            torch.einsum("bed,en->bnd", weighted_sd, D_dst)
            + torch.einsum("bed,en->bnd", weighted_ds, D_src)
        )
        return coupling

    def forward(self, t: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt.  z_flat: [B, N_JOINTS * d_state]."""
        B = z_flat.shape[0]
        N = N_JOINTS
        ds = self.d_state
        K = self._K  # [B, N_EDGES]

        # Reshape flat state -> [B, N, d_state]
        z = z_flat.view(B, N, ds)

        # Apply learned per-joint input gain
        gain = self.log_gain.exp()  # [N, 1]
        z_g = z * gain.unsqueeze(0)  # [B, N, d_state]

        # --- Intrinsic dynamics (vectorised over joints) ---
        f_z = self.f_self_z(z_g)
        f_in = f_z + self._f_embed_proj.unsqueeze(0)
        dzdt = self.f_self_out(f_in)  # [B, N, d_state]

        # --- Anatomical coupling (11 edges, g_anat) ---
        coupling_anat = self._compute_coupling(
            z_g, K[:, :N_ANAT_EDGES],
            self.g_anat_z, self.g_anat_out, self._g_anat_embed_proj,
            self.src_nodes_anat, self.dst_nodes_anat,
            self.D_dst_anat, self.D_src_anat,
        )

        # --- Skip coupling (4 edges, g_skip) ---
        coupling_skip = self._compute_coupling(
            z_g, K[:, N_ANAT_EDGES:],
            self.g_skip_z, self.g_skip_out, self._g_skip_embed_proj,
            self.src_nodes_skip, self.dst_nodes_skip,
            self.D_dst_skip, self.D_src_skip,
        )

        return (dzdt + coupling_anat + coupling_skip).reshape(B, N * ds)


# ---------------------------------------------------------------------------
# Stride encoder  (stride kinematics → K)
# ---------------------------------------------------------------------------
class StrideEncoder(nn.Module):
    """Map a raw stride [B, T, N_JOINTS] → coupling strengths K [B, N_EDGES].

    Uses per-joint summary statistics (mean, std) as input features.
    K is constrained positive via softplus, with an additive floor on
    anatomical edges to prevent collapse.
    """

    def __init__(self, n_joints: int = N_JOINTS, n_edges: int = N_EDGES,
                 d_hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_joints * 2, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, n_edges),
        )

    def forward(self, stride: torch.Tensor) -> torch.Tensor:
        """stride: [B, T, N_JOINTS] → K: [B, N_EDGES] (positive)."""
        mean = stride.mean(dim=1)   # [B, N]
        std  = stride.std(dim=1)    # [B, N]
        feat = torch.cat([mean, std], dim=-1)  # [B, 2N]
        raw = self.mlp(feat)
        K_anat = F.softplus(raw[:, :N_ANAT_EDGES]) + K_ANAT_FLOOR
        K_skip = F.softplus(raw[:, N_ANAT_EDGES:])
        return torch.cat([K_anat, K_skip], dim=-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GraphGaitODE(nn.Module):
    """Graph-constrained Neural ODE for whole-body gait coordination.

    Extends the anatomical graph with 4 cross-body neuromotor skip edges
    and uses **separate coupling functions** for anatomical vs skip pathways.

    Parameters
    ----------
    d_embed : int
        Dimension of joint-type embeddings.
    d_state : int
        State dimensions per joint.
    d_hidden_self : int
        Hidden width of f_self MLP.
    d_hidden_couple : int
        Hidden width of g_anat and g_skip coupling MLPs.
    d_hidden_encoder : int
        Hidden width of K encoder MLP.
    solver : str
        ODE solver method (default 'midpoint').
    """

    def __init__(
        self,
        d_embed: int = 16,
        d_state: int = 8,
        d_hidden_self: int = 64,
        d_hidden_couple: int = 64,
        d_hidden_encoder: int = 32,
        solver: str = "midpoint",
    ):
        super().__init__()
        self.d_state = d_state

        # Joint-type embeddings (10 canonical types, shared across all pathways)
        self.joint_embed = nn.Embedding(N_JOINTS, d_embed)

        # --- Intrinsic dynamics (f_self) ---
        self.f_self_z = nn.Linear(d_state, d_hidden_self, bias=False)
        self.f_self_e = nn.Linear(d_embed, d_hidden_self, bias=True)
        self.f_self_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden_self, d_hidden_self),
            nn.SiLU(),
            nn.Linear(d_hidden_self, d_state),
        )

        # --- Anatomical coupling dynamics (g_anat, 11 edges) ---
        self.g_anat_z = nn.Linear(2 * d_state, d_hidden_couple, bias=False)
        self.g_anat_e = nn.Linear(2 * d_embed, d_hidden_couple, bias=True)
        self.g_anat_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden_couple, d_hidden_couple),
            nn.SiLU(),
            nn.Linear(d_hidden_couple, d_state),
        )

        # --- Neuromotor skip coupling dynamics (g_skip, 4 edges) ---
        self.g_skip_z = nn.Linear(2 * d_state, d_hidden_couple, bias=False)
        self.g_skip_e = nn.Linear(2 * d_embed, d_hidden_couple, bias=True)
        self.g_skip_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden_couple, d_hidden_couple),
            nn.SiLU(),
            nn.Linear(d_hidden_couple, d_state),
        )

        # ODE dynamics wrapper
        self.ode_func = CouplingODEFunc(
            self.f_self_z, self.f_self_e, self.f_self_out,
            self.g_anat_z, self.g_anat_e, self.g_anat_out,
            self.g_skip_z, self.g_skip_e, self.g_skip_out,
            self.joint_embed, d_state=d_state,
        )

        # Stride → K encoder (single encoder, asymmetric activation in forward)
        self.encoder = StrideEncoder(
            n_joints=N_JOINTS, n_edges=N_EDGES, d_hidden=d_hidden_encoder,
        )
        # Start with near-zero coupling (K ≈ softplus(-5) ≈ 0.007).
        # Without this, random-init K ≈ 0.7/edge can cause the ODE to diverge.
        nn.init.constant_(self.encoder.mlp[-1].bias, -5.0)

        self.solver = solver
        self.use_adjoint = False

    def forward(
        self,
        stride: torch.Tensor,
        t_span: torch.Tensor | None = None,
        ablate_coupling: bool = False,
        ablate_anat: bool = False,
        ablate_skip: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode K, integrate ODE, return prediction + K.

        Parameters
        ----------
        stride : [B, T, N_JOINTS]
            Normalised stride kinematics (observed angles only).
        t_span : [T] or None
            Integration time points. Default: linspace(0, 1, T).
        ablate_coupling : bool
            If True, set K=0 (ablates all coupling).
        ablate_anat : bool
            If True, zero out anatomical edge K (first 11 edges).
        ablate_skip : bool
            If True, zero out skip/cross-chain edge K (last 4 edges).

        Returns
        -------
        z_pred : [B, T, N_JOINTS]
            Predicted trajectory (observed dimension only).
        K : [B, N_EDGES]
            Coupling strengths.
        """
        B, T, N = stride.shape
        ds = self.d_state

        # Encode coupling strengths from observed angles
        K = self.encoder(stride)  # [B, N_EDGES]
        if ablate_coupling:
            K = torch.zeros_like(K)
        if ablate_anat:
            K = torch.cat([torch.zeros_like(K[:, :N_ANAT_EDGES]), K[:, N_ANAT_EDGES:]], dim=-1)
        if ablate_skip:
            K = torch.cat([K[:, :N_ANAT_EDGES], torch.zeros_like(K[:, N_ANAT_EDGES:])], dim=-1)

        # Initial condition: lift observed angles into augmented state
        z0 = torch.zeros(B, N, ds, device=stride.device, dtype=stride.dtype)
        z0[:, :, 0] = stride[:, 0, :]  # [B, N] -> first state dim
        z0_flat = z0.reshape(B, N * ds)  # [B, N * d_state]

        # Time grid
        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, T, device=stride.device)

        # Integrate
        self.ode_func.set_K(K)

        odeint_fn = torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint

        z_pred_flat = odeint_fn(
            self.ode_func, z0_flat, t_span, method=self.solver,
        )  # [T_out, B, N * d_state]

        # Extract observed dimension (first state dim of each joint)
        T_out = z_pred_flat.shape[0]
        z_pred = z_pred_flat.view(T_out, B, N, ds)[:, :, :, 0]  # [T_out, B, N]

        return z_pred.permute(1, 0, 2), K  # [B, T_out, N], [B, N_EDGES]

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
