# Differentiable Movement Dynamics: Body graph neural ODE for gait kinematics

This repository provides the PyTorch implementation of the Graph-Constrained Neural ODE (`GraphGaitODE`) designed to extract time-varying effective locomotor coupling from continuous gait kinematics.

This codebase is packaged as a tool for researchers to apply our Differentiable Movement Dynamics (DMD) framework to their own kinematic data.

## 1. Installation

Requires Python 3.11+. Recommend [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
# Clone the repository
git clone https://github.com/neurocog-mobility/diff-dynamics-pd-gait.git
cd diff-dynamics-pd-gait

# Install dependencies and create a virtual environment
uv sync

# All scripts should be run via uv run, or with the venv activated
source .venv/bin/activate
```

Alternatively:

```bash
pip install .
```

## 2. Using your own data

The model expects a dataset of time-normalized individual strides, shaped as `[N_strides, 101, 10]`, where the 10 columns are the canonical joints.

### Step 2a: Formatting your CSV
If you have raw kinematics exported as generic CSVs, your data should look like this:
- **Columns**: `Time, L_Sho, R_Sho, Trunk, Pelvis, L_Hip, R_Hip, L_Knee, R_Knee, L_Ank, R_Ank, L_Heel_Strike, R_Heel_Strike`
- Where the `_Heel_Strike` columns are binary indicators (1 at the frame of foot contact, 0 otherwise).

### Step 2b: Building the dataset
We provide a template script to read your CSVs, segment the data by heel strikes, interpolate/normalize to 101 points (0-100% of gait cycle), and perform the **canonicalization step** (swapping left/right so that the reference/striking limb is always in the same column position).

```bash
# Edit scripts/build_dataset.py to point to your CSV folder
uv run python scripts/build_dataset.py
```
This outputs `data/processed/strides_custom.npz`.

## 3. Inference & feature extraction

Once your data is formatted, you can load our pre-trained checkpoints (or your own) to extract the Effective Locomotor Coupling (ELC) metrics.

**Basic extraction:**
```bash
# Shows how to load the model and extract the 15-edge static coupling matrix (K)
uv run python scripts/example_load_checkpoint.py
```

**Single stride analysis:**
```bash
# Computes the edge coupling matrix (K) and the Fiedler value (λ₂) for a chosen stride
uv run python scripts/analyze_stride.py --stride-index 0 --checkpoint results/model/checkpoints/model_sub01.pt
```

**Visualization:**
```bash
# Generates a filmstrip visualizing the Neural ODE's trajectory reconstruction
uv run python scripts/fig_model_walks.py --stride-index 0 --checkpoint results/model/checkpoints/model_sub01.pt
```

## 4. Data

The included `data/processed/strides.npz` is a derived dataset extracted and time-normalized from the raw kinematic recordings of Shida et al. (2023), available at https://doi.org/10.6084/m9.figshare.14896881 under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The original data have been segmented into individual strides, interpolated to 101 time-normalized points, and canonicalized (striking limb → reference position).

## 5. Training (reference)
If you wish to train the model from scratch on your own dataset, you can use the provided training script:
```bash
uv run python scripts/train.py --fold-limit 1  # smoke test (1 fold)
uv run python scripts/train.py                 # full 5-fold run
```
