# Time-MVDP: Multivariate Extension of TimeDP

A parameter-efficient multivariate extension of [TimeDP](https://github.com/microsoft/TimeCraft/tree/main) (AAAI 2025). This project adds cross-variate dependency modeling to TimeDP's prototype-based diffusion framework using only **51K trainable adapter parameters** (0.19% of the 27M base model), while keeping the entire pretrained model frozen.

## Overview

TimeDP generates univariate time series by learning domain-unified prototypes and using a Prototype Assignment Module (PAM) to condition a diffusion model. Our extension, **Time-MVDP**, adds two lightweight adapter modules that enable multivariate generation while preserving the pretrained base model:

1. **CrossVariateAdapter** — Top-k sparse self-attention across variates on PAM mask vectors, learning which variates should inform each other's conditioning.
2. **CopulaAdapter** — Post-hoc joint distribution correction inspired by Sklar's theorem, using a learned sparse adjacency graph to adjust cross-variate correlations.

The adapters are fine-tuned on target multivariate datasets while the base model (UNet, PAM, prototypes) remains frozen.

## Repository Structure

This code extends the [Microsoft TimeCraft](https://github.com/microsoft/TimeCraft/tree/main) repository. All changes are within the `TimeDP/` directory.

### New Files

```
TimeDP/
├── ldm/modules/
│   ├── cross_variate_adapter.py    # CrossVariateAdapter: sparse self-attention on PAM masks
│   └── copula_adapter.py           # CopulaAdapter: post-hoc joint distribution correction
├── ldm/data/
│   └── multivariate_dataset.py     # MultivariateDataset/DataModule: (N, T, C) data loading
├── train_adapters.py               # Phase 2: adapter fine-tuning script
├── evaluate_multivariate.py        # Phase 3: evaluation (Time-MVDP vs Time-DP at K=3,5,10)
├── evaluate_sig_mmd.py             # Optional: Signature Kernel MMD evaluation
└── run_experiment.sh               # Orchestrator: runs all phases
```

### Modified Files

```
TimeDP/
├── ldm/models/diffusion/
│   └── ddpm_time.py                # Added multivariate adapter init, training, and sampling
├── utils/
│   ├── cli_utils.py                # Added multivariate CLI arguments
│   ├── init_utils.py               # Added multivariate model/data setup
│   └── test_utils.py               # Added test_model_multivariate()
└── main_train.py                   # Routes to multivariate evaluation when --multivariate
```

### Unchanged Files (frozen base model)

```
TimeDP/
├── ldm/modules/
│   ├── attention.py                # Cross-attention (receives enriched mask, same interface)
│   ├── encoders/modules.py         # PAM (DomainUnifiedPrototyper) — called per-variate, frozen
│   └── diffusionmodules/
│       ├── ts_unet.py              # 1D UNet — frozen, runs per-variate
│       └── util.py                 # Timestep embeddings — unchanged
├── ldm/models/diffusion/
│   └── ddim_time.py                # DDIM sampler — called per-variate, unchanged
└── ldm/data/
    └── tsg_dataset.py              # Original univariate dataset — still used for Phase 1
```

## Key Modifications

### `ddpm_time.py` — LatentDiffusion class

Added to `__init__`:
- Pops `n_variates`, `adapter_top_k`, `adapter_d_model`, etc. from kwargs
- Calls `_init_multivariate_adapters()` to create CrossVariateAdapter and CopulaAdapter

New methods:
- `freeze_base_model()` — Freezes all 27M base parameters, keeps 51K adapter params trainable
- `_get_input_multivariate()` — Encodes each variate through frozen PAM, stacks masks, runs CrossVariateAdapter
- `_p_losses_multivariate()` — Per-variate denoising loss + adapter correlation loss + copula loss
- `sample_multivariate()` — Few-shot multivariate generation with DDIM sampling + copula correction

Modified methods:
- `shared_step()` — Branches to multivariate path when `self.multivariate_mode` is True
- `configure_optimizers()` — Returns optimizer over adapter-only parameters in multivariate mode

### `cli_utils.py`

Added arguments: `--multivariate`, `--n_variates`, `--pretrained_ckpt`, `--adapter_top_k`, `--adapter_d_model`, `--adapter_n_heads`, `--copula_d_model`, `--copula_n_heads`, `--corr_loss_weight`, `--mv_data_paths`, `--ddim_steps`.

### `init_utils.py`

Added multivariate setup: loads pretrained checkpoint, calls `setup_multivariate()`, swaps to `MultivariateDataModule`. Removed `input_channels==1` assertion for multivariate mode.

## Architecture

```
Input: (B, T, C) multivariate window
         │
         ├── Per-variate PAM encoding (frozen) ──► M ∈ R^{B×C×Np}
         │
         ▼
    CrossVariateAdapter (trainable, 4K params)
    ├── Top-k sparse self-attention across C variates
    └── Outputs: M̃ ∈ R^{B×C×Np}, A ∈ {0,1}^{B×C×C}
         │
         ├── Per-variate UNet denoising (frozen) using m̃_c
         │
         ▼
    CopulaAdapter (trainable, 48K params)
    ├── Encode → sparse cross-variate attention (reuses A) → decode
    └── Gated residual correction on marginals
         │
         ▼
Output: (B, C, T) multivariate generated series
```

### Training Losses

1. **Denoising loss** — Per-variate DDPM loss through frozen UNet (gradient flows to adapter via enriched mask)
2. **Adapter loss** — MSE between mask cosine similarity and real cross-variate correlation matrix (direct gradient to CrossVariateAdapter)
3. **Copula loss** — Correlation loss on shuffled→restored data (direct gradient to CopulaAdapter)

## Usage

### Prerequisites

Clone the base repository and set up the environment:

```bash
git clone https://github.com/microsoft/TimeCraft.git
cd TimeCraft/TimeDP
# Follow TimeCraft setup instructions for environment
```

Copy the new/modified files into the repository.

### Phase 1: Train Base Model

```bash
export WANDB_MODE=disabled
export DATA_ROOT="/path/to/TimeDP-Data"

python main_train.py \
    -b configs/multi_domain_timedp.yaml \
    -up -sl 96 -nl 16 -bs 128 -lr 0.001 \
    --gpus 0, -l ./logs --no-test True --max_steps 50000
```

### Phase 2: Fine-tune Adapters

```bash
python train_adapters.py \
    --base_ckpt /path/to/last.ckpt \
    --dataset_csv /path/to/dataset.csv \
    --save_dir ./multivariate/dataset_name \
    --adapter_top_k 3 --batch_size 8 --lr 1e-4 --n_steps 5000
```

### Phase 3: Evaluate

```bash
python evaluate_multivariate.py \
    --ckpt ./multivariate/dataset_name/checkpoints/best.ckpt \
    --dataset_csv /path/to/dataset.csv \
    --save_dir ./multivariate/dataset_name \
    --display_k 5
```

### Or run all phases via the orchestrator:

```bash
bash run_experiment.sh
```

## References

- **TimeDP**: Shen et al., "TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts", AAAI 2025. [arXiv:2501.05403](https://arxiv.org/abs/2501.05403)
- **TimeCraft**: [github.com/microsoft/TimeCraft](https://github.com/microsoft/TimeCraft/tree/main)
- **Sig-MMD**: Redhead et al., "Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting", 2026. [arXiv:2602.10182](https://arxiv.org/abs/2602.10182)

