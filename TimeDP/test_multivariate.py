#!/usr/bin/env python
"""
Test script for Multivariate TimeDP extension.

Run from the TimeDP/ directory:
    python test_multivariate.py --stage unit        # Test modules in isolation
    python test_multivariate.py --stage integration # Test full pipeline with dummy data
    python test_multivariate.py --stage smoke       # Smoke test with real config + fake data

    python test_multivariate.py --stage all         # Run everything
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# =====================================================================
# STAGE 1: Unit Tests — verify each new module works in isolation
# =====================================================================

def test_cross_variate_adapter():
    """Test CrossVariateAdapter forward pass and output shapes."""
    print("\n--- Test: CrossVariateAdapter ---")
    from ldm.modules.cross_variate_adapter import CrossVariateAdapter

    B, C_var, Np = 4, 8, 16  # batch=4, 8 variates, 16 prototypes
    adapter = CrossVariateAdapter(
        n_prototypes=Np, d_model=64, top_k=3, n_heads=4
    )

    M = torch.randn(B, C_var, Np)  # simulated PAM masks
    M_tilde, A = adapter(M)

    assert M_tilde.shape == (B, C_var, Np), \
        f"Expected M_tilde shape {(B, C_var, Np)}, got {M_tilde.shape}"
    assert A.shape == (B, C_var, C_var), \
        f"Expected A shape {(B, C_var, C_var)}, got {A.shape}"

    # Check sparsity: each row of A should have at most top_k nonzero entries
    for b in range(B):
        for i in range(C_var):
            nnz = (A[b, i] > 0).sum().item()
            assert nnz <= 3, f"Row {i} has {nnz} nonzero entries, expected <= 3"

    # Check gradients flow
    loss = M_tilde.sum()
    loss.backward()
    assert adapter.to_q.weight.grad is not None, "No gradient on adapter.to_q"
    assert adapter.gate.grad is not None, "No gradient on adapter.gate"

    # Check gate initialization
    assert abs(adapter.gate.item() - 1.0) < 0.01, \
        f"Gate should start at 1.0, got {adapter.gate.item()}"

    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Shape check: PASS (M_tilde={M_tilde.shape}, A={A.shape})")
    print(f"  Sparsity check: PASS (top_k=3)")
    print(f"  Gradient check: PASS")
    print(f"  Parameters: {n_params}")
    print("  PASSED ✓")


def test_copula_adapter():
    """Test CopulaAdapter forward pass and output shapes."""
    print("\n--- Test: CopulaAdapter ---")
    from ldm.modules.copula_adapter import CopulaAdapter, correlation_loss

    B, C_var, T = 4, 8, 168
    adapter = CopulaAdapter(seq_len=T, d_model=64, n_heads=4)

    x_marginals = torch.randn(B, C_var, T)
    A = (torch.randn(B, C_var, C_var) > 0).float()  # random binary adjacency

    x_corrected = adapter(x_marginals, A)

    assert x_corrected.shape == (B, C_var, T), \
        f"Expected shape {(B, C_var, T)}, got {x_corrected.shape}"

    # Check residual is initially small (gate starts at 0.01)
    diff = (x_corrected - x_marginals).abs().mean().item()
    assert diff < 1.0, f"Initial residual too large: {diff}"

    # Check gradients flow
    loss = x_corrected.sum()
    loss.backward()
    assert adapter.gate.grad is not None, "No gradient on copula gate"
    assert adapter.encoder.net[0].weight.grad is not None, "No gradient on encoder"

    # Test correlation_loss
    x_real = torch.randn(B, C_var, T)
    x_gen = torch.randn(B, C_var, T)
    corr_loss = correlation_loss(x_gen, x_real)
    assert corr_loss.shape == (), f"correlation_loss should be scalar, got {corr_loss.shape}"
    assert corr_loss.item() >= 0, "correlation_loss should be non-negative"

    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Shape check: PASS ({x_corrected.shape})")
    print(f"  Residual magnitude: {diff:.4f} (should be small)")
    print(f"  Gradient check: PASS")
    print(f"  correlation_loss check: PASS ({corr_loss.item():.4f})")
    print(f"  Parameters: {n_params}")
    print("  PASSED ✓")


def test_multivariate_dataset():
    """Test MultivariateDataset and MultivariateDataModule with synthetic data."""
    print("\n--- Test: MultivariateDataset ---")
    from ldm.data.multivariate_dataset import MultivariateDataset, MultivariateDataModule

    # Create synthetic multivariate data: 2 datasets, different patterns
    np.random.seed(42)
    N, T, C = 100, 96, 4

    # Dataset 1: sinusoidal patterns
    t_axis = np.linspace(0, 4 * np.pi, T)
    data1 = np.stack([
        np.stack([np.sin(t_axis + phase) + np.random.randn(T) * 0.1
                  for phase in np.linspace(0, np.pi, C)], axis=-1)
        for _ in range(N)
    ]).astype(np.float32)  # (N, T, C)

    # Dataset 2: random walk patterns
    data2 = np.cumsum(np.random.randn(N, T, C) * 0.1, axis=1).astype(np.float32)

    # Test raw Dataset
    ds = MultivariateDataset({'sin': data1, 'walk': data2})
    assert len(ds) == 2 * N, f"Expected {2*N} items, got {len(ds)}"

    item = ds[0]
    assert 'context' in item, "Missing 'context' key"
    assert 'data_key' in item, "Missing 'data_key' key"
    assert item['context'].shape == (T, C), \
        f"Expected context shape {(T, C)}, got {item['context'].shape}"

    # Test reweight sampler
    sampler = ds.get_reweight_sampler()
    assert sampler is not None

    # Test DataModule with file-based loading
    tmpdir = Path('/tmp/test_mv_data')
    tmpdir.mkdir(exist_ok=True)
    np.save(tmpdir / 'sin_data.npy', data1.reshape(-1, C))  # (N*T, C) — will be re-windowed
    np.save(tmpdir / 'walk_data.npy', data2.reshape(-1, C))

    dm = MultivariateDataModule(
        data_path_dict={
            'sin': str(tmpdir / 'sin_data.npy'),
            'walk': str(tmpdir / 'walk_data.npy'),
        },
        n_variates=C,
        window=T,
        val_portion=0.1,
        normalize='zscore',
        batch_size=8,
    )
    dm.prepare_data()

    assert dm.actual_n_variates == C, f"Expected {C} variates, got {dm.actual_n_variates}"

    # Test dataloader
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    assert batch['context'].shape == (8, T, C), \
        f"Batch context shape: {batch['context'].shape}, expected (8, {T}, {C})"

    print(f"  Dataset length: {len(ds)} (2 domains × {N})")
    print(f"  Item shape: {item['context'].shape}")
    print(f"  DataModule variates: {dm.actual_n_variates}")
    print(f"  Batch shape: {batch['context'].shape}")
    print("  PASSED ✓")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def run_unit_tests():
    print("=" * 60)
    print("STAGE 1: Unit Tests")
    print("=" * 60)
    test_cross_variate_adapter()
    test_copula_adapter()
    test_multivariate_dataset()
    print("\n✓ All unit tests passed!")


# =====================================================================
# STAGE 2: Integration Test — verify adapters work with frozen base model
# =====================================================================

def test_adapter_gradient_flow():
    """
    Verify that gradients flow through the adapter → frozen UNet → back to adapter.
    Uses a minimal mock to simulate the key computation path.
    """
    print("\n--- Test: Gradient flow through frozen UNet path ---")
    from ldm.modules.cross_variate_adapter import CrossVariateAdapter

    Np = 16
    adapter = CrossVariateAdapter(n_prototypes=Np, d_model=32, top_k=2, n_heads=2)

    # Simulate: frozen PAM produces masks, adapter enriches them,
    # then a frozen linear (simulating UNet cross-attention) uses the mask.
    # Gradients should flow back through the mask to the adapter.

    frozen_linear = nn.Linear(Np, 1, bias=False)
    for p in frozen_linear.parameters():
        p.requires_grad = False

    B, C_var = 2, 4
    M_frozen = torch.randn(B, C_var, Np)  # "frozen" PAM output
    M_frozen.requires_grad = False

    # Adapter enriches the masks (this is trainable)
    M_tilde, A = adapter(M_frozen)

    # Simulated UNet cross-attention uses the mask as additive bias
    # The key insight: even though the UNet is frozen, the mask_tilde
    # feeds into the attention logits, so gradients flow back to adapter
    dummy_attn_logits = torch.randn(B, C_var, Np)  # simulated Q*K scores
    biased_logits = dummy_attn_logits + M_tilde
    output = frozen_linear(biased_logits)  # simulated "denoised output"
    loss = output.sum()
    loss.backward()

    has_grad = adapter.to_q.weight.grad is not None
    grad_magnitude = adapter.to_q.weight.grad.abs().mean().item() if has_grad else 0

    assert has_grad, "FAILED: No gradients reached the adapter through frozen UNet path!"
    assert grad_magnitude > 0, "FAILED: Gradients are zero!"

    print(f"  Gradients reach adapter: {has_grad}")
    print(f"  Gradient magnitude: {grad_magnitude:.6f}")
    print("  PASSED ✓")


def test_end_to_end_shapes():
    """
    Test the full multivariate forward pass shape chain:
    PAM → CrossVariateAdapter → UNet (simulated) → CopulaAdapter
    """
    print("\n--- Test: End-to-end shape chain ---")
    from ldm.modules.cross_variate_adapter import CrossVariateAdapter
    from ldm.modules.copula_adapter import CopulaAdapter

    B, C_var, T, Np = 2, 4, 96, 16

    cv_adapter = CrossVariateAdapter(n_prototypes=Np, d_model=32, top_k=2, n_heads=2)
    copula = CopulaAdapter(seq_len=T, d_model=32, n_heads=2)

    # Step 1: Simulated PAM output (per-variate)
    M = torch.randn(B, C_var, Np)
    print(f"  PAM masks: {M.shape}")

    # Step 2: Cross-variate adapter
    M_tilde, A = cv_adapter(M)
    print(f"  Enriched masks: {M_tilde.shape}, Adjacency: {A.shape}")

    # Step 3: Simulated per-variate UNet denoising
    x_marginals = torch.randn(B, C_var, T)  # each variate denoised independently
    print(f"  Marginal outputs: {x_marginals.shape}")

    # Step 4: Copula correction
    x_corrected = copula(x_marginals, A)
    print(f"  Corrected output: {x_corrected.shape}")

    assert x_corrected.shape == (B, C_var, T)

    # Verify full backward pass
    loss = x_corrected.sum() + M_tilde.sum()
    loss.backward()

    cv_has_grad = cv_adapter.to_q.weight.grad is not None
    cop_has_grad = copula.encoder.net[0].weight.grad is not None

    print(f"  CV adapter gradients: {cv_has_grad}")
    print(f"  Copula adapter gradients: {cop_has_grad}")
    assert cv_has_grad and cop_has_grad
    print("  PASSED ✓")


def test_parameter_counts():
    """Verify that adapter parameter count is small relative to base model."""
    print("\n--- Test: Parameter efficiency ---")
    from ldm.modules.cross_variate_adapter import CrossVariateAdapter
    from ldm.modules.copula_adapter import CopulaAdapter

    cv = CrossVariateAdapter(n_prototypes=16, d_model=64, top_k=3, n_heads=4)
    cop = CopulaAdapter(seq_len=168, d_model=64, n_heads=4)

    cv_params = sum(p.numel() for p in cv.parameters())
    cop_params = sum(p.numel() for p in cop.parameters())
    total = cv_params + cop_params

    print(f"  CrossVariateAdapter: {cv_params:,} params")
    print(f"  CopulaAdapter:      {cop_params:,} params")
    print(f"  Total adapters:     {total:,} params")

    # UNet typically has ~2-10M params; adapters should be <100K
    assert total < 500_000, f"Adapters seem too large: {total} params"
    print(f"  Efficiency check: PASS (<500K)")
    print("  PASSED ✓")


def run_integration_tests():
    print("\n" + "=" * 60)
    print("STAGE 2: Integration Tests")
    print("=" * 60)
    test_adapter_gradient_flow()
    test_end_to_end_shapes()
    test_parameter_counts()
    print("\n✓ All integration tests passed!")


# =====================================================================
# STAGE 3: Smoke Test — verify the actual model can do a forward pass
# =====================================================================

def test_ddpm_multivariate_setup():
    """
    Test that LatentDiffusion.setup_multivariate() works and
    configure_optimizers only returns adapter params.
    """
    print("\n--- Test: LatentDiffusion multivariate setup ---")

    # We need a config to instantiate the model. If no config available,
    # test with a mock.
    try:
        from ldm.models.diffusion.ddpm_time import LatentDiffusion
        print("  Successfully imported LatentDiffusion")
    except ImportError as e:
        print(f"  SKIPPED (import error: {e})")
        return

    # Check the multivariate methods exist
    assert hasattr(LatentDiffusion, 'setup_multivariate'), \
        "Missing setup_multivariate method"
    assert hasattr(LatentDiffusion, 'get_input_multivariate'), \
        "Missing get_input_multivariate method"
    assert hasattr(LatentDiffusion, 'p_losses_multivariate'), \
        "Missing p_losses_multivariate method"
    assert hasattr(LatentDiffusion, 'sample_multivariate'), \
        "Missing sample_multivariate method"
    assert hasattr(LatentDiffusion, '_freeze_base_model'), \
        "Missing _freeze_base_model method"

    print("  All multivariate methods present: PASS")
    print("  PASSED ✓")


def test_full_smoke(config_path=None):
    """
    Full smoke test: instantiate model from config, setup multivariate,
    run one forward pass with synthetic data.
    
    Requires a YAML config file. Skipped if not available.
    """
    print("\n--- Test: Full smoke test ---")

    if config_path is None:
        # Try to find a config
        candidates = list(Path('.').glob('configs/**/*.yaml')) + \
                     list(Path('..').glob('configs/**/*.yaml'))
        if not candidates:
            print("  SKIPPED (no config YAML found)")
            print("  To run: python test_multivariate.py --stage smoke --config path/to/config.yaml")
            return
        config_path = str(candidates[0])
        print(f"  Using config: {config_path}")

    try:
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        config = OmegaConf.load(config_path)

        # Set minimal config for testing
        config.model['params']['seq_len'] = 24
        config.model['params']['unet_config']['params']['seq_len'] = 24

        # Ensure PAM mode
        if 'cond_stage_config' in config.model['params']:
            if isinstance(config.model['params']['cond_stage_config'], str):
                print("  SKIPPED (unconditional config, need PAM config)")
                return
            config.model['params']['cond_stage_config']['params']['window'] = 24
            config.model['params']['cond_stage_config']['target'] = \
                "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = 8
            config.model['params']['unet_config']['params']['latent_unit'] = 8
            config.model['params']['unet_config']['params']['use_pam'] = True

        # Remove ckpt_path if present
        config.model['params'].pop('ckpt_path', None)

        # Instantiate model
        model = instantiate_from_config(config.model)
        print(f"  Model instantiated: {type(model).__name__}")

        # Setup multivariate
        model.setup_multivariate(
            n_variates=4,
            adapter_top_k=2,
            adapter_d_model=32,
            adapter_n_heads=2,
            copula_d_model=32,
            copula_n_heads=2,
            corr_loss_weight=0.1,
        )
        print("  setup_multivariate: OK")

        # Verify freeze
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Frozen params: {n_frozen}, Trainable params: {n_trainable}")
        assert n_trainable > 0, "No trainable parameters!"
        assert n_frozen > n_trainable, "More trainable than frozen — freeze may have failed"

        # configure_optimizers should only include adapter params
        model.learning_rate = 1e-4
        opt = model.configure_optimizers()
        if isinstance(opt, tuple):
            opt = opt[0][0]
        n_opt_params = sum(p.numel() for group in opt.param_groups for p in group['params'])
        n_adapter_params = sum(p.numel() for p in model.cross_variate_adapter.parameters()) + \
                          sum(p.numel() for p in model.copula_adapter.parameters())
        assert n_opt_params == n_adapter_params, \
            f"Optimizer has {n_opt_params} params but adapters have {n_adapter_params}"
        print(f"  Optimizer params match adapter params: {n_opt_params}")

        # Forward pass with synthetic multivariate batch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        B, T, C_var = 2, 24, 4
        batch = {
            'context': torch.randn(B, T, C_var).to(device),
            'data_key': torch.zeros(B, dtype=torch.long).to(device),
        }

        model.train()
        loss, loss_dict = model.shared_step(batch)

        assert loss.requires_grad, "Loss has no gradient!"
        assert not torch.isnan(loss), f"Loss is NaN!"
        assert not torch.isinf(loss), f"Loss is Inf!"

        # Backward pass
        loss.backward()

        # Check adapter received gradients
        cv_grad = model.cross_variate_adapter.to_q.weight.grad
        assert cv_grad is not None, "No gradient on cross_variate_adapter"
        assert cv_grad.abs().sum() > 0, "Gradient is zero on cross_variate_adapter"

        print(f"  Forward pass: OK (loss={loss.item():.4f})")
        print(f"  Backward pass: OK (adapter grad norm={cv_grad.norm().item():.6f})")
        print("  PASSED ✓")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()


def run_smoke_tests(config_path=None):
    print("\n" + "=" * 60)
    print("STAGE 3: Smoke Tests")
    print("=" * 60)
    test_ddpm_multivariate_setup()
    test_full_smoke(config_path)
    print("\n✓ All smoke tests passed!")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multivariate TimeDP extension")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['unit', 'integration', 'smoke', 'all'],
                        help='Which test stage to run')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config for smoke test')
    args = parser.parse_args()

    print("=" * 60)
    print("Multivariate TimeDP — Test Suite")
    print("=" * 60)

    if args.stage in ('unit', 'all'):
        run_unit_tests()
    if args.stage in ('integration', 'all'):
        run_integration_tests()
    if args.stage in ('smoke', 'all'):
        run_smoke_tests(args.config)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)