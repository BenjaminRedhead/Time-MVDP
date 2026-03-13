#!/usr/bin/env python
# Created for internship coding assessment requires additional packages not in setup.sh
"""
Evaluate generated multivariate time series using Signature Kernel MMD.

Loads saved .npy files from evaluate_multivariate.py and computes SigMMD,
which captures temporal ordering and cross-variate dependencies jointly —
a much stronger metric than per-variate polynomial MMD.

Usage:
    python evaluate_sig_mmd.py \
        --results_dir /path/to/multivariate/exchange \
        --dataset_csv /path/to/exchange_rate.csv \
        --seq_len 96

Requires: signatory, sigkernel (pip install signatory sigkernel)
"""

import os
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---- Signature MMD components (from provided code) ----

class PathReVIN(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(1, 1, dim))
        self.register_buffer('std', torch.ones(1, 1, dim))

    def fit(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True)
        self.std = torch.where(std < 1e-1, torch.ones_like(std), std)

    def forward(self, x):
        return (x - self.mean) / self.std


class AugmentPipeline(torch.nn.Module):
    def __init__(self, append_time=True):
        super().__init__()
        self.append_time = append_time

    def forward(self, x):
        B, L, C = x.shape
        device = x.device
        if self.append_time:
            time_channel = torch.linspace(0, 1, L, device=device).reshape(1, L, 1).expand(B, L, 1)
            x = torch.cat([x, time_channel], dim=2)
        curr_C = x.shape[-1]
        zero_start = torch.zeros(B, 1, curr_C, device=device)
        zero_end = torch.zeros(B, 1, curr_C, device=device)
        zero_end[:, :, -1] = x[:, -1:, -1]
        return torch.cat([zero_start, x, zero_end], dim=1)


class SigmaEstimator:
    @staticmethod
    def fit(train_data_norm, max_samples=250):
        N = train_data_norm.shape[0]
        x_flat = train_data_norm.reshape(N, -1)
        idx = torch.randperm(N)[:max_samples]
        subset = x_flat[idx]
        dists = torch.cdist(subset, subset, p=2)
        valid_dists = dists[dists > 1e-6]
        if len(valid_dists) > 0:
            sigma = torch.median(valid_dists)
        else:
            sigma = torch.tensor(1.0)
        return sigma.item()


def compute_sig_mmd(real_ntc, gen_ntc, sigma=None, max_samples=50, device='cpu'):
    """
    Compute Signature Kernel MMD between real and generated multivariate paths.

    Args:
        real_ntc: (N, T, C) real multivariate windows
        gen_ntc:  (N, T, C) generated multivariate windows
        sigma:    RBF kernel bandwidth (auto-estimated if None)
        max_samples: subsample size (sig kernel is O(N^2 * T^2))
        device:   'cpu' or 'cuda'

    Returns:
        dict with sig_mmd value and sigma used
    """
    import sigkernel

    C = real_ntc.shape[-1]

    # Subsample — sig kernel is expensive
    n_real = min(max_samples, len(real_ntc))
    n_gen = min(max_samples, len(gen_ntc))
    real_sub = real_ntc[:n_real].clone()
    gen_sub = gen_ntc[:n_gen].clone()

    # Normalize using PathReVIN (fit on real data)
    revin = PathReVIN(C)
    revin.fit(real_sub)
    real_norm = revin(real_sub)
    gen_norm = revin(gen_sub)

    # Augment (add time channel + zero padding)
    augment = AugmentPipeline(append_time=True)
    real_aug = augment(real_norm)
    gen_aug = augment(gen_norm)

    # Estimate sigma if not provided
    if sigma is None:
        sigma = SigmaEstimator.fit(real_norm)
        print(f'    Auto sigma: {sigma:.4f}')

    # Move to device
    real_aug = real_aug.to(device)
    gen_aug = gen_aug.to(device)

    # Compute signature kernel MMD
    static_kernel = sigkernel.RBFKernel(sigma=sigma)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)

    def normalized_gram(X, Y):
        K_xy = sig_kernel.compute_Gram(X, Y, sym=(X is Y))
        diag_X = sig_kernel.compute_kernel(X, X)
        diag_Y = sig_kernel.compute_kernel(Y, Y)
        denom = torch.sqrt(torch.outer(diag_X, diag_Y)) + 1e-8
        return K_xy / denom

    with torch.no_grad():
        K_xx = normalized_gram(real_aug, real_aug)
        K_yy = normalized_gram(gen_aug, gen_aug)
        K_xy = normalized_gram(real_aug, gen_aug)

        mmd = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()).item()

    return {'sig_mmd': float(mmd), 'sigma': float(sigma),
            'n_real': n_real, 'n_gen': n_gen}


def parse_args():
    parser = argparse.ArgumentParser(description='Signature Kernel MMD evaluation')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing generated_MVDP_K*.npy and generated_DP_K*.npy')
    parser.add_argument('--dataset_csv', type=str, required=True,
                        help='Path to dataset CSV')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Max samples for SigMMD (expensive: O(N^2 * T^2))')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_real_windows(csv_path, seq_len, seed):
    """Load and normalize real data windows."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.columns if c != 'date']
    data = df[numeric_cols].values.astype(np.float32)
    C_var = data.shape[1]

    n_win = len(data) // seq_len
    windows = data[:n_win * seq_len].reshape(n_win, seq_len, C_var)
    for c in range(C_var):
        col = windows[:, :, c]
        windows[:, :, c] = (col - col.mean()) / (col.std() + 1e-8)

    np.random.seed(seed)
    indices = np.random.permutation(len(windows))
    n_val = max(int(len(windows) * 0.15), 3)
    test = windows[indices[-n_val:]]  # Use test split

    return torch.tensor(windows).float(), torch.tensor(test).float(), numeric_cols


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real data — (N, T, C) format for sig kernel
    all_windows, test_windows, variate_names = load_real_windows(
        args.dataset_csv, args.seq_len, args.seed
    )
    C_var = all_windows.shape[-1]
    print(f'Dataset: {C_var} variates, Test: {test_windows.shape}')
    print(f'Device: {device}, Max samples: {args.max_samples}')

    # Estimate sigma from real data once
    revin = PathReVIN(C_var)
    revin.fit(all_windows)
    sigma = SigmaEstimator.fit(revin(all_windows))
    print(f'Global sigma: {sigma:.4f}')

    results = {}

    for K in [3, 5, 10]:
        print(f'\n{"="*50}')
        print(f'  K = {K}')
        print(f'{"="*50}')

        for method in ['MVDP', 'DP']:
            npy_path = os.path.join(args.results_dir, f'generated_{method}_K{K}.npy')
            if not os.path.exists(npy_path):
                print(f'  Time-{method}-{K}: {npy_path} not found, skipping')
                continue

            gen_nct = np.load(npy_path)  # (N, C, T)
            gen_ntc = torch.tensor(gen_nct).float().permute(0, 2, 1)  # → (N, T, C)

            print(f'  Time-{method}-{K}: computing SigMMD ({gen_ntc.shape[0]} samples)...')
            sig_result = compute_sig_mmd(
                test_windows, gen_ntc,
                sigma=sigma,
                max_samples=args.max_samples,
                device=device
            )
            results[f'Time-{method}-{K}'] = sig_result
            print(f'    SigMMD: {sig_result["sig_mmd"]:.6f}')

    # Save results
    out_path = os.path.join(args.results_dir, 'sig_mmd_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print('\n' + '=' * 60)
    print('SIGNATURE KERNEL MMD RESULTS')
    print('=' * 60)
    print(f'  {"Method":<20} {"SigMMD":>12} {"poly MMD":>12}')
    print('  ' + '-' * 46)

    # Load original results for comparison
    eval_path = os.path.join(args.results_dir, 'evaluation_results.json')
    orig = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            orig = json.load(f)

    for k in [3, 5, 10]:
        for method in ['MVDP', 'DP']:
            key = f'Time-{method}-{k}'
            sig_val = results.get(key, {}).get('sig_mmd', float('nan'))
            poly_val = orig.get(key, {}).get('avg_mmd', float('nan'))
            print(f'  {key:<20} {sig_val:>12.6f} {poly_val:>12.4f}')
        print('  ' + '-' * 46)

    # Generate comparison figure
    fig_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    k_values = [3, 5, 10]
    mvdp_sig = [results.get(f'Time-MVDP-{k}', {}).get('sig_mmd', 0) for k in k_values]
    dp_sig = [results.get(f'Time-DP-{k}', {}).get('sig_mmd', 0) for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(x - width/2, mvdp_sig, width, label='Time-MVDP', color='#2196F3')
    ax.bar(x + width/2, dp_sig, width, label='Time-DP', color='#FF9800')
    ax.set_xlabel('Number of Few-Shot Prompts (K)')
    ax.set_ylabel('Signature Kernel MMD')
    ax.set_title('Multivariate Generation Quality (Signature Kernel MMD)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/sig_mmd_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {fig_dir}/sig_mmd_comparison.pdf')
    print(f'Results: {out_path}')


if __name__ == '__main__':
    main()