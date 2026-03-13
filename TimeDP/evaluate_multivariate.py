#!/usr/bin/env python
"""
Phase 3: Evaluate multivariate generation — Time-MVDP vs Time-DP.

Compares at matched K=3,5,10 shot settings and produces report figures.
Both methods use DDIM sampling with the same number of steps.

Usage:
    python evaluate_multivariate.py \
        --ckpt /path/to/best.ckpt \
        --dataset_csv /path/to/exchange_rate.csv \
        --save_dir /path/to/output \
        --display_k 5
"""

import os
import gc
import json
import glob
import argparse
import numpy as np
import pandas as pd
import torch

os.environ.setdefault('DATA_ROOT', '/tmp')
os.environ['WANDB_MODE'] = 'disabled'

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.copula_adapter import correlation_loss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multivariate TimeDP')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/multi_domain_timedp.yaml')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--base_log_dir', type=str, default=None,
                        help='Path to Phase 1 log directory (for base training loss curve)')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--n_gen', type=int, default=100)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--display_k', type=int, default=5,
                        help='Which K to use for visual figures (generated_vs_real, correlation_matrices)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_data(csv_path, seq_len, seed):
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
    return windows[indices[:-n_val]], windows[indices[-n_val:]], numeric_cols, C_var


def build_and_load_model(config_path, ckpt_path, seq_len, num_latents, C_var):
    config = OmegaConf.load(config_path)
    config.model['params']['seq_len'] = seq_len
    config.model['params']['unet_config']['params']['seq_len'] = seq_len
    config.model['params']['cond_stage_config']['params']['window'] = seq_len
    config.model['params']['cond_stage_config']['params']['num_latents'] = num_latents
    config.model['params']['unet_config']['params']['latent_unit'] = num_latents
    config.model['params'].pop('ckpt_path', None)
    config.model['params']['n_variates'] = C_var
    config.model['params']['adapter_top_k'] = 3
    config.model['params']['adapter_d_model'] = 64
    config.model['params']['adapter_n_heads'] = 4
    config.model['params']['copula_d_model'] = 64
    config.model['params']['copula_n_heads'] = 4
    config.model['params']['corr_loss_weight'] = 0.1

    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd['state_dict'], strict=False)
    step = sd.get('step', '?')
    del sd
    gc.collect()
    print(f'Loaded checkpoint: step {step}')
    return model


def mmd_poly(x, y, degree=2, gamma=1.0, coef0=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    return ((gamma * xx + coef0)**degree).mean() + ((gamma * yy + coef0)**degree).mean() - 2 * ((gamma * xy + coef0)**degree).mean()


def generate_baseline(model, train_windows, K, n_gen, C_var, device, ddim_steps):
    """
    Time-DP baseline: generate each variate independently using K prompts.
    Uses DDIM sampling to match Time-MVDP inference cost.
    """
    uncond_gen = []
    for c in range(C_var):
        x_prompt = torch.tensor(train_windows[:K, :, c:c+1]).float().to(device).permute(0, 2, 1)
        with torch.no_grad():
            ctx, mask = model.cond_stage_model(x_prompt)
        n_rep = n_gen // K + 1
        cond = torch.repeat_interleave(ctx, n_rep, dim=0)[:n_gen]
        mask_r = torch.repeat_interleave(mask, n_rep, dim=0)[:n_gen]
        with torch.no_grad():
            samples, _ = model.sample_log(
                cond=cond, batch_size=n_gen, ddim=True,
                ddim_steps=ddim_steps, cfg_scale=1, mask=mask_r
            )
            decoded = model.decode_first_stage(samples)
        uncond_gen.append(decoded.squeeze(1))

    return torch.stack(uncond_gen, dim=1).to(device)  # (n_gen, C, T)


def compute_metrics(generated, real, C_var):
    n_eval = min(len(generated), len(real))
    gen = generated[:n_eval]
    ref = real[:n_eval]

    corr_err = correlation_loss(gen, ref).item()
    per_var_mmd = [mmd_poly(gen[:, c].cpu(), ref[:, c].cpu()).item() for c in range(C_var)]
    avg_mmd = np.mean(per_var_mmd)

    return {
        'corr_error': float(corr_err),
        'avg_mmd': float(avg_mmd),
        'per_variate_mmd': [float(m) for m in per_var_mmd],
    }


def evaluate(model, train_windows, test_windows, C_var, n_gen, ddim_steps, device, save_dir):
    """Run Time-MVDP and Time-DP at matched K=3,5,10."""
    results = {}
    real = torch.tensor(test_windows).float().to(device).permute(0, 2, 1)

    for K in [3, 5, 10]:
        print(f'\n{"="*50}')
        print(f'  K = {K} shot')
        print(f'{"="*50}')

        # ---- Time-MVDP ----
        print(f'  Time-MVDP-{K}: generating {n_gen} samples...')
        few_shot = torch.tensor(train_windows[:K]).float()
        with torch.no_grad():
            mvdp_generated = model.sample_multivariate(
                {'context': few_shot}, n_samples=n_gen,
                ddim_steps=ddim_steps, eta=1.0
            )

        mvdp_metrics = compute_metrics(mvdp_generated.to(device), real, C_var)
        results[f'Time-MVDP-{K}'] = mvdp_metrics
        np.save(f'{save_dir}/generated_MVDP_K{K}.npy', mvdp_generated.cpu().numpy())
        del mvdp_generated
        print(f'    Corr: {mvdp_metrics["corr_error"]:.4f}, MMD: {mvdp_metrics["avg_mmd"]:.6f}')

        # ---- Time-DP ----
        print(f'  Time-DP-{K}: generating {n_gen} samples (independent)...')
        dp_generated = generate_baseline(model, train_windows, K, n_gen, C_var, device, ddim_steps)

        dp_metrics = compute_metrics(dp_generated, real, C_var)
        results[f'Time-DP-{K}'] = dp_metrics
        np.save(f'{save_dir}/generated_DP_K{K}.npy', dp_generated.cpu().numpy())
        del dp_generated
        print(f'    Corr: {dp_metrics["corr_error"]:.4f}, MMD: {dp_metrics["avg_mmd"]:.6f}')

        # ---- Improvement ----
        corr_imp = (dp_metrics['corr_error'] - mvdp_metrics['corr_error']) / dp_metrics['corr_error'] * 100
        mmd_imp = (dp_metrics['avg_mmd'] - mvdp_metrics['avg_mmd']) / dp_metrics['avg_mmd'] * 100
        print(f'    Improvement: corr {corr_imp:+.1f}%, mmd {mmd_imp:+.1f}%')

    return results


def extract_base_losses(base_log_dir):
    """Try to extract training loss from Phase 1 Lightning logs."""
    if base_log_dir is None:
        return None, None

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = glob.glob(os.path.join(base_log_dir, '**', 'events.out.tfevents.*'), recursive=True)
        if event_files:
            ea = EventAccumulator(os.path.dirname(event_files[0]))
            ea.Reload()
            for tag in ['train/loss_simple', 'train/loss', 'train/loss_simple_step']:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    steps = np.array([e.step for e in events])
                    values = np.array([e.value for e in events])
                    print(f'  Extracted {len(steps)} base loss values from tensorboard ({tag})')
                    return steps, values
    except ImportError:
        pass
    except Exception as e:
        print(f'  Tensorboard extraction failed: {e}')

    csv_files = glob.glob(os.path.join(base_log_dir, '**', 'metrics.csv'), recursive=True)
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            for col in ['train/loss_simple_step', 'train/loss_step', 'train/loss_simple']:
                if col in df.columns:
                    valid = df[col].dropna()
                    print(f'  Extracted {len(valid)} base loss values from CSV ({col})')
                    return np.arange(len(valid)), valid.values
        except Exception as e:
            print(f'  CSV extraction failed: {e}')

    version_dirs = glob.glob(os.path.join(base_log_dir, '**', 'version_*'), recursive=True)
    for vdir in version_dirs:
        csv_path = os.path.join(vdir, 'metrics.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for col in df.columns:
                    if 'loss' in col.lower() and 'step' in col.lower():
                        valid = df[col].dropna()
                        if len(valid) > 10:
                            print(f'  Extracted {len(valid)} base loss values from {csv_path} ({col})')
                            return np.arange(len(valid)), valid.values
            except:
                pass

    print('  Could not extract base training losses')
    return None, None


def generate_figures(results, model, windows, variate_names, C_var,
                     save_dir, device, base_log_dir, display_k):
    fig_dir = f'{save_dir}/figures'
    os.makedirs(fig_dir, exist_ok=True)

    smooth = lambda x, w=50: np.convolve(x, np.ones(w)/w, mode='valid') if len(x) > w else x

    # ==================================================================
    # Fig 1: Combined convergence — base model then adapter
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print('\nExtracting base model training losses...')
    base_steps, base_losses = extract_base_losses(base_log_dir)
    if base_losses is not None:
        w = min(100, max(1, len(base_losses) // 5))
        axes[0].plot(base_steps[:len(smooth(base_losses, w))], smooth(base_losses, w),
                     linewidth=0.8, color='#2196F3', label='Base model loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Phase 1: Base Model Training (27M params)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Base training logs\nnot available',
                     ha='center', va='center', transform=axes[0].transAxes, fontsize=14, alpha=0.5)
        axes[0].set_title('Phase 1: Base Model Training (27M params)')

    for name, color, label in [
        ('loss', '#F44336', 'Total loss'),
        ('adapter', '#FF9800', 'Adapter loss'),
        ('copula', '#4CAF50', 'Copula loss'),
    ]:
        path = f'{save_dir}/train_{name}.npy'
        if os.path.exists(path):
            vals = np.load(path)
            axes[1].plot(smooth(vals), linewidth=0.8, color=color, label=label,
                        alpha=0.7 if name == 'loss' else 1.0)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Phase 2: Adapter Fine-tuning (51K params)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/convergence_combined.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: convergence_combined.pdf')

    # ==================================================================
    # Fig 2: Training efficiency comparison
    # ==================================================================
    timing_path = f'{save_dir}/adapter_timing.json'
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            adapter_timing = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        base_params = adapter_timing['base_params']
        adapter_params = adapter_timing['adapter_params']
        labels = ['Base Model\n(frozen)', 'Adapters\n(trained)']
        sizes = [base_params, adapter_params]

        axes[0].bar(labels, sizes, color=['#90CAF9', '#F44336'], width=0.5)
        axes[0].set_ylabel('Parameter Count')
        axes[0].set_title('Parameter Efficiency')
        for i, size in enumerate(sizes):
            pct = 100 * size / sum(sizes)
            axes[0].text(i, size + max(sizes) * 0.02, f'{size:,}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=10)

        wall_time_path = f'{save_dir}/train_wall_time.npy'
        if os.path.exists(wall_time_path):
            wall_times = np.load(wall_time_path)
            loss_vals = np.load(f'{save_dir}/train_loss.npy')
            smoothed_loss = smooth(loss_vals)
            axes[1].plot(wall_times[:len(smoothed_loss)] / 60, smoothed_loss, linewidth=0.8, color='#F44336')
            axes[1].set_xlabel('Wall Time (minutes)')
            axes[1].set_ylabel('Loss')
            total_min = adapter_timing['adapter_train_seconds'] / 60
            axes[1].set_title(f'Adapter Training: {total_min:.1f} min total')
            axes[1].grid(True, alpha=0.3)
        else:
            total_min = adapter_timing['adapter_train_seconds'] / 60
            axes[1].text(0.5, 0.5, f'Total: {total_min:.1f} min\n'
                        f'{adapter_timing["seconds_per_step"]:.3f} s/step',
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Adapter Training Time')

        plt.tight_layout()
        plt.savefig(f'{fig_dir}/training_efficiency.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved: training_efficiency.pdf')

    # ==================================================================
    # Fig 3: Generated vs Real (at display_k)
    # ==================================================================
    mvdp_path = f'{save_dir}/generated_MVDP_K{display_k}.npy'
    dp_path = f'{save_dir}/generated_DP_K{display_k}.npy'

    if os.path.exists(mvdp_path):
        gen_mvdp = np.load(mvdp_path)
        variate_colors = plt.cm.tab10(np.linspace(0, 1, C_var))

        fig, axes = plt.subplots(C_var, 2, figsize=(14, 2.2 * C_var), sharex=True)
        fig.suptitle(f'Time-MVDP-{display_k} (Generated) vs Real Exchange Rates',
                     fontsize=13, y=1.01)

        n_show = min(5, len(gen_mvdp), len(windows))
        alphas = np.linspace(0.3, 0.8, n_show)

        for c in range(C_var):
            color = variate_colors[c]
            for i in range(n_show):
                axes[c, 0].plot(gen_mvdp[i, c, :], alpha=alphas[i], linewidth=0.9, color=color)
                axes[c, 1].plot(windows[i, :, c], alpha=alphas[i], linewidth=0.9, color=color)
            axes[c, 0].set_ylabel(variate_names[c], fontsize=9)
            if c == 0:
                axes[c, 0].set_title('Generated (Time-MVDP)')
                axes[c, 1].set_title('Real')

        axes[-1, 0].set_xlabel('Time Step')
        axes[-1, 1].set_xlabel('Time Step')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/generated_vs_real.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved: generated_vs_real.pdf')

    # ==================================================================
    # Fig 4: Correlation matrices — Time-MVDP vs Time-DP vs Real
    # ==================================================================
    if os.path.exists(mvdp_path) and os.path.exists(dp_path):
        gen_mvdp = np.load(mvdp_path)
        gen_dp = np.load(dp_path)

        def corr_mat(x_nct):
            flat = x_nct.transpose(0, 2, 1).reshape(-1, x_nct.shape[1])
            return np.corrcoef(flat.T)

        real_corr = corr_mat(windows.transpose(0, 2, 1))
        mvdp_corr = corr_mat(gen_mvdp)
        dp_corr = corr_mat(gen_dp)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        kw = dict(vmin=-1, vmax=1, cmap='RdBu_r')
        diff_kw = dict(vmin=-0.5, vmax=0.5, cmap='RdBu_r')

        for ax, mat, title, use_kw in zip(
            axes,
            [real_corr, mvdp_corr, dp_corr, mvdp_corr - real_corr],
            ['Real', f'Time-MVDP-{display_k}', f'Time-DP-{display_k}', 'MVDP - Real'],
            [kw, kw, kw, diff_kw]
        ):
            im = ax.imshow(mat, **use_kw)
            ax.set_title(title)
            ax.set_xticks(range(C_var))
            ax.set_yticks(range(C_var))
            ax.set_xticklabels(variate_names, rotation=45, fontsize=8)
            ax.set_yticklabels(variate_names, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f'{fig_dir}/correlation_matrices.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved: correlation_matrices.pdf')

    # ==================================================================
    # Fig 5: Adjacency graph
    # ==================================================================
    all_A = []
    with torch.no_grad():
        for i in range(min(20, len(windows))):
            ml = []
            for c in range(C_var):
                x_c = torch.tensor(windows[i:i+1, :, c]).float().unsqueeze(1).to(device)
                _, mc = model.cond_stage_model(x_c)
                ml.append(mc.cpu())
            masks = torch.stack(ml, dim=1).to(device)
            _, A = model.cross_variate_adapter(masks)
            all_A.append(A.squeeze(0).cpu().numpy())

    avg_A = np.mean(all_A, axis=0)
    np.save(f'{save_dir}/adjacency_matrix.npy', avg_A)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(avg_A, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Learned Sparse Cross-Variate Adjacency (top-k=3)')
    ax.set_xticks(range(C_var))
    ax.set_yticks(range(C_var))
    ax.set_xticklabels(variate_names, rotation=45, fontsize=9)
    ax.set_yticklabels(variate_names, fontsize=9)
    for i in range(C_var):
        for j in range(C_var):
            ax.text(j, i, f'{avg_A[i, j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if avg_A[i, j] > 0.5 else 'black')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/adjacency_graph.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: adjacency_graph.pdf')

    # ==================================================================
    # Fig 6: Results comparison — grouped bar chart
    # ==================================================================
    k_values = [3, 5, 10]
    mvdp_corr = [results[f'Time-MVDP-{k}']['corr_error'] for k in k_values]
    dp_corr = [results[f'Time-DP-{k}']['corr_error'] for k in k_values]
    mvdp_mmd = [results[f'Time-MVDP-{k}']['avg_mmd'] for k in k_values]
    dp_mmd = [results[f'Time-DP-{k}']['avg_mmd'] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - width/2, mvdp_corr, width, label='Time-MVDP', color='#2196F3')
    ax1.bar(x + width/2, dp_corr, width, label='Time-DP', color='#FF9800')
    ax1.set_xlabel('Number of Few-Shot Prompts (K)')
    ax1.set_ylabel('Correlation Error (Frobenius)')
    ax1.set_title('Cross-Variate Correlation Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'K={k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis='y')

    ax2.bar(x - width/2, mvdp_mmd, width, label='Time-MVDP', color='#2196F3')
    ax2.bar(x + width/2, dp_mmd, width, label='Time-DP', color='#FF9800')
    ax2.set_xlabel('Number of Few-Shot Prompts (K)')
    ax2.set_ylabel('Average MMD')
    ax2.set_title('Per-Variate Generation Quality (MMD)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'K={k}' for k in k_values])
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/results_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results_comparison.pdf')

    print(f'\nAll figures saved to: {fig_dir}/')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_windows, test_windows, variate_names, C_var = load_data(
        args.dataset_csv, args.seq_len, args.seed
    )
    print(f'Data: {C_var} variates, Train: {train_windows.shape}, Test: {test_windows.shape}')

    model = build_and_load_model(
        args.config, args.ckpt, args.seq_len, args.num_latents, C_var
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    n_gen = min(args.n_gen, len(train_windows) * 3)
    results = evaluate(
        model, train_windows, test_windows, C_var,
        n_gen, args.ddim_steps, device, args.save_dir
    )

    with open(f'{args.save_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print('\n' + '=' * 70)
    print('EVALUATION SUMMARY')
    print('=' * 70)
    print(f'  {"Method":<20} {"Corr Error":>12} {"Avg MMD":>12}')
    print('  ' + '-' * 48)
    for k in [3, 5, 10]:
        mvdp = results[f'Time-MVDP-{k}']
        dp = results[f'Time-DP-{k}']
        print(f'  {"Time-MVDP-"+str(k):<20} {mvdp["corr_error"]:>12.4f} {mvdp["avg_mmd"]:>12.4f}')
        print(f'  {"Time-DP-"+str(k):<20} {dp["corr_error"]:>12.4f} {dp["avg_mmd"]:>12.4f}')
        print('  ' + '-' * 48)

    # All windows for visualization
    df = pd.read_csv(args.dataset_csv)
    numeric_cols = [c for c in df.columns if c != 'date']
    data = df[numeric_cols].values.astype(np.float32)
    n_win = len(data) // args.seq_len
    all_windows = data[:n_win * args.seq_len].reshape(n_win, args.seq_len, C_var)
    for c in range(C_var):
        col = all_windows[:, :, c]
        all_windows[:, :, c] = (col - col.mean()) / (col.std() + 1e-8)

    generate_figures(
        results, model, all_windows, variate_names, C_var,
        args.save_dir, device, args.base_log_dir, args.display_k
    )

    print(f'\nAll outputs: {args.save_dir}/')


if __name__ == '__main__':
    main()