#!/usr/bin/env python
# Created for internship coding assessment
"""
Phase 2: Fine-tune multivariate adapters on Exchange rate data.

Loads a pretrained univariate TimeDP checkpoint, attaches cross-variate
and copula adapters (51K params), freezes the 27M base model, and trains
only the adapters on 8-currency Exchange data.

Records wall-clock training time and per-step loss history.

Usage:
    python train_adapters.py \
        --base_ckpt /path/to/last.ckpt \
        --dataset_csv /path/to/exchange_rate.csv \
        --save_dir /path/to/output \
        --n_steps 5000
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

os.environ.setdefault('DATA_ROOT', '/tmp')
os.environ['WANDB_MODE'] = 'disabled'

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train multivariate adapters')
    parser.add_argument('--base_ckpt', type=str, required=True,
                        help='Path to pretrained univariate TimeDP checkpoint')
    parser.add_argument('--dataset_csv', type=str, required=True,
                        help='Path to exchange_rate.csv')
    parser.add_argument('--config', type=str, default='configs/multi_domain_timedp.yaml',
                        help='Path to base config YAML')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--adapter_top_k', type=int, default=3)
    parser.add_argument('--adapter_d_model', type=int, default=64)
    parser.add_argument('--adapter_n_heads', type=int, default=4)
    parser.add_argument('--copula_d_model', type=int, default=64)
    parser.add_argument('--copula_n_heads', type=int, default=4)
    parser.add_argument('--corr_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--val_every', type=int, default=500)
    parser.add_argument('--val_portion', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_exchange_data(csv_path, seq_len, val_portion, seed):
    """Load Exchange CSV, window, normalize, split."""
    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.columns if c != 'date']
    data = df[numeric_cols].values.astype(np.float32)
    C_var = data.shape[1]

    n_win = len(data) // seq_len
    windows = data[:n_win * seq_len].reshape(n_win, seq_len, C_var)

    # Per-variate z-score
    norm_params = {}
    for c in range(C_var):
        col = windows[:, :, c]
        mean, std = float(col.mean()), float(col.std() + 1e-8)
        norm_params[c] = {'mean': mean, 'std': std}
        windows[:, :, c] = (col - mean) / std

    # Split
    np.random.seed(seed)
    indices = np.random.permutation(len(windows))
    n_val = max(int(len(windows) * val_portion), 3)
    train = windows[indices[:-n_val]]
    val = windows[indices[-n_val:]]

    return train, val, norm_params, numeric_cols, C_var


def build_model(config_path, seq_len, num_latents, C_var, adapter_kwargs):
    """Instantiate LatentDiffusion with multivariate adapter config."""
    config = OmegaConf.load(config_path)
    config.model['params']['seq_len'] = seq_len
    config.model['params']['unet_config']['params']['seq_len'] = seq_len
    config.model['params']['cond_stage_config']['params']['window'] = seq_len
    config.model['params']['cond_stage_config']['params']['num_latents'] = num_latents
    config.model['params']['unet_config']['params']['latent_unit'] = num_latents
    config.model['params'].pop('ckpt_path', None)

    # Adapter config
    config.model['params']['n_variates'] = C_var
    for k, v in adapter_kwargs.items():
        config.model['params'][k] = v

    return instantiate_from_config(config.model)


def main():
    args = parse_args()

    os.makedirs(f'{args.save_dir}/checkpoints', exist_ok=True)

    # ---- Data ----
    print('Loading Exchange data...')
    train_windows, val_windows, norm_params, variate_names, C_var = load_exchange_data(
        args.dataset_csv, args.seq_len, args.val_portion, args.seed
    )
    print(f'  {C_var} currencies, Train: {train_windows.shape}, Val: {val_windows.shape}')

    # Save metadata
    with open(f'{args.save_dir}/norm_params.json', 'w') as f:
        json.dump(norm_params, f)
    np.save(f'{args.save_dir}/variate_names.npy', np.array(variate_names))

    # ---- Model ----
    print('Building model...')
    adapter_kwargs = {
        'adapter_top_k': args.adapter_top_k,
        'adapter_d_model': args.adapter_d_model,
        'adapter_n_heads': args.adapter_n_heads,
        'copula_d_model': args.copula_d_model,
        'copula_n_heads': args.copula_n_heads,
        'corr_loss_weight': args.corr_loss_weight,
    }
    model = build_model(args.config, args.seq_len, args.num_latents, C_var, adapter_kwargs)

    # Load pretrained base
    print(f'Loading pretrained base: {args.base_ckpt}')
    model.init_from_ckpt(args.base_ckpt)

    # Freeze base, keep adapters trainable
    model.freeze_base_model()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total: {n_total:,} | Trainable: {n_trainable:,} ({100*n_trainable/n_total:.2f}%)')

    # ---- Optimizer ----
    model.learning_rate = args.lr
    opt_result = model.configure_optimizers()
    if isinstance(opt_result, (list, tuple)):
        optimizer = opt_result[0][0] if isinstance(opt_result[0], list) else opt_result[0]
        scheduler = opt_result[1][0]['scheduler'] if len(opt_result) > 1 else None
    else:
        optimizer = opt_result
        scheduler = None

    # ---- Training loop ----
    B = args.batch_size
    model.train()
    best_val_loss = float('inf')
    history = {'loss': [], 'denoise': [], 'adapter': [], 'copula': [], 'wall_time': []}

    print(f'\nTraining {args.n_steps} steps (batch={B}, lr={args.lr})')
    print(f'  {"Step":>5} | {"loss":>8} | {"denoise":>8} | {"adapter":>8} | {"copula":>8} | {"cv_grad":>8} | {"cop_grad":>8}')
    print('  ' + '-' * 70)

    t_start = time.time()

    for step in range(1, args.n_steps + 1):
        idx = np.random.randint(0, len(train_windows), size=B)
        batch = {
            'context': torch.tensor(train_windows[idx]).float().to(device),
            'data_key': torch.zeros(B, dtype=torch.long).to(device),
        }

        optimizer.zero_grad()
        loss, loss_dict = model.shared_step(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Record
        prefix = 'train'
        d = loss_dict.get(f'{prefix}/denoise_loss', torch.tensor(0)).item()
        a = loss_dict.get(f'{prefix}/adapter_loss', torch.tensor(0)).item()
        c = loss_dict.get(f'{prefix}/copula_loss', torch.tensor(0)).item()
        history['loss'].append(loss.item())
        history['denoise'].append(d)
        history['adapter'].append(a)
        history['copula'].append(c)
        history['wall_time'].append(time.time() - t_start)

        if step % args.log_every == 0:
            cv_g = sum(p.grad.norm().item() for p in model.cross_variate_adapter.parameters() if p.grad is not None)
            cop_g = sum(p.grad.norm().item() for p in model.copula_adapter.parameters() if p.grad is not None)
            elapsed = time.time() - t_start
            print(f'  {step:5d} | {loss.item():8.4f} | {d:8.4f} | {a:8.4f} | {c:8.4f} | {cv_g:8.5f} | {cop_g:8.5f}  [{elapsed:.0f}s]')

        if step % args.val_every == 0:
            model.eval()
            val_losses = []
            for vi in range(0, len(val_windows), B):
                vb = min(B, len(val_windows) - vi)
                vbatch = {
                    'context': torch.tensor(val_windows[vi:vi+vb]).float().to(device),
                    'data_key': torch.zeros(vb, dtype=torch.long).to(device),
                }
                with torch.no_grad():
                    vloss, _ = model.shared_step(vbatch)
                val_losses.append(vloss.item())
            val_loss = np.mean(val_losses)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'step': step,
                    'val_loss': val_loss,
                }, f'{args.save_dir}/checkpoints/best.ckpt')
            tag = ' [saved]' if improved else ''
            print(f'  >>> Val: {val_loss:.4f} (best: {best_val_loss:.4f}){tag}')
            model.train()

    total_time = time.time() - t_start

    # Save final
    torch.save({
        'state_dict': model.state_dict(),
        'step': args.n_steps,
    }, f'{args.save_dir}/checkpoints/last.ckpt')

    for k, v in history.items():
        np.save(f'{args.save_dir}/train_{k}.npy', np.array(v))

    # Save timing info
    timing = {
        'adapter_train_seconds': total_time,
        'adapter_train_steps': args.n_steps,
        'adapter_params': n_trainable,
        'base_params': n_total - n_trainable,
        'total_params': n_total,
        'seconds_per_step': total_time / args.n_steps,
    }
    with open(f'{args.save_dir}/adapter_timing.json', 'w') as f:
        json.dump(timing, f, indent=2)

    print(f'\nAdapter training complete in {total_time:.1f}s ({total_time/60:.1f} min)')
    print(f'  {total_time/args.n_steps:.3f} s/step')
    print(f'  Output: {args.save_dir}/')


if __name__ == '__main__':
    main()