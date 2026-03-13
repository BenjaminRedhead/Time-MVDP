# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from pathlib import Path
from utils.pkl_utils import save_pkl
from metrics.metrics_sets import run_metrics, calculate_one
from ldm.data.tsg_dataset import TSGDataset
import os
import json

data_root = os.environ['DATA_ROOT']

def test_model_with_dp(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)
    model = model.cuda()
    model.eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    seq_len = data.window
    num_dp = 100  # number of samples for constructingdomain prompts
    all_metrics = {}
    for dataset in data.norm_train_dict:
        dataset_data = TSGDataset({dataset: data.norm_train_dict[dataset]})
        dataset_samples = []
        for idx in np.random.randint(dataset_data.__len__(),size=num_dp):  # randomly sample num_dp samples from the dataset
            dataset_samples.append(dataset_data.__getitem__(idx)['context'])
        dataset_samples = np.vstack(dataset_samples)
            
        x = torch.tensor(dataset_samples).to('cuda').float().unsqueeze(1)[:num_dp]
        c, mask = model.get_learned_conditioning(x, return_mask=True)
        repeats = int(1000 / num_dp) if not opt.debug else 1

        if c is None:
            mask_repeat = None
            cond = None
        elif mask is None:
            cond = torch.repeat_interleave(c, repeats, dim=0)
            mask_repeat = None
        else:
            cond = torch.repeat_interleave(c, repeats, dim=0)
            mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)

        all_gen = []
        for _ in range(5 if not opt.debug else 1):  # iterate to reduce maximum memory usage
            samples, _ = model.sample_log(cond=cond, batch_size=1000 if not opt.debug else 100, ddim=False, cfg_scale=1, mask=mask_repeat)
            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)
        generated_data = np.vstack(all_gen).transpose(0, 2, 1)
        # save data in original scale, for fairness in evaluation
        tmp_name = f'{dataset}_{seq_len}_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)
        all_metrics = run_metrics(data_name=dataset, seq_len=seq_len, model_name=tmp_name, gen_data=generated_data, scale='zscore', exist_dict=all_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'metric_dict.pkl')
    

def test_model_uncond(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)
    model = model.cuda()
    model.eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    seq_len = data.window
    all_metrics = {}
    for dataset in data.norm_train_dict:            

        all_gen = []
        for _ in range(5 if not opt.debug else 1):
            samples, _ = model.sample_log(cond=None, batch_size=1000 if not opt.debug else 100, ddim=False, cfg_scale=1)
            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)
        generated_data = np.vstack(all_gen).transpose(0, 2, 1)
        # save data in original scale. for fair use in evaluation
        tmp_name = f'{dataset}_{seq_len}_uncond_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)
        all_metrics = run_metrics(data_name=dataset, seq_len=seq_len, model_name=tmp_name, gen_data=generated_data, scale='zscore', exist_dict=all_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'metric_dict.pkl')
    
def zero_shot_k_repeat(samples, model, train_data_module, num_gen_samples=1000):
    data = train_data_module
    k_samples = samples.transpose(0,2,1)
    k = k_samples.shape[0]
    normalizer = data.fit_normalizer(k_samples)

    norm_k_samples = data.transform(k_samples, normalizer=normalizer)

    x = torch.tensor(norm_k_samples).float().to('cuda')
    c, mask = model.get_learned_conditioning(x, return_mask=True)

    repeats = int(num_gen_samples / k)
    extra = num_gen_samples - repeats * k
    
    cond = torch.repeat_interleave(c, repeats, dim=0)
    cond = torch.cat([cond, c[:extra]], dim=0)
    mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)
    mask_repeat = torch.cat([mask_repeat, mask[:extra]], dim=0)
    
    samples, z_denoise_row = model.sample_log(cond=cond, batch_size=cond.shape[0], ddim=False, cfg_scale=1, mask=mask_repeat)
    norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
    inv_samples = data.inverse_transform(norm_samples, normalizer=normalizer)
    gen_data = inv_samples.transpose(0,2,1)
    
    return gen_data, k_samples.transpose(0,2,1)

def merge_dicts(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = v
    return result

def test_model_unseen(model, data, trainer, opt, logdir):
    all_metrics = {}
    seq_len = opt.seq_len
    for data_name in ['stock', 'web']:
        data_result_dicts = []
        uni_ori_data = np.load(f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_test_sample.npy')
        if data_name == 'web':
            uni_ori_data = uni_ori_data[uni_ori_data<np.percentile(uni_ori_data,99)]
        uni_data_mean, uni_data_std = np.mean(uni_ori_data), np.std(uni_ori_data)
        uni_data_sub, uni_data_div = uni_data_mean, uni_data_std + 1e-7
        uni_scaled_ori = (uni_ori_data - uni_data_sub) / uni_data_div
        print(data_name, 'univar', uni_scaled_ori.shape)

        scaled_ori = uni_scaled_ori
        
        total_samples = 2000
        for k in [3, 10, 100]: 
            k_samples = np.load(f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_k_{k}_sample.npy')
            for i in range(1):
                gen_data, _ = zero_shot_k_repeat(k_samples, model=model, train_data_module=data, num_gen_samples=total_samples)
                np.save(logdir/f"generated_samples/{data_name}_{seq_len}_k{k}_repeat_gen.npy", gen_data)
                this_metrics = calculate_one(gen_data.squeeze(), scaled_ori.squeeze(), '', i, f"{data_name}_{k}", seq_len, uni_data_sub, uni_data_div, total_samples)
                data_result_dicts.append(this_metrics)
                

        data_metrics = merge_dicts(data_result_dicts)
        all_metrics.update(data_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'unseen_domain_metric_dict.pkl')


# ===========================================================================
# Multivariate evaluation
# ===========================================================================

def _compute_correlation_matrix(x):
    """
    Compute cross-variate correlation matrix.
    x: (N, C, T) numpy array
    Returns: (C, C) correlation matrix
    """
    N, C, T = x.shape
    # Flatten samples and time: (C, N*T)
    flat = x.transpose(1, 0, 2).reshape(C, -1)
    # Standardize per variate
    means = flat.mean(axis=1, keepdims=True)
    stds = flat.std(axis=1, keepdims=True) + 1e-8
    normed = (flat - means) / stds
    # Correlation = (1/M) X @ X^T
    corr = normed @ normed.T / normed.shape[1]
    return corr


def _compute_per_variate_mmd(gen, real, kernel='rbf', gamma=1.0):
    """
    Compute MMD per variate between generated and real data.
    gen, real: (N, C, T) numpy arrays
    Returns: list of per-variate MMD values, and average
    """
    C = gen.shape[1]
    mmds = []
    for c in range(C):
        g = gen[:, c, :]  # (N, T)
        r = real[:, c, :]  # (N, T)
        
        # RBF kernel MMD
        n_g, n_r = g.shape[0], r.shape[0]
        
        # Pairwise squared distances
        def sq_dist(a, b):
            aa = np.sum(a**2, axis=1, keepdims=True)
            bb = np.sum(b**2, axis=1, keepdims=True)
            return aa + bb.T - 2 * a @ b.T
        
        Kgg = np.exp(-gamma * sq_dist(g, g))
        Krr = np.exp(-gamma * sq_dist(r, r))
        Kgr = np.exp(-gamma * sq_dist(g, r))
        
        mmd = Kgg.sum() / (n_g * n_g) + Krr.sum() / (n_r * n_r) - 2 * Kgr.sum() / (n_g * n_r)
        mmds.append(float(max(mmd, 0)))
    
    return mmds, float(np.mean(mmds))


def test_model_multivariate(model, data, trainer, opt, logdir):
    """
    Evaluate multivariate generation quality.
    
    Metrics:
        - Per-variate MMD: how well each channel's marginal distribution matches
        - Correlation matrix error: Frobenius norm between generated and real
          cross-variate correlation matrices
        - Joint MMD: MMD on the flattened (C*T)-dimensional vectors
    """
    # Load best checkpoint if available
    if hasattr(trainer, 'callbacks') and len(trainer.callbacks) > 0:
        if hasattr(trainer.callbacks[-1], 'best_model_path') and trainer.callbacks[-1].best_model_path:
            best_ckpt_path = trainer.callbacks[-1].best_model_path
            print(f"[Multivariate] Loading best model from {best_ckpt_path}")
            model.init_from_ckpt(best_ckpt_path)
    
    model = model.cuda()
    model.eval()
    
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    
    seq_len = data.window
    all_metrics = {}
    
    for dataset_name in data.key_list:
        print(f"\n[Multivariate] Evaluating {dataset_name}...")
        
        # Get normalized train data for few-shot prompts: (N, T, C)
        train_data = data.norm_train_dict[dataset_name]
        val_data = data.norm_val_dict[dataset_name]
        C_var = train_data.shape[2]
        
        # Few-shot prompts
        K = min(10, train_data.shape[0])
        prompt_indices = np.random.choice(train_data.shape[0], K, replace=False)
        few_shot = torch.tensor(train_data[prompt_indices]).float()  # (K, T, C)
        
        # Real data for comparison
        n_eval = min(val_data.shape[0], 200)
        real_samples = val_data[:n_eval]  # (N_eval, T, C)
        real_samples_ct = np.transpose(real_samples, (0, 2, 1))  # (N_eval, C, T)
        
        # Generate multivariate samples
        n_gen = min(n_eval, 100) if not opt.debug else 10
        few_shot_batch = {'context': few_shot}
        
        with torch.no_grad():
            gen_samples = model.sample_multivariate(
                few_shot_batch,
                n_samples=n_gen,
                ddim_steps=opt.ddim_steps,
                eta=1.0,
            )  # (n_gen, C, T) tensor
        
        gen_np = gen_samples.cpu().numpy()  # (n_gen, C, T)
        
        # Save generated data
        np.save(save_path / f'{dataset_name}_{seq_len}_mv_gen.npy', gen_np)
        
        # --- Metrics ---
        
        # 1. Per-variate MMD
        real_for_mmd = real_samples_ct[:n_gen]
        per_var_mmd, avg_mmd = _compute_per_variate_mmd(gen_np, real_for_mmd)
        
        # 2. Correlation matrix error
        corr_gen = _compute_correlation_matrix(gen_np)
        corr_real = _compute_correlation_matrix(real_for_mmd)
        corr_error = float(np.linalg.norm(corr_gen - corr_real, 'fro'))
        
        # 3. Joint MMD (flatten all variates)
        gen_flat = gen_np.reshape(gen_np.shape[0], -1)  # (N, C*T)
        real_flat = real_for_mmd.reshape(real_for_mmd.shape[0], -1)
        # Use smaller gamma for high-dim
        gamma_joint = 1.0 / (C_var * seq_len)
        joint_mmds, joint_mmd_avg = _compute_per_variate_mmd(
            gen_flat[:, np.newaxis, :], real_flat[:, np.newaxis, :], gamma=gamma_joint
        )
        joint_mmd = joint_mmds[0]
        
        metrics = {
            'avg_variate_mmd': avg_mmd,
            'per_variate_mmd': per_var_mmd,
            'correlation_error': corr_error,
            'joint_mmd': joint_mmd,
        }
        all_metrics[dataset_name] = metrics
        
        print(f"  Avg Variate MMD: {avg_mmd:.6f}")
        print(f"  Correlation Error: {corr_error:.6f}")
        print(f"  Joint MMD: {joint_mmd:.6f}")
        print(f"  Per-variate MMDs: {[f'{m:.6f}' for m in per_var_mmd]}")
        
        # Save correlation matrices for visualization
        np.save(save_path / f'{dataset_name}_{seq_len}_corr_gen.npy', corr_gen)
        np.save(save_path / f'{dataset_name}_{seq_len}_corr_real.npy', corr_real)
    
    # Save all metrics
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj
    
    results_path = Path(logdir) / 'multivariate_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(to_serializable(all_metrics), f, indent=2)
    
    save_pkl(all_metrics, Path(logdir) / 'multivariate_metric_dict.pkl')
    
    print(f"\n[Multivariate] Results saved to {results_path}")
    print(f"[Multivariate] Full metrics: {json.dumps(to_serializable(all_metrics), indent=2)}")
    
    return all_metrics