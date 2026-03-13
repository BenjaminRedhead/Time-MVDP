# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from pathlib import Path
from ldm.data.catsg_dataset import CaTSGDataset
import os
import time
import csv
from datetime import datetime
from utils.metrics.feature_distance_eval import get_mdd_eval, get_flat_distance, mmd_metric
from utils.metrics.catsg_eval import get_jftsd

@torch.no_grad()
def sinkhorn(scores, eps=0.05, niters=3):
    """
    Sinkhorn algorithm for optimal transport
    scores: (B, K)  ->  Q: (B, K), balanced assignment
    """
    Q = torch.exp(scores / eps).t()          # (K, B)
    Q /= Q.sum()                             
    K, B = Q.shape
    r = torch.ones(K, device=Q.device) / K
    c = torch.ones(B, device=Q.device) / B
    for _ in range(niters):
        u = Q.sum(dim=1)
        Q *= (r / (u + 1e-12)).unsqueeze(1)
        Q *= (c / (Q.sum(dim=0) + 1e-12)).unsqueeze(0)
    Q = Q / Q.sum(dim=0, keepdim=True)
    return Q.t()  # (B, K)


def swav_loss_from_scores(scores_list, is_global, eps=0.05, temp=0.1):
    """
    SwAV loss from scores
    scores_list: [scores_v], each (B, K) from different views
    is_global:   bool list aligned with scores_list, marking "global views" for computing q

    Referece:
        - SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, ECCV 2020
    """
    V = len(scores_list)
    p_list = [torch.softmax(s/ temp, dim=-1) for s in scores_list]

    # Only compute q for global views
    q_list = [sinkhorn(s, eps) if g else None for s, g in zip(scores_list, is_global)]

    loss = 0.0
    n_terms = 0
    # swapped prediction: use v's p to predict u's q (u must be global and u!=v)
    for u in range(V):
        if q_list[u] is None: 
            continue
        q_u = q_list[u].detach()              # stop gradient
        for v in range(V):
            if v == u: 
                continue
            p_v = p_list[v]
            loss -= (q_u * torch.log(p_v + 1e-12)).sum(dim=-1).mean()
            n_terms += 1
    loss = loss / max(1, n_terms)

    with torch.no_grad():
        avg_codes = torch.stack([q for q in q_list if q is not None], 0).mean(0)  # (B,K)->(K,)
        proto_usage = avg_codes.mean(0)  # (K,)
    balance_reg = -(proto_usage * (proto_usage + 1e-12).log()).sum()  # higher = more balanced
    return loss, balance_reg

def _find_checkpoint_parent(logdir):
    """Find checkpoint directory from logdir"""
    logdir_path = Path(logdir)
    
    if (logdir_path / "checkpoints").exists():
        return logdir_path
    else:
        model_dirs = list(logdir_path.parent.glob("*"))
        checkpoint_dirs = [d for d in model_dirs if d.is_dir() and (d / "checkpoints").exists()]
        
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: x.name)
            return checkpoint_dirs[-1]
        else:
            return logdir_path

def _generate_env_prob_files(model, val_data, opt, env_stats_dir, train_env_prob_path, val_env_prob_path):
    """Generate train_env_prob.npy and val_env_prob.npy"""
    env_stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train dataset
    split_method = getattr(opt, 'split_method', 'default')
    
    # Handle synthetic datasets that store directly without split method subdirectory
    if opt.dataset in ['harmonic_vm', 'harmonic_vp']:
        dataset_path = f"./dataset/{opt.dataset}"
    else:
        dataset_path = f"./dataset/{opt.dataset}/{split_method}"
    
    train_dataset = CaTSGDataset(
        x_path=f"{dataset_path}/x_train.npy",
        c_path=f"{dataset_path}/c_train.npy",
        window=val_data.window,
        normalize=val_data.normalize,
        norm_type=val_data.norm_type,
        dataset_name=opt.dataset,
        split_method=split_method
    )
    
    # Apply same scalers as val_data
    train_dataset.c_scaler = val_data.c_scaler
    train_dataset.x_scaler = val_data.x_scaler
    
    # Generate train_env_prob.npy
    print(f"[ENV_PROB] Generating train_env_prob.npy...")
    _, train_env_prob, _ = collect_full_dataset_env_prob(model, train_dataset, stage="train", batch_size=32, save_dir=env_stats_dir.parent)
    np.save(train_env_prob_path, train_env_prob)
    print(f"Saved train_env_prob.npy: {train_env_prob.shape}")
    
    # Generate val_env_prob.npy  
    print(f"[ENV_PROB] Generating val_env_prob.npy...")
    _, val_env_prob, _ = collect_full_dataset_env_prob(model, val_data, stage="val", batch_size=32, save_dir=env_stats_dir.parent)
    np.save(val_env_prob_path, val_env_prob)
    print(f"Saved val_env_prob.npy: {val_env_prob.shape}")

from pytorch_lightning.callbacks import ModelCheckpoint
def test_prepare(model, val_data, test_data, trainer, opt, logdir, task_type='int', config=None):
    # load best checkpoint
    ckpt_cb = next(cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint))
    best_ckpt_path = Path(getattr(ckpt_cb, "dirpath", None)) / 'last.ckpt'
    model.init_from_ckpt(best_ckpt_path)

    sampling_seeds = None
    generation_config = {}
    
    if config and hasattr(config, 'sampling'):
        sampling_seeds = getattr(config.sampling, 'seeds', None)
        num_repeat = getattr(config.sampling, 'num_repeat', 5)
    else:
        num_repeat = 5
    
    # Read generation_config for task-specific sampling parameters
    if config and hasattr(config, 'generation_config'):
        generation_config = config.generation_config
        print(f"[Config] Found generation_config: {generation_config}")
    else:
        print(f"[Config] No generation_config found, using model defaults")

    model = model.cuda().eval()
    
    # Generate env_prob files if they don't exist
    checkpoint_parent = _find_checkpoint_parent(logdir)
    env_stats_dir = checkpoint_parent / "env_stats"
    train_env_prob_path = env_stats_dir / "train_env_prob.npy"
    val_env_prob_path = env_stats_dir / "val_env_prob.npy"

    if not train_env_prob_path.exists() or not val_env_prob_path.exists():
        _generate_env_prob_files(model, val_data, opt, env_stats_dir, train_env_prob_path, val_env_prob_path)

    save_path = checkpoint_parent
    print(f"Test results will be saved to: {save_path}")

    seq_len = getattr(test_data, "window", 96)

    if hasattr(test_data, "transform"):
        norm_test_data = test_data.transform(test_data)
    else:
        norm_test_data = test_data

    c = torch.tensor(np.stack([norm_test_data[idx]['c'] for idx in range(len(norm_test_data))])).float().cuda()
    c = model.prepare_c(c)

    if task_type == 'cf':
        n_test_samples = len(test_data)
        env_stats_dir = save_path / "env_stats"
        val_env_prob_files = env_stats_dir / "val_env_prob.npy"
        env_prob_val = np.load(val_env_prob_files)
        print(f"[CF] Loaded val_env_prob from: {val_env_prob_files}")
        print(f"[CF] Loaded env_prob shape: {env_prob_val.shape}")

        subset_val_indices = np.random.choice(len(env_prob_val), n_test_samples, replace=False)
        env_prob_val_subset = env_prob_val[subset_val_indices]
        
        # Extract val_data subset manually since Dataset doesn't support fancy indexing
        val_data_subset = []
        for idx in subset_val_indices:
            val_data_subset.append(val_data[idx])
        # Convert to consistent format
        val_data_subset = {
            'c': np.stack([sample['c'] for sample in val_data_subset]),
            'x': np.stack([sample['x'] for sample in val_data_subset])
        }
    else:
        env_prob_val_subset = None
        val_data_subset = None

    para_dict = {
        'seq_len': seq_len,
        'num_repeat': num_repeat,
        'save_path': save_path,
        'env_prob_val_subset': env_prob_val_subset,
        'val_data_subset': val_data_subset,
        'generation_config': generation_config,
    }
    
    # Add sampling_seeds if available
    if sampling_seeds:
        para_dict['sampling_seeds'] = sampling_seeds
    
    return model, c, para_dict

def sample_interventional(model, c, para):
    all_gen = []
    
    model.env_manager.set_stage("test")
    print(f"[Generation] Set env_manager to test stage for env_prob collection")
    
    # Get generation config for int task
    generation_config = para.get('generation_config', {})
    int_config = generation_config.get('int_generation', {}) if generation_config else {}
    
    # Get sampling parameters from config
    use_ddim = int_config.get('ddim', False)
    dpm_solver_steps = int_config.get('dpm_solver_steps', None)
    
    print(f"[Config] Int generation config: ddim={use_ddim}, dpm_solver_steps={dpm_solver_steps}")
    
    # Get sampling configuration: either seeds list or num_repeat
    sampling_seeds = para.get('sampling_seeds', None)
    if sampling_seeds:
        sampling_list = sampling_seeds
        print(f"Using explicit seeds for sampling: {sampling_list}")
    else:
        # Fallback to num_repeat with generated seeds
        num_repeat = para.get('num_repeat', 3)
        sampling_list = [42 + i for i in range(num_repeat)]
        print(f"Using num_repeat={num_repeat} with generated seeds: {sampling_list}")
    
    for i, seed in enumerate(sampling_list):
        print(f"Sampling round {i+1}/{len(sampling_list)} (seed={seed})")
        
        torch.manual_seed(seed)
        np.random.seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        
        test_batch_size = c.shape[0]
        start_time = time.time()
        
        # Build sampling arguments from config
        sample_kwargs = {'c': c, 'batch_size': test_batch_size, 'ddim': use_ddim, 'mask': None}
        if dpm_solver_steps is not None:
            # Override model's default DPM solver steps
            model.dpm_solver_steps = dpm_solver_steps
            print(f"[Sampling] Overriding model.dpm_solver_steps: {dpm_solver_steps}")
            # Use same value for ddim_steps as reference
            sample_kwargs['ddim_steps'] = dpm_solver_steps
        
        samples, _ = model.sample_log(**sample_kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Sampling success!! Runtime: {runtime:.3f}s")
        
        samples = samples.cpu().numpy() 
        all_gen.append(samples.transpose(0, 2, 1))

    return all_gen

def collect_full_dataset_env_prob(model, dataset, stage="test", batch_size=64, save_dir=None):
    print(f"[Full Dataset] Collecting env_prob from full {stage} dataset ({len(dataset)} samples)")
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    model.env_manager.set_stage(stage)
    
    all_env_scores, all_z_envs = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            # Handle different batch formats
            if isinstance(batch, dict):
                x, c = batch['x'], batch['c']
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, c = batch[0], batch[1]
                
            # Fix x dimensions: (B, T, 1) -> (B, 1, T) to match env_manager expectations
            if x.dim() == 3 and x.shape[2] == 1:
                x = x.transpose(1, 2)  # (B, T, 1) -> (B, 1, T)
            
            x = torch.as_tensor(x, device=model.device)
            c = torch.as_tensor(c, device=model.device)
            c = model.prepare_c(c) 

            # Use both c and x for complete env_prob calculation as in training
            env_outputs = model.env_manager.env_prob_infer(c.shape[0], c, x=x)
            env_scores = env_outputs['scores']
            z_envs = env_outputs['z_env']
            all_env_scores.append(env_scores.cpu().numpy())
            all_z_envs.append(z_envs.cpu().numpy())
    
    # Combine all env_probs
    S_all = torch.from_numpy(np.concatenate(all_env_scores, 0))  # (N,K)
    P_global = torch.softmax(S_all / 0.1, dim=-1).cpu().numpy()
    Q_global = sinkhorn(S_all).cpu().numpy()               # (N,K)
    
    all_z_envs = torch.from_numpy(np.concatenate(all_z_envs, 0))

    from pathlib import Path
    env_stats_dir = Path(save_dir) / "env_stats"
    env_stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(env_stats_dir / f"{stage}_env_q.npy", Q_global)
    np.save(env_stats_dir / f"{stage}_env_prob.npy", P_global)
    print(f"Saved {env_stats_dir} with shape {Q_global.shape}")
    
    return Q_global, P_global, all_z_envs

def aggregate_repeat_metrics(all_repeat_metrics, model_name, dataset, task_type):
    """
    Aggregate metrics across multiple repeats, computing mean ± std
    Return single aggregated result with total repeat count
    """
    if not all_repeat_metrics:
        return {}
    
    # Extract all metric names from the first repeat
    first_repeat = all_repeat_metrics[0]
    if not first_repeat:
        return {}
    
    # Get the key format from the first repeat
    sample_key = list(first_repeat.keys())[0]
    key_prefix = (model_name, dataset, sample_key[2])  # (model_name, dataset, seq_len)
    
    # Find all metric names across all repeats for aggregation
    all_metric_names = set()
    for repeat_metrics in all_repeat_metrics:
        for key, metrics in repeat_metrics.items():
            all_metric_names.update(metrics.keys())
    
    # Create aggregated metrics
    aggregated_dict = {}
    for metric_name in all_metric_names:
        metric_values = []
        for repeat_metrics in all_repeat_metrics:
            for key, metrics in repeat_metrics.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_values.append(value)
                    elif isinstance(value, np.ndarray):
                        # Handle numpy arrays
                        scalar_val = float(value.item()) if value.size == 1 else float(value.mean())
                        if not np.isnan(scalar_val):
                            metric_values.append(scalar_val)
                    break
        
        if metric_values:
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            # Format as mean ± std
            if std_val > 0:
                aggregated_value = f"{mean_val:.6f} ± {std_val:.6f}"
            else:
                aggregated_value = f"{mean_val:.6f} ± 0.000000"
            
            aggregated_dict[metric_name] = aggregated_value
            # Also store raw values for merge_results.py processing
            aggregated_dict[f'{metric_name}_mean'] = mean_val
            aggregated_dict[f'{metric_name}_std'] = std_val
            aggregated_dict[f'{metric_name}_raw_values'] = metric_values
    
    # Store aggregated metrics with total repeat count
    total_repeats = len(all_repeat_metrics)
    aggregated_metrics = {}
    if aggregated_dict:
        # Use total repeat count as the repeat_id in the key
        aggregated_key = (*key_prefix, total_repeats)
        aggregated_metrics[aggregated_key] = aggregated_dict
    
    return aggregated_metrics

def log_and_save_metrics(generated_data, test_data, c, logdir, para, task_type, dataset, device, extra_metrics={}, config=None, opt=None):

    tmp_name = f"{dataset}_{para['seq_len']}_generation"
    
    generation_dir = para['save_path'] / "generation" / task_type
    generation_dir.mkdir(parents=True, exist_ok=True)
    
    if task_type == 'cf_harmonic':
        # For cf_harmonic, use ground truth counterfactual data for comparison
        real_data = test_data
    else:
        # Standard case: use test data
        real_data = np.stack([test_data[idx]['x'] for idx in range(len(test_data))])

    # Calculate metrics for each repeat
    all_repeat_metrics = []
    for repeat_idx, gen_data in enumerate(generated_data):

        np.save(generation_dir / f"{tmp_name}_repeat_{repeat_idx}.npy", gen_data)
        print(f"[Metrics] Saved normalized generated data for repeat {repeat_idx}")

        metric_input = {
            'gen_data': gen_data,
            'real_data': real_data,
            'task_type': task_type,
            'dataset': dataset,
            'exist_dict': {},
            'scale': 'raw',
            'model_name': tmp_name,
            'device': device,
            'repeat_id': repeat_idx,
            'metirc_data': {'c_data': c},
            'config': config
        }

        if task_type in ['int', 'cf', 'cf_harmonic']:
            metric_input['metirc_data'].update(extra_metrics)

        repeat_metrics = compute_metrics(**metric_input)
        all_repeat_metrics.append(repeat_metrics)
    
    # Aggregate metrics across repeats
    aggregated_metrics = aggregate_repeat_metrics(all_repeat_metrics, tmp_name, dataset, task_type)
    
    print(f"Aggregated metrics for {dataset}: {aggregated_metrics}")
    results_path = getattr(opt, 'results_path', './results') if opt else './results'
    save_metrics_to_csv(aggregated_metrics, logdir, task_type=task_type, results_path=results_path)
    return aggregated_metrics


def test_model_catsg(model, val_data, test_data, trainer, opt, logdir, dataset='default', task_type='int', config=None):
    # Prepare model and data for testing
    model, c, para = test_prepare(model, val_data, test_data, trainer, opt, logdir, task_type, config)
    model.task_type = task_type
    batch_size = getattr(opt, 'batch_size', 32)
    model.env_manager.set_stage("test")
    _, test_env_prob, _ = collect_full_dataset_env_prob(model, test_data, stage="test", batch_size=batch_size, save_dir=para['save_path'])
    
    extra = {}
    if task_type == 'int':
        # Standard intervention generation
        gen_data = sample_interventional(model, c, para)
        
        checkpoint_parent = para['save_path'] 
        env_stats_dir = checkpoint_parent / "env_stats"
        env_stats_dir.mkdir(parents=True, exist_ok=True)
        env_prob_path = env_stats_dir / "test_env_prob.npy"
        
        # Save full_env_prob to the defined path
        np.save(env_prob_path, test_env_prob)
        model.env_manager._sample_env_stats[f"test_{trainer.current_epoch}"] = test_env_prob
        env_bank = model.env_manager.env_bank.detach().cpu().numpy()
        
        extra = {'env_prob': test_env_prob, 'env_bank': env_bank}

    elif task_type == 'cf':
        # Counterfactual testing logic (integrated from test_model_cf)
        print("[CF] Starting counterfactual sampling...")
        
        # Step 1: Load val_env_prob
        print("[CF] Loading val_env_prob...")
        env_prob_val_subset = para.get('env_prob_val_subset', None) 
        val_data_subset = para.get('val_data_subset', None)

        # Create CF generation directory for subset files
        cf_generation_dir = para['save_path'] / "generation" / "cf"
        cf_generation_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(cf_generation_dir / "val_env_prob_subset.npy", env_prob_val_subset)
        np.save(cf_generation_dir / "val_data_subset.npy", val_data_subset)

        # Step 2: Set back to test mode and generate counterfactual samples
        model.env_manager.set_stage("test")
        
        # Step 3: Generate counterfactual samples
        env_bank = model.env_manager.env_bank.detach().cpu().numpy()
        print(f"Generating CF samples...")
        gen_data = sample_counterfactual(
            model=model,
            c_test=c,
            env_prob=torch.tensor(env_prob_val_subset).float().cuda(),
            para=para
        )

        extra = {'val_data': val_data_subset, 
                 'val_env_prob': env_prob_val_subset,
                 'test_env_prob': test_env_prob,
                 'env_bank': env_bank}

    elif task_type == 'cf_harmonic':
        # Special counterfactual testing for harmonic datasets
        print("[CF_HARMONIC] Starting counterfactual testing for harmonic datasets...")
        cf_generation_dir = para['save_path'] / "generation" / "cf_harmonic"
        cf_generation_dir.mkdir(parents=True, exist_ok=True)
        
        # Load counterfactual conditioning data (c_cf.npy)
        data_dir = Path(opt.data_dir) if hasattr(opt, 'data_dir') and opt.data_dir else Path('./dataset') / dataset
        c_cf_path = data_dir / 'c_cf.npy'
        if not c_cf_path.exists():
            raise FileNotFoundError(f"Counterfactual conditioning data not found: {c_cf_path}")
        c_cf = np.load(c_cf_path)
        
        #  Set to test mode and generate counterfactual samples using c_cf
        model.env_manager.set_stage("test")
        
        # Generate counterfactual samples with counterfactual conditioning. we need c' and p(e) and e
        env_bank = model.env_manager.env_bank.detach().cpu().numpy()
        
        c_cf_tensor = torch.tensor(c_cf, dtype=torch.float32).to(model.device)
        c_cf_conditioned = model.prepare_c(c_cf_tensor)  # Apply conditioning encoder
        
        gen_data = sample_counterfactual(
            model=model,
            c_test=c_cf_conditioned,  # c'
            env_prob=torch.tensor(test_env_prob).float().cuda(), # (x,c) -> p(e)
            para=para
        )

        # Step 5: Load ground truth counterfactual data for comparison
        x_cf_path = data_dir / 'x_cf.npy'
        x_cf_gt = np.load(x_cf_path)
        
        extra = {'env_prob': test_env_prob, 'env_bank': env_bank}
        test_data = x_cf_gt
        c = c_cf_tensor
        

    all_metrics = log_and_save_metrics(
        generated_data=gen_data,
        test_data=test_data,
        c=c,
        logdir=logdir,
        para=para,
        task_type=task_type,
        dataset=dataset,
        device=model.device,
        extra_metrics=extra,
        config=config,
        opt=opt
    )
    
    return all_metrics

def compute_metrics(gen_data, real_data, metirc_data, exist_dict, task_type, dataset='default', scale='zscore', repeat_id=0, model_name='CaTSG',device=None, train_x=None, config=None):

    if task_type == 'int' or  task_type =='cf_harmonic':
        # Check if environment data is available
        env_bank = metirc_data.get('env_bank', None)
        env_prob = metirc_data.get('env_prob', None)
        
        exist_dict = update_metrics_int(
            exist_dict,
            key=model_name,
            data_name=dataset, 
            seq_len=real_data.shape[1],
            ori_data=real_data,
            gen_data=gen_data,
            repeat_id=repeat_id,
            device=device,
            c_data=metirc_data['c_data'],
            env_bank=env_bank,
            env_prob=env_prob
        )

    elif task_type == 'cf':
        # Check if environment data is available
        env_bank = metirc_data.get('env_bank', None)
        val_data = metirc_data.get('val_data', None)
        val_env_prob = metirc_data.get('val_env_prob', None)
        test_env_prob = metirc_data.get('test_env_prob', None)
                
        exist_dict = update_metrics_cf(
            exist_dict,
            key=model_name,
            data_name=dataset, 
            seq_len=real_data.shape[1], 
            test_x_data=real_data,  # test data
            test_c_data=metirc_data['c_data'], # test data
            test_env_prob=test_env_prob,
            env_bank=env_bank,
            val_data=val_data,
            val_env_prob=val_env_prob,
            gen_data=gen_data,
            repeat_id=repeat_id,
            device=device,
            config=config
        )
    
    return exist_dict




def sample_counterfactual(model, c_test, env_prob, para):
    all_gen = []
    model.env_manager.set_stage("test")
    
    # Get generation config for cf task
    generation_config = para.get('generation_config', {})
    cf_config = generation_config.get('cf_generation', {}) if generation_config else {}
    
    # Get sampling parameters from config
    use_ddim = cf_config.get('ddim', False)
    dpm_solver_steps = cf_config.get('dpm_solver_steps', None)
    
    print(f"[Config] CF generation config: ddim={use_ddim}, dpm_solver_steps={dpm_solver_steps}")
    
    # Get sampling configuration: either seeds list or num_repeat
    sampling_seeds = para.get('sampling_seeds', None)
    if sampling_seeds:
        sampling_list = sampling_seeds
        print(f"Using explicit seeds for sampling: {sampling_list}")
    else:
        # Fallback to num_repeat with generated seeds
        num_repeat = para.get('num_repeat', 3)
        sampling_list = [42 + i for i in range(num_repeat)]
        print(f"Using num_repeat={num_repeat} with generated seeds: {sampling_list}")

    print("[CF] Starting counterfactual generation...")
    
    # Use fixed environment probabilities directly
    all_gen = []
    for i, seed in enumerate(sampling_list):
        import torch
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"[CF] Counterfactual sampling round {i+1}")
        
        # Build sampling arguments from config
        sample_kwargs = {'cond': c_test, 'env_prob': env_prob, 'batch_size': c_test.shape[0]}
        if dpm_solver_steps is not None:
            # Override model's default DPM solver steps
            model.dpm_solver_steps = dpm_solver_steps
            print(f"[CF Sampling] Overriding model.dpm_solver_steps: {dpm_solver_steps}")
            # Use same value for ddim_steps as reference
            sample_kwargs['ddim_steps'] = dpm_solver_steps
        
        gen_samples = model.sample_counterfactual(**sample_kwargs)
        
        # Create CF-specific generation directory
        cf_generation_dir = para['save_path'] / "generation" / "cf"
        cf_generation_dir.mkdir(parents=True, exist_ok=True)
        save_path = cf_generation_dir / f'cf_generation_{i}.npy'
        np.save(save_path, gen_samples)

        print(f"[CF] Sampled shape: {gen_samples.shape}")
        all_gen.append(gen_samples)
    
    return all_gen

def update_metrics_cf(the_dict, key, data_name, seq_len,  gen_data, 
                     test_c_data=None, test_x_data=None, test_env_prob=None, 
                     env_bank=None, repeat_id=0, device=None, 
                     val_data=None, val_env_prob=None, config=None):
    """
    Compute counterfactual-specific metrics
    
    Args:
        val_data: validation data for train metrics
        test_data: test data for test metrics  
        gen_data: generated data
        val_env_prob: validation environment probabilities
        env_bank: environment bank
        test_env_prob: test environment probabilities
    """
    tag = (key, data_name, seq_len, repeat_id)
    the_dict[tag] = {}
    
    ####### Normalize val data ##########
    # Handle val_data format - it's a dict {'c': np.array, 'x': np.array}
    val_x_data_raw = val_data['x']  # Already numpy array
    val_c_data_raw = val_data['c']  # Already numpy array
    # Use test data stats for normalization
    mean, std = np.mean(val_x_data_raw), np.std(val_x_data_raw) + 1e-7
    val_x_data = (val_x_data_raw - mean) / std
    mean_c, std_c = np.mean(val_c_data_raw), np.std(val_c_data_raw) + 1e-7
    val_c_data = (val_c_data_raw - mean_c) / std_c

    ################ val_data vs gen_data  ################ 
    print(f"[CF] Computing val metrics - val_data: {val_x_data.shape}, gen_data: {gen_data.shape}")

    mdd_train = get_mdd_eval(val_x_data, gen_data)
    flat_result_train = get_flat_distance(val_x_data, gen_data)  
    mmd_result_train = mmd_metric(val_x_data, gen_data)

    the_dict[tag].update({
        'mdd_train': float(mdd_train.item()) if hasattr(mdd_train, 'item') else float(mdd_train),
        'flat_kl_train': flat_result_train['flat_kl'],
        'mmd_rbf_train': mmd_result_train['mmd_rbf']
    })

    ########### test_data vs gen_data ##########
    print(f"[CF] Computing test metrics - test_data: {test_x_data.shape}, gen_data: {gen_data.shape}")

    mdd_test = get_mdd_eval(test_x_data, gen_data)
    flat_result_test = get_flat_distance(test_x_data, gen_data)
    mmd_result_test = mmd_metric(test_x_data, gen_data)

    the_dict[tag].update({
        'mdd_test': float(mdd_test.item()) if hasattr(mdd_test, 'item') else float(mdd_test),
        'flat_kl_test': flat_result_test['flat_kl'], 
        'mmd_rbf_test': mmd_result_test['mmd_rbf']
    })

    ######## J-FTSD ########
    jftsd_score = get_jftsd(val_x_data, test_c_data, gen_data, device=device)
    the_dict[tag]['jftsd_train'] = float(jftsd_score.item()) if hasattr(jftsd_score, 'item') else float(jftsd_score)

    jftsd_score = get_jftsd(test_x_data, test_c_data, gen_data, device=device)
    the_dict[tag]['jftsd_test'] = float(jftsd_score.item()) if hasattr(jftsd_score, 'item') else float(jftsd_score)
    
    return the_dict


def update_metrics_int(the_dict, key, data_name, seq_len, ori_data, gen_data, 
                      c_data=None, env_bank=None, env_prob=None, repeat_id=0, device=None):
    """
    Compute intervention-specific metrics
    
    Args:
        ori_data: real test data
        gen_data: generated data
        c_data: conditioning data
        env_bank: environment bank
        env_prob: test environment probabilities
    """

    tag = (key, data_name, seq_len, repeat_id)
    the_dict[tag] = {}
    
    # Basic distribution metrics
    mdd = get_mdd_eval(ori_data, gen_data)
    flat_result = get_flat_distance(ori_data, gen_data)
    mmd_result = mmd_metric(ori_data, gen_data)
    
    the_dict[tag].update({
        'mdd': float(mdd.item()) if hasattr(mdd, 'item') else float(mdd),
        'flat_kl': flat_result['flat_kl'],
        'mmd_rbf': mmd_result['mmd_rbf']
    })
    
    jftsd_score = get_jftsd(ori_data, c_data, gen_data, device=device)
    the_dict[tag]['jftsd_test'] = float(jftsd_score.item()) if hasattr(jftsd_score, 'item') else float(jftsd_score)
    
    return the_dict


def _get_method_name(task_type):
    if task_type == 'int':
        return 'CaTSG-Int'
    elif task_type == 'cf':
        return 'CaTSG-CF'
    elif task_type == 'cf_harmonic':
        return 'CaTSG-CF-Harmonic'


def safe_float_convert(value, default=0.00):
    """Safely convert a value to float, handling strings with ± notation"""
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Handle strings like "0.323017 ± 0.005100"
        if '±' in value:
            try:
                parts = value.split('±')
                mean_part = float(parts[0].strip())
                std_part = float(parts[1].strip())
                return f"{mean_part:.3f} ± {std_part:.3f}"
            except (ValueError, IndexError):
                return value  # Return original if parsing fails
        else:
            try:
                return float(value)
            except ValueError:
                return default
    
    return default

def save_metrics_to_csv(metric_dict, logdir, task_type='int', results_path="./results"):
    """
    Save metrics dictionary to a CSV file in {results_path}/{dataset}/{method}.csv
    """
    logdir_path = Path(logdir)

    # Expected path: logs/dataset_splitmethod/method/experiment_name/timestamp/
    path_parts = logdir_path.parts
    dataset_with_split = path_parts[-4]  # e.g., 'aq_station_based'
    experiment_name = path_parts[-2]  # e.g., 'catsg_seq96_bs32_lr1.0e-03_env4_seed42'
    method_name = _get_method_name(task_type)
    
    # Create results directory structure
    results_dir = Path(results_path) / dataset_with_split
    output_csv_path = results_dir / f"{method_name}.csv"
    csv_exists = output_csv_path.exists()

    # Extract config info from path
    config_summary = experiment_name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)

    with open(output_csv_path, mode='a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header if needed - include all possible metrics
        if not csv_exists:
            writer.writerow([
                "Timestamp", "ModelName", "Domain", "Dataset", "SeqLen", "Repeat",
                "MDD", "KL", "MMD", "J-FTSD",
                "ConfigSummary"
            ])
        for (model_name, dataset, seq_len, repeat), metrics in metric_dict.items():
            writer.writerow([
                timestamp,
                model_name,
                task_type,
                dataset,
                seq_len,
                repeat,
                safe_float_convert(metrics.get('mdd_test', metrics.get('mdd', 0.00))),
                safe_float_convert(metrics.get('flat_kl_test', metrics.get('flat_kl', 0.00))),
                safe_float_convert(metrics.get('mmd_rbf_test', metrics.get('mmd_rbf', 0.00))),
                safe_float_convert(metrics.get('jftsd_test', metrics.get('jftsd_test', 0.00))),
                config_summary   
            ])

    print(f"[INFO] Appended metrics to: {output_csv_path}")

