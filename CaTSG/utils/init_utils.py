# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from utils.cli_utils import nondefault_trainer_args
from utils.callback_utils import prepare_trainer_configs
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
import datetime
from utils.cli_utils import nondefault_trainer_args

def init_model_data_trainer(parser):
    
    opt, unknown = parser.parse_known_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.name:
        name = opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Handle dataset parameter override
    dataset_base_name = opt.dataset if opt.dataset is not None else getattr(config, 'dataset', 'demo')
    if opt.dataset is not None:
        print(f"Overriding dataset to: {opt.dataset}")
        config.dataset = opt.dataset
    
    # Set default split_method based on dataset if not specified
    if opt.split_method in (None, "", "None"):
        if opt.dataset == "aq":
            opt.split_method = "station_based"
        elif opt.dataset == "traffic":
            opt.split_method = "temp_based"
    
    # Handle split_method parameter override  
    split_method_name = opt.split_method if opt.split_method is not None else 'standard'
    
    if opt.split_method is not None:
        print(f"Overriding split_method to: {opt.split_method}")
    
    # Set split_method in config for OmegaConf resolution
    config.split_method = split_method_name
    
    if dataset_base_name in ['harmonic_vm', 'harmonic_vp']:
        dataset_name = dataset_base_name  # Use base name without split method suffix
    else:
        dataset_name = f"{dataset_base_name}_{split_method_name}"
        
    # Convert config to container to handle interpolation manually
    config_dict = OmegaConf.to_container(config, resolve=False)
    
    # Update data paths with variable substitution
    def update_variable_paths(config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                update_variable_paths(value)
            elif isinstance(value, str):
                if "${dataset}" in value:
                    config_dict[key] = value.replace("${dataset}", dataset_name)
                if "${split_method}" in value:
                    # Special handling for synthetic datasets - remove the split_method path component
                    if dataset_base_name in ['harmonic_vm', 'harmonic_vp']:
                        # For synthetic datasets, remove the whole split_method path component
                        config_dict[key] = value.replace("/${split_method}", "")
                    else:
                        config_dict[key] = value.replace("${split_method}", split_method_name)
    
    update_variable_paths(config_dict)
    
    config = OmegaConf.create(config_dict)
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
    
    # Get actual config values for better logging names
    actual_seq_len = config.get('seq_len', opt.seq_len)
    actual_batch_size = config.get('batch_size', opt.batch_size)
    
    # Generate better logging name with actual config values
    experiment_name = f"{name.split('-')[-1]}_seq{actual_seq_len}_bs{actual_batch_size}_lr{base_lr:.1e}"
    
    # Add environment info for tasks that use it
    if hasattr(config, 'model') and hasattr(config.model, 'params') and hasattr(config.model.params, 'env_config'):
        n_envs = config.model.params.env_config.params.get('num_envs', 'auto')
        experiment_name += f"_env{n_envs}"
    
    experiment_name += f"_seed{opt.seed}"
    run_id = now  
    nowname = f"{experiment_name}_{run_id}"
    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        if 'unet_config' in config.model['params']:
            config.model['params']['unet_config']['params']['context_dim'] = None
        experiment_name += f"_uncond"
    else:
        if opt.use_pam:
            if 'unet_config' in config.model['params']:
                config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
                config.model['params']['unet_config']['params']['use_pam'] = True
            experiment_name += f"_pam"
        else:
            if 'unet_config' in config.model['params']:
                config.model['params']['unet_config']['params']['use_pam'] = False
    
    nowname = f"{experiment_name}_{run_id}"
    model_dir = "CaTSG"
    
    if hasattr(opt, 'test') and opt.test is not None:
        # Test mode: find existing checkpoint directory
        base_logdir = os.path.join(opt.logdir, dataset_name, model_dir, experiment_name)
        if os.path.exists(base_logdir):
            # Find the most recent checkpoint directory
            existing_dirs = [d for d in os.listdir(base_logdir) if os.path.isdir(os.path.join(base_logdir, d))]
            if existing_dirs:
                # Sort to get the most recent timestamp directory
                existing_dirs.sort()
                logdir = os.path.join(base_logdir, existing_dirs[-1])
                print(f"Test mode: Using existing checkpoint directory: {logdir}")
            else:
                # No existing directory found, create new one
                logdir = os.path.join(base_logdir, run_id)
                print(f"No existing checkpoint directory found, creating: {logdir}")
        else:
            # No base directory exists, create new one
            logdir = os.path.join(base_logdir, run_id)
            print(f"No base experiment directory found, creating: {logdir}")
    else:
        # Training mode: create new timestamped directory
        logdir = os.path.join(opt.logdir, dataset_name, model_dir, experiment_name, run_id)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    # Create structured subdirectories (removed metrics directory - now saved to ./results/)
    os.makedirs(os.path.join(logdir, "model"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "generation"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "evaluation"), exist_ok=True)
    
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # Handle GPU configuration for PyTorch Lightning 1.4.2 compatibility
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    
    if "gpus" in trainer_config and trainer_config["gpus"] is not None:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        # For PyTorch Lightning 1.4.2, don't set accelerator when using GPUs
        # Remove accelerator key if it exists
        if "accelerator" in trainer_config:
            del trainer_config["accelerator"]
        cpu = False
    else:
        if "accelerator" in trainer_config:
            del trainer_config["accelerator"]
        cpu = True
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    if not opt.train:
        base_logdir = f"./logs/{dataset_name}/{model_dir}"
        experiment_pattern = experiment_name
        matching_dirs = []
        
        for item in os.listdir(base_logdir):
            item_path = os.path.join(base_logdir, item)
            if os.path.isdir(item_path) and experiment_pattern in item:
                matching_dirs.append(item)
        
        if matching_dirs:
            # Sort to get the most recent experiment
            matching_dirs.sort()
            latest_experiment = matching_dirs[-1]
            experiment_path = os.path.join(base_logdir, latest_experiment)
            
            # Look for timestamp directories within the experiment
            timestamp_dirs = []
            if os.path.exists(experiment_path):
                for item in os.listdir(experiment_path):
                    item_path = os.path.join(experiment_path, item)
                    if os.path.isdir(item_path) and item.startswith('20'):  # timestamp format
                        timestamp_dirs.append(item)
            
            if timestamp_dirs:
                # Sort to get the most recent timestamp and search for checkpoints
                timestamp_dirs.sort()
                
                # Try each timestamp directory from newest to oldest until we find checkpoints
                for timestamp in reversed(timestamp_dirs):
                    checkpoint_dir = os.path.join(experiment_path, timestamp, "checkpoints")
                    
                    if os.path.exists(checkpoint_dir):
                        # Look for checkpoint files
                        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                        
                        if ckpt_files:
                            # Prefer last.ckpt, otherwise use the most recent one
                            if 'last.ckpt' in ckpt_files:
                                checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
                            else:
                                ckpt_files.sort()
                                checkpoint_path = os.path.join(checkpoint_dir, ckpt_files[-1])
                            
                            print(f"Found checkpoint: {checkpoint_path}")
                            config.model['params']['ckpt_path'] = checkpoint_path
                            found_checkpoint = True
                            break  # Stop searching once we find a valid checkpoint
        
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = prepare_trainer_configs(nowname, logdir, opt, lightning_config, ckptdir, model, now, cfgdir, config, trainer_opt)
    trainer_kwargs.setdefault("callbacks", [])
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    train_data = instantiate_from_config(config.train_data)
    val_data = instantiate_from_config(config.val_data)
    test_data = instantiate_from_config(config.test_data)
    # Pass actual batch size from config to avoid hardcoded values
    actual_batch_size = config.get('batch_size', opt.batch_size)
    model.set_datasets(train_data, val_data, batch_size=actual_batch_size)
    print("#### Data Preparation Finished #####")

    # for the int and cf we need to save the env
    if config.model['params']['task_type'] == 'int' or config.model['params']['task_type'] == 'cf':
        model.env_prob_train = []
        model.env_prob_val = []
        model.env_prob_test = []

    if not cpu:
        gpu_config = lightning_config.trainer.gpus
        if isinstance(gpu_config, str):
            ngpu = len(gpu_config.strip(",").split(','))
        elif isinstance(gpu_config, int):
            ngpu = 1 if gpu_config >= 0 else 0
        elif isinstance(gpu_config, list):
            ngpu = len(gpu_config)
        else:
            ngpu = 1
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        
    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; # type: ignore
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    
    return model, val_data, test_data, trainer, opt, logdir, melk
