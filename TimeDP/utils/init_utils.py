# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from utils.cli_utils import nondefault_trainer_args
from utils.callback_utils import prepare_trainer_configs
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from pathlib import Path
import datetime
from utils.cli_utils import nondefault_trainer_args

data_root = os.environ['DATA_ROOT']

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
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"
    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False

    # --- Multivariate: inject adapter params into model config ---
    if opt.multivariate:
        config.model['params']['n_variates'] = opt.n_variates
        config.model['params']['adapter_top_k'] = opt.adapter_top_k
        config.model['params']['adapter_d_model'] = opt.adapter_d_model
        config.model['params']['adapter_n_heads'] = opt.adapter_n_heads
        config.model['params']['copula_d_model'] = opt.copula_d_model
        config.model['params']['copula_n_heads'] = opt.copula_n_heads
        config.model['params']['corr_loss_weight'] = opt.corr_loss_weight
        nowname += f"_mv{opt.n_variates}_k{opt.adapter_top_k}"
            
    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    metrics_dir = Path(logdir) / 'metric_dict.pkl'
    if metrics_dir.exists():
        print(f"Metric exists! Skipping {nowname}")
        sys.exit(0)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    if opt.resume:
        ckpt_path = logdir / 'checkpoints' / 'last.ckpt'
        config.model['params']['ckpt_path'] = ckpt_path
    model = instantiate_from_config(config.model)

    # --- Multivariate: load pretrained checkpoint and freeze base ---
    if opt.multivariate:
        assert opt.pretrained_ckpt is not None, \
            "Multivariate mode requires --pretrained_ckpt pointing to a trained univariate model"
        assert opt.use_pam, \
            "Multivariate mode requires --use_pam (prototype assignment module)"
        print(f"[Multivariate] Loading pretrained checkpoint: {opt.pretrained_ckpt}")
        model.init_from_ckpt(opt.pretrained_ckpt)
        model.freeze_base_model()

    # trainer and callbacks
    trainer_kwargs = prepare_trainer_configs(nowname, logdir, opt, lightning_config, ckptdir, model, now, cfgdir, config, trainer_opt)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    if opt.multivariate:
        # --- Multivariate data path ---
        data = _init_multivariate_data(opt, config)
    else:
        # --- Original univariate data path ---
        for k, v in config.data.params.data_path_dict.items():
            config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        assert config.data.params.input_channels == 1, \
            "Assertion failed: Only univariate input is supported. Please ensure input_channels == 1."
    
    print("#### Data Preparation Finished #####")
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
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
    
    return model, data, trainer, opt, logdir, melk


def _init_multivariate_data(opt, config):
    """
    Initialize multivariate data module.
    
    Data paths can come from:
      1. --mv_data_paths CLI arg: 'name1:path1,name2:path2'
      2. Fallback: reuse the config's data_path_dict (same datasets, loaded as multivariate)
    """
    from ldm.data.multivariate_dataset import MultivariateDataModule
    
    if opt.mv_data_paths is not None:
        # Parse 'name:path,name:path' format
        data_path_dict = {}
        for pair in opt.mv_data_paths.split(','):
            name, path = pair.strip().split(':')
            path = path.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
            data_path_dict[name.strip()] = path.strip()
    else:
        # Reuse config paths — these point to .npy or .csv files 
        data_path_dict = {}
        for k, v in config.data.params.data_path_dict.items():
            path = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
            data_path_dict[k] = path

    # Get normalization from config or default
    normalize = config.data.params.get('normalize', 'centered_pit')
    
    data = MultivariateDataModule(
        data_path_dict=data_path_dict,
        n_variates=opt.n_variates,
        window=opt.seq_len,
        val_portion=config.data.params.get('val_portion', 0.1),
        normalize=normalize,
        batch_size=opt.batch_size,
        num_workers=config.data.params.get('num_workers', 0),
        pin_memory=config.data.params.get('pin_memory', True),
        drop_last=config.data.params.get('drop_last', False),
        reweight=config.data.params.get('reweight', False),
    )
    data.prepare_data()
    
    # Store for compatibility with test functions
    data.window = opt.seq_len
    
    print(f"[Multivariate] Loaded {len(data.key_list)} datasets with {data.actual_n_variates} variates")
    
    return data


def load_model_data(parser):
    
    opt, unknown = parser.parse_known_args()
        
    if opt.name:
        name = opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False

    # --- Multivariate: inject adapter params ---
    if opt.multivariate:
        config.model['params']['n_variates'] = opt.n_variates
        config.model['params']['adapter_top_k'] = opt.adapter_top_k
        config.model['params']['adapter_d_model'] = opt.adapter_d_model
        config.model['params']['adapter_n_heads'] = opt.adapter_n_heads
        config.model['params']['copula_d_model'] = opt.copula_d_model
        config.model['params']['copula_n_heads'] = opt.copula_n_heads
        config.model['params']['corr_loss_weight'] = opt.corr_loss_weight
        nowname += f"_mv{opt.n_variates}_k{opt.adapter_top_k}"
            
    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    
    # model
    ckpt_name = opt.ckpt_name
    ckpt_path = logdir / 'checkpoints' / f'{ckpt_name}.ckpt'
    config.model['params']['ckpt_path'] = ckpt_path
    model = instantiate_from_config(config.model)

    # data
    if opt.multivariate:
        data = _init_multivariate_data(opt, config)
    else:
        for k, v in config.data.params.data_path_dict.items():
            config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
    print("#### Data Preparation Finished #####")
    
    return model, data, opt, logdir