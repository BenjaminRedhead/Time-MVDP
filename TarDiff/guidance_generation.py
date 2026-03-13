# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from tqdm import tqdm
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.callback_utils import prepare_trainer_configs
from ldm.util import instantiate_from_config
from pathlib import Path
import matplotlib.pyplot as plt6
import pandas as pd
from ldm.modules.guidance_scorer import GradDotCalculator
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
from collections import Counter
from classifier.model import RNNClassifier
from classifier.classifier_train import TimeSeriesDataset
from ldm.modules.guidance_scorer import GradDotCalculator


def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        const=True,
                        default="",
                        nargs="?",
                        help="postfix for logdir")
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="resume and test",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="normalization method",
    )
    parser.add_argument("-p",
                        "--project",
                        help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="/mnt/storage/ts_diff_newer",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "-dw",
        "--dis_weight",
        type=float,
        const=True,
        default=1.,
        nargs="?",
        help="weight of disentangling loss",
    )
    parser.add_argument(
        "-dt",
        "--dis_loss_type",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="type of disentangling loss",
    )
    parser.add_argument(
        "-tg",
        "--train_stage",
        type=str,
        const=True,
        default='pre',
        nargs="?",
        help="pre / dis",
    )
    parser.add_argument(
        "-ds",
        "--dataset_name",
        type=str,
        const=True,
        default='elec',
        nargs="?",
        help="dataset name",
    )
    parser.add_argument(
        "-dp",
        "--dataset_prefix",
        type=str,
        const=True,
        default='/mnt/storage/tsdiff/data',
        nargs="?",
        help="dataset prefix",
    )
    parser.add_argument(
        "-cp",
        "--ckpt_prefix",
        type=str,
        const=True,
        default='/mnt/storage/tsdiff/outputs',
        nargs="?",
        help="ckpt prefix",
    )
    parser.add_argument(
        "-sp",
        "--sample_path",
        type=str,
        const=True,
        default='/mnt/storage/ts_generated/ours_amlt',
        nargs="?",
        help="samples prefix",
    )

    parser.add_argument(
        "-pl",
        "--pair_loss_type",
        type=str,
        const=True,
        default='',
        nargs="?",
        help="pair loss type: cosine or l2, otherwise not used")
    parser.add_argument("-sl",
                        "--seq_len",
                        type=int,
                        const=True,
                        default=24,
                        nargs="?",
                        help="sequence length")
    parser.add_argument("-uc",
                        "--uncond",
                        action='store_true',
                        help="unconditional generation")
    parser.add_argument("-si",
                        "--split_inv",
                        action='store_true',
                        help="split invariant encoder")
    parser.add_argument("-cl",
                        "--ce_loss",
                        action='store_true',
                        help="cross entropy loss")
    parser.add_argument("-up",
                        "--use_prototype",
                        action='store_true',
                        help="use prototype")
    parser.add_argument("-pd",
                        "--part_drop",
                        action='store_true',
                        help="use partial dropout conditions")
    parser.add_argument("-o",
                        "--orth_emb",
                        action='store_true',
                        help="use orthogonal prototype embedding")
    parser.add_argument("-ma",
                        "--mask_assign",
                        action='store_true',
                        help="use mask assignment")
    parser.add_argument("-ha",
                        "--hard_assign",
                        action='store_true',
                        help="use hard assignment")
    parser.add_argument("-im",
                        "--inter_mask",
                        action='store_true',
                        help="use intermediate assignment")
    parser.add_argument("-bs",
                        "--batch_size",
                        type=int,
                        const=True,
                        default=256,
                        nargs="?",
                        help="batch_size")
    parser.add_argument("-ms",
                        "--max_step_sum",
                        type=int,
                        const=True,
                        default=20000,
                        nargs="?",
                        help="max training steps")
    parser.add_argument("-nl",
                        "--num_latents",
                        type=int,
                        const=True,
                        default=16,
                        nargs="?",
                        help="sequence length")
    parser.add_argument("-pw",
                        "--pair_weight",
                        type=float,
                        const=True,
                        default=1.0,
                        nargs="?",
                        help="pair loss weight")
    parser.add_argument("-lr",
                        "--overwrite_learning_rate",
                        type=float,
                        const=True,
                        default=None,
                        nargs="?",
                        help="learning rate")
    parser.add_argument("-g",
                        "--gen_ckpt_path",
                        type=str,
                        const=True,
                        default='',
                        nargs="?",
                        help="ckpt path for diffusion")
    parser.add_argument("-pp",
                        "--downstream_pth_path",
                        type=str,
                        const=True,
                        default='',
                        nargs="?",
                        help="ckpt path for downstream model")
    parser.add_argument("-sv",
                        "--save_path",
                        type=str,
                        const=True,
                        default='',
                        nargs="?",
                        help="path for saving generated data")
    parser.add_argument("-op",
                        "--origin_data_path",
                        type=str,
                        const=True,
                        default='',
                        nargs="?",
                        help="path for origin data")
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    opt.save_path_origin = opt.save_path
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
    n_data = len(config.data['params']['data_path_dict'])
    config.model['params']['image_size'] = opt.seq_len
    config.model['params']['unet_config']['params']['image_size'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
        max_steps = opt.max_steps
    else:
        max_steps = config.lightning['trainer']['max_steps']
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params'][
            'batch_frequency'] = 5
        max_steps = 10
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(
            f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}"
        )
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']

    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}_ms{int(max_steps/1000)}k"
    # config.data['params']['batch_size'] = opt.batch_size

    if opt.normalize is not None:
        config.data['params']['normalize'] = opt.normalize
        nowname += f"_{config.data['params']['normalize']}"
    else:
        assert 'normalize' in config.data['params']
        nowname += f"_{config.data['params']['normalize']}"

    config.model['params']['pair_loss_flag'] = False
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params'][
            'window'] = opt.seq_len
        config.model['params']['cond_stage_config']['params'][
            'num_latents'] = opt.num_latents

        config.model['params']['pair_loss_flag'] = False
        config.model['params']['pair_loss_type'] = None
        config.model['params']['pair_loss_weight'] = 0
        nowname += f"_pl-None"

        config.model['params']['cond_stage_config']['params'][
            'split_inv'] = False
        config.model['params']['unet_config']['params'][
            'latent_unit'] = opt.num_latents

        config.model['params']['cond_stage_config']['params'][
            'use_prototype'] = False
        config.model['params']['cond_stage_config']['params'][
            'mask_assign'] = False
        config.model['params']['cond_stage_config']['params'][
            'hard_assign'] = False
        config.model['params']['unet_config']['params']['inter_mask'] = False
        config.model['params']['cond_stage_config']['params'][
            'orth_proto'] = False

    nowname += f"_seed{opt.seed}"
    # nowname = nowname
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
    # trainer_config["accelerator"] = "ddp"
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
    if "LatentDiffusion" in config.model['target']:
        if opt.dis_loss_type != None:
            config.model["params"]["dis_loss_type"] = opt.dis_loss_type
        config.model["params"]["dis_weight"] = opt.dis_weight

    alpha = 0.00001
    save_path = opt.save_path_origin

    train_tuple = pd.read_pickle(opt.origin_data_path)

    generation_nums_label = dict(Counter(train_tuple[1]))

    synt_nums_label = dict(Counter(train_tuple[1]))
    config.model['params']['ckpt_path'] = opt.gen_ckpt_path
    model = instantiate_from_config(config.model)
    model = model.cuda()
    model.eval()
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print("#### Data Preparation Finished #####")

    downstream_model = RNNClassifier(
        input_dim=7,
        hidden_dim=256,
        num_layers=2,
        rnn_type='gru',
        num_classes=1,
    )

    downstream_model.load_state_dict(
        torch.load(opt.downstream_pth_path)['model_state'])
    print('#### Downstream Model Loaded #####')

    print("#### Start Generating Samples #####")
    for dataset in data.norm_train_dict:
        normalizer = data.normalizer_dict[dataset]
        train_dataset = TimeSeriesDataset(data.transform(
            train_tuple[0].transpose(0, 2, 1), normalizer),
                                          train_tuple[1],
                                          normalize=False)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=1,
                                      shuffle=False)
        c = GradDotCalculator(downstream_model, train_dataloader,
                              nn.BCEWithLogitsLoss(), alpha)
        generated_data_all = None

        print(f"#### Start Generating Samples with alpha {alpha} #####")
        for label_sample, total_samples in tqdm(generation_nums_label.items()):
            label = torch.tensor([label_sample] * total_samples,
                                 device='cuda',
                                 dtype=torch.long)
            samples, z_denoise_row = model.sample_log(cond=None,
                                                      batch_size=total_samples,
                                                      ddim=True,
                                                      ddim_steps=20,
                                                      eta=1.,
                                                      sem_guide=True,
                                                      sem_guide_type='GDC',
                                                      label=label,
                                                      GDCalculater=c)
            norm_samples = model.decode_first_stage(
                samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples,
                                                 data_name=dataset)
            generated_data = np.array(inv_samples).transpose(0, 2, 1)
            if generated_data_all is None:
                generated_data_all = generated_data
            else:
                generated_data_all = np.concatenate(
                    [generated_data_all, generated_data], axis=0)
        generated_data_all = generated_data_all.transpose(0, 2, 1)
        label = np.concatenate([
            np.full(count, label)
            for label, count in generation_nums_label.items()
        ])
        tmp_name = f'synt_tardiff_noise_rnn_train_guidance_sc{alpha}'
        with open(save_path + '/'
                  f'{tmp_name}.pkl', 'wb') as f:
            #with open(save_path / f'alpha_search/{tmp_name}.pkl', 'wb') as f:

            pkl.dump((generated_data_all, label), f)
        print(f"Saved {tmp_name}.pkl in {save_path}")
