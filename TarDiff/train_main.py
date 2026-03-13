# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse, os, sys, datetime
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from utils.callback_utils import prepare_trainer_configs
from ldm.util import instantiate_from_config
from pathlib import Path


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

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":

    data_root = 'data/'
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

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
    if unknown:
        print(f"Ignoring unknown CLI overrides: {unknown}")
    config = OmegaConf.merge(*configs)
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

    nowname += f"_seed{opt.seed}"
    # nowname = nowname
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

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
    if "LatentDiffusion" in config.model['target']:
        if opt.dis_loss_type != None:
            config.model["params"]["dis_loss_type"] = opt.dis_loss_type
        config.model["params"]["dis_weight"] = opt.dis_weight

    if opt.resume:
        ckpt_path = logdir + '/' + 'checkpoints' + "/" + 'last.ckpt'
        config.model['params']['ckpt_path'] = ckpt_path

    print(f"Loading model........")
    model = instantiate_from_config(config.model)
    # trainer and callbacks
    trainer_kwargs = prepare_trainer_configs(nowname, logdir, opt,
                                             lightning_config, ckptdir, model,
                                             now, cfgdir, config, trainer_opt)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace(
            '/mnt/storage/', data_root)

    print("Preparing data.......")
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data Preparation Finished #####")

    # print(f"Train: {data.train_shape}, Validation: {data.val_shape}, Test: {data.test_shape}")
    # for k in data.datasets:
    #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    # bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
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
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)"
            .format(model.learning_rate, accumulate_grad_batches, ngpu, bs,
                    base_lr))
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
            import pudb
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    # run
    print(f"Starting training................")
    if opt.train:
        try:
            trainer.logger.experiment.config.update(opt)
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    print("Training finished!")
