from diffusion.cont_ctrl_net import UnetCC, DataModuleCC
from diffusion.disc_ctrl_net import UnetDC, DataModuleDC
from diffusion.ddpm import GaussianDiffusion, PLModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from utils.metrics_utils import get_metrics_func
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='SZAMain')
    parser.add_argument("--ctrl_type", type=str, default='continuous')
    parser.add_argument("--ctrl_target", type=str, default='return')
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--diffsteps", type=int, default=200)
    parser.add_argument("--samsteps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--maxsteps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--checkpoints", type=int, default=2)
    parser.add_argument("--data_path", type=str)  # data/container
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)
    print(f"Set seed: {args.seed}")

    exp_name = f'DiGA_{args.data_name.split("_")[0]}_{args.ctrl_type}_{args.ctrl_target}_{args.seed}'  # _{now}'

    results_folder = Path(args.output_path) / exp_name
    results_folder.mkdir(parents=True, exist_ok=True)
    print(f"Set output path: {results_folder}")
    
    train_path = Path(args.data_path) / f'{args.data_name}_train.npy'
    vali_path = Path(args.data_path) / f'{args.data_name}_vali.npy'
    selected_metrics = [get_metrics_func(args.ctrl_target)]

    if args.ctrl_type == 'continuous':
        pl_data = DataModuleCC(train_path, vali_path, selected_metrics=args.ctrl_target, batch_size=args.batch_size, n_bins=args.n_bins, num_workers=args.num_workers)
        model = UnetCC(dim = 64, n_ctrls = 1, dim_mults = (1, 4, 16), channels = 2)
    elif args.ctrl_type == 'discrete':
        pl_data = DataModuleDC(train_path, vali_path, selected_metrics=args.ctrl_target, batch_size=args.batch_size, n_bins=args.n_bins, num_workers=args.num_workers)
        model = UnetDC(dim = 64, num_classes=args.n_bins, dim_mults = (1, 4, 16), channels = 2)
        
    diffusion = GaussianDiffusion(model, seq_length = 236, timesteps = args.diffsteps, sampling_timesteps = args.samsteps, objective = 'pred_noise', auto_normalize = False)
    LitModel = PLModel(model = diffusion, train_lr = args.learning_rate, results_folder = results_folder)

    ckpt_callback = ModelCheckpoint(results_folder, filename='model-{epoch}', monitor='val/loss', save_top_k=args.checkpoints, save_last=True, mode='min')

    logger = WandbLogger(name=exp_name, project='DiGA-Meta-Controller', config=args, offline=True)

    trainer = Trainer(max_epochs=args.epochs, max_steps=args.maxsteps, callbacks=[ckpt_callback], logger=logger, default_root_dir=results_folder, accelerator="gpu")
    trainer.fit(LitModel, train_dataloaders=pl_data.train_dataloader(), val_dataloaders=pl_data.val_dataloader())


