# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, wandb
from pytorch_lightning.trainer import Trainer
from utils.cli_utils import get_parser
from utils.init_utils import init_model_data_trainer
from utils.catsg_utils import test_model_catsg


def run_training(model, trainer, opt, logdir, melk, config):
    """Run training"""
    print(f"Training model...")
    
    try:
        trainer.logger.experiment.config.update(opt)
        trainer.fit(model)
    except KeyboardInterrupt:
        print("Training interrupted. Finalizing wandb upload...")
        wandb.finish()
        melk()
        raise
    except Exception as e:
        print("Training failed due to exception:", e)
        wandb.finish()
        melk()
        raise
    else:
        print("Training finished normally.")
        wandb.finish()

def _run_catsg_tests(model, val_data, test_data, trainer, opt, logdir, config, tasks):
    """Run CaTSG model tests for specified tasks"""
    dataset = opt.dataset or 'default'
    for task in tasks:
        test_model_catsg(model, val_data, test_data, trainer, opt, logdir, 
                        dataset=dataset, task_type=task, config=config)
    

def run_tests(model, val_data, test_data, trainer, opt, logdir, config):    
    if opt.test and opt.test != "auto":
        tasks = [opt.test]
    else:
        tasks = ['int']
        dataset_name = opt.dataset or 'default'
        if dataset_name.lower() in ['harmonic_vm', 'harmonic_vp']:
            tasks.append('cf_harmonic') # if harmonic data, do we can have the gound truth cf
        # else:
        #     tasks.append('cf') # if not harmonic data, do normal cf

    _run_catsg_tests(model, val_data, test_data, trainer, opt, logdir, config, tasks)
    
    print("Testing completed!!!")


if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    model, val_data, test_data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

    from omegaconf import OmegaConf
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    config = OmegaConf.merge(*configs)

    if opt.train:
        run_training(model, trainer, opt, logdir, melk, config)
        run_tests(model, val_data, test_data, trainer, opt, logdir, config)
        
    elif opt.test is not None:
        run_tests(model, val_data, test_data, trainer, opt, logdir, config)
        