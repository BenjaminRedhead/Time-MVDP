# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import sys
import traceback  

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion'))

from pytorch_lightning.trainer import Trainer
from diffusion.utils.cli_utils import get_parser
from diffusion.utils.init_utils import init_model_data_trainer
from diffusion.utils.test_utils import test_model_with_dp, test_model_uncond, test_model_unseen, test_model_guidance

if __name__ == "__main__":

    # data_root = os.environ.get('DATA_ROOT', None)
    # if not data_root or not os.path.exists(data_root):
    #     raise ValueError("DATA_ROOT is not defined or does not exist!")

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)


    model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

    if opt.train:
        try:
            trainer.logger.experiment.config.update(opt)
            trainer.fit(model, data)
        except Exception as e:
            print("Exception occurred during training!")
            print(traceback.format_exc())


            if trainer is not None and trainer.lightning_module is not None:
                print("Attempting to save checkpoint in exception handler via melk() ...")
                melk()  #
            else:
                print("Skipped calling melk() because trainer.lightning_module is None")

            raise e  #

    if not opt.no_test and not getattr(trainer, "interrupted", False):
        if opt.uncond and not opt.use_guidance:
            test_model_uncond(model, data, trainer, opt, logdir)
        if opt.use_guidance:
            test_model_guidance(model, data, trainer, opt, logdir)
        else:
            test_model_with_dp(model, data, trainer, opt, logdir, use_pam=opt.use_pam, use_text=opt.use_text)
            test_model_unseen(model, data, trainer, opt, logdir, use_pam=opt.use_pam, use_text=opt.use_text)
