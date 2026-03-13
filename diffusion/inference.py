# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pytorch_lightning.trainer import Trainer
from utils.cli_utils import get_parser
from utils.init_utils import init_model_data_trainer
from utils.test_utils import test_model_with_dp, test_model_uncond, test_model_unseen, test_model_guidance

if __name__ == "__main__":
    data_root = os.environ.get('DATA_ROOT', './')

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(train=False)
    parser.set_defaults(no_test=False)

    model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

    if opt.ckpt_path is not None:
        print(f"Loading checkpoint from {opt.ckpt_path}")
        model.init_from_ckpt(opt.ckpt_path)
    elif trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path}")
        model.init_from_ckpt(best_ckpt_path)
    else:
        print("⚠️  No checkpoint path provided and no best_model_path found! Proceeding without loading weights...")

    model.cuda()
    model.eval()

    if opt.use_text:
        print("Inference with text input enabled.")
    else:
        print("Inference without text input.")

    if not opt.no_test:
        if opt.uncond and not opt.use_guidance:
            test_model_uncond(model, data, trainer, opt, logdir)
        elif opt.use_guidance:
            test_model_guidance(model, data, trainer, opt, logdir)

        else:
            test_model_with_dp(model, data, trainer, opt, logdir, use_pam=opt.use_pam, use_text=opt.use_text, text_emb_dir=opt.text_emb_dir)
            test_model_unseen(model, data, trainer, opt, logdir,use_text=opt.use_text, text_emb_dir=opt.text_emb_dir)
