# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytorch_lightning as pl
from ldm.util import instantiate_from_config

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
    
    def parameters(self):
        return self.diffusion_model.parameters()

    def forward(self, x, t, c_crossattn: list = None, cond_drop_prob = 0., mask=None, **kwargs):
        
        if c_crossattn is not None:
            if isinstance(c_crossattn, list):
                if not None in c_crossattn:
                    cc = torch.cat(c_crossattn, 1)
                else:
                    cc = None
            else:
                cc = c_crossattn
        else:
            cc = None
        out = self.diffusion_model(x, t, context=cc, mask=mask, cond_drop_prob=cond_drop_prob)
        
        return out
        
    def cfg_forward(self, x, t, c_crossattn: list = None, mask=None, **kwargs):
        
        if c_crossattn is not None:
            if isinstance(c_crossattn, list):
                if not None in c_crossattn:
                    cc = torch.cat(c_crossattn, 1)
                else:
                    cc = None
            else:
                cc = c_crossattn
        else:
            cc = None
        out = self.diffusion_model.forward_with_cfg(x, t, context=cc, mask=mask, **kwargs)
        
        return out

