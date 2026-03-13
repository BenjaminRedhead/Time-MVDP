# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.ts_unet import UNetModel

from .util import Return


class BAUNetModel(UNetModel):
    def __init__(self, *args, num_env=10, latent_e_dim=32, task_type=None, hid_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = task_type  # 'int' or 'cf'
        self.hid_dim = hid_dim
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, hid_dim))

    def forward_with_bag(self, x, 
                         timesteps=None, 
                         context=None, 
                         EnvManager=None, # provide p(e|x_t,c) and εθ(x_t,t,c,e_k)
                         y=None, 
                         cfg_scale=1, 
                         env_probs_cf=None, # for cf, use fixed env prob
                         **kwargs):
        """
        Backdoor-adjusted guidance (BAG) aligned with:
            s_t^int ∝ (1+ω) * Σ_k p(e_k | x_t, c) * εθ(x_t,t,c,e_k)  -  ω * εθ(x_t,t,∅) (Eq.7 in CaTSG paper)

        Returns:
            eps_pred: tensor shaped like εθ output, e.g. (B, C, H, W) or (B, D).
        """

        B = x.shape[0]
        omega  = cfg_scale - 1.0 # ω 
        # omega  = 0.5

        # 1) p(e | x_t, c)  and εθ(x_t,t,c,e_k)
        eps_env_all, env_probs = EnvManager.compute_eps_env_all(
            model=self,
            x=x,                       # shape: (B, C, H, W)
            timesteps=timesteps,       # (B,)
            context=context,           # (B, N, D) 
            mask=None,                 # optional
            y=y                        # optional
        )

        if self.task_type == 'cf':
            env_probs = env_probs_cf

        # 2) Σ_k p_k ε_k
        eps_mix = (env_probs.unsqueeze(-1).unsqueeze(-1) * eps_env_all).sum(dim=1)  # -> (B, 1, 96)

        if omega == 0.0:
            return Return(pred=eps_mix)
        
        # 3) uncond εθ(x_t,t,∅) 
        eps_base = self._forward(x=x, timesteps=timesteps, context=self.null_ctx.expand(B, 96, -1), y=y, cond_drop_prob=1.0,**kwargs).pred

        # 4) (1+ω) * backdoor   -  ω * uncond
        eps_pred = (1 + omega) * eps_mix - omega * eps_base

        return Return(pred=eps_pred)
