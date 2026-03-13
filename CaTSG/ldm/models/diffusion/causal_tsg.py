# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from functools import partial
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from contextlib import contextmanager
from torch.utils.data import DataLoader
from pathlib import Path
import math

from ldm.modules.diffusionmodules.util import (
    make_beta_schedule, noise_like, extract_into_tensor
)
from ldm.util import instantiate_from_config, default, exists, count_params
from ldm.modules.ema import LitEma
from ldm.models.diffusion.ddpm_time import DiffusionWrapper
from ldm.modules.diffusionmodules.util import return_wrap
from ldm.modules.diffusionmodules.dpm_sampler import  DDIMSampler as DPMDDIMSampler
from utils.catsg_utils import swav_loss_from_scores, sinkhorn

class CaTSGDiffusion(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        env_config=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="x",
        cond_stage_key="c",
        seq_len=96,
        channels=1,
        cond_channels=4,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.,
        v_posterior=0.,
        l_simple_weight=1.,
        conditioning_key="crossattn",
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.,
        cond_drop_prob=0.1,
        task_type="int", # ['int','cf'] 
        scale_by_std=False,
        scale_factor=1.0,
        orthogonal_loss_weight=0.1,  # Orthogonal regularization for env_bank embeddings
        warmup_steps=100,  
        warmup_losses=['swapped', 'orth'],  # Loss combination for warmup phase
        normal_losses=['mse', 'orth'],  # Loss combination for normal phase
        use_dpm_solver=True,   # Enable DPM-Solver by default
        dpm_solver_steps=20,   # Default DPM-Solver steps
        dpm_solver_order=2,    # Default DPM-Solver order
        dpm_solver_method='singlestep',  # Default DPM-Solver method
        dataset_name=None,     # Dataset name for embedding lookup
        split_method=None,     # Split method for embedding lookup
        enable_multi_view=True,   # Enable multi-view contrastive learning for swapped prediction loss
        swapped_loss_weight=1.0,   # Weight for swapped prediction loss

    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        
        self.env_config = env_config
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.seq_len = seq_len
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.conditioning_key = conditioning_key
        self.cond_drop_prob = cond_drop_prob
        self.task_type = task_type
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.warmup_steps = warmup_steps
        self.enable_multi_view = enable_multi_view
        self.swapped_loss_weight = swapped_loss_weight
        self.warmup_losses = warmup_losses
        self.normal_losses = normal_losses
        
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.embedding_config, self.cond_channels = self._setup_embedding_config(
            dataset_name, split_method
        )
        
        # Create embedding layers and index mapping based on configuration
        self.categorical_embeddings = nn.ModuleDict()
        self.index_to_feature = {}  # Map from position index to feature name
        total_embedded_dim = 0
        
        if self.embedding_config.get('categorical_features') == {}:
            print("No categorical features. ")
            num_categorical = 0
        else:
            for feature_name, config in self.embedding_config.get('categorical_features', {}).items():
                embedding_layer = nn.Embedding(config['vocab_size'], config['embedding_dim'])
                self.categorical_embeddings[feature_name] = embedding_layer
                # Now we have the dynamic index from c_var
                self.index_to_feature[config['index']] = feature_name
                total_embedded_dim += config['embedding_dim']
            num_categorical = len(self.embedding_config.get('categorical_features', {}))

        num_numerical = self.cond_channels - num_categorical
        self.c_total_dim = num_numerical + total_embedded_dim
        # ---- model ----
        if self.task_type in ['int', 'cf'] and env_config is not None:
            hid_dim = env_config['params']['hid_dim']
            original_context_dim = unet_config['params']['context_dim']
            adjusted_context_dim = hid_dim  # Match fusion MLP output
            unet_config['params']['context_dim'] = adjusted_context_dim
            print(f"Adjusted UNet context_dim: {original_context_dim} -> {adjusted_context_dim} (fusion MLP output)")
        
        self.model = CaDiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        if self.task_type == 'int' or self.task_type == 'cf':
            self.env_manager = self.instantiate_env_manager(self.env_config)
            
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(
            given_betas=given_betas, 
            beta_schedule=beta_schedule, 
            timesteps=timesteps,
            linear_start=linear_start, 
            linear_end=linear_end, 
            cosine_s=cosine_s
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
            
        # DPM-Solver parameters
        self.use_dpm_solver = use_dpm_solver
        self.dpm_solver_steps = dpm_solver_steps
        self.dpm_solver_order = dpm_solver_order
        self.dpm_solver_method = dpm_solver_method

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer("shift_coef", - to_torch(np.sqrt(alphas)) * (1. - self.alphas_cumprod_prev) / torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer("ddim_coef", -self.sqrt_one_minus_alphas_cumprod)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def instantiate_env_manager(self, env_config):
        # env_config['params']['hid_dim'] = self.c_hidden_dim
        env_config['params']['c_dim'] = self.c_total_dim
        env_config['params']['seq_len'] = self.seq_len
        
        env_manager = instantiate_from_config(env_config)
        
        # Set warmup_steps for adaptive temperature
        if hasattr(env_manager, 'adaptive_temperature_config'):
            env_manager.adaptive_temperature_config['warmup_steps'] = self.warmup_steps
            
        return env_manager

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        
        self.env_manager.set_stage("train")

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, 
                score_corrector=None, corrector_kwargs=None, **kwargs):
        '''
        used in sampling
        '''
         # get the noise prediction
        model_out = self.apply_model(x, t, c,  **kwargs)

        eps_pred = return_wrap(model_out,extract_into_tensor(self.ddim_coef, t, x.shape))

        # apply corrector if provided
        if score_corrector is not None:
            assert self.parameterization == "eps"
            eps_pred = score_corrector.modify_score(self, eps_pred, x, t, c, **corrector_kwargs)

        # reconstruct x_start from eps_pred
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=eps_pred)
        elif self.parameterization == "x0":
            x_recon = eps_pred
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        x_0_hat_1 = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t # -->  sqrt(1 / ᾱ_t) * x_t
        x_0_hat_2 = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise # --> sqrt(1 / ᾱ_t - 1) * noise
        x_0_hat = x_0_hat_1 - x_0_hat_2
        return x_0_hat

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_x0=False, temperature=1., noise_dropout=0., 
                 score_corrector=None, corrector_kwargs=None,**kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0, score_corrector=score_corrector, 
                                       corrector_kwargs=corrector_kwargs,**kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, seq_callback=None, start_T=None,
                      log_every_t=None, **kwargs):
        
        device = self.betas.device
        if x_T is None:
            seq = torch.randn(shape, device=device)
        else:
            seq = x_T
    
        if not log_every_t:
            log_every_t = self.log_every_t
        
        b = shape[0]

        intermediates = [seq]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))


        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            print(f"Sampling timestep {i}, ts shape: {ts.shape}, seq shape: {seq.shape}, cond shape: {cond.shape}, mask shape: {mask.shape if mask is not None else None}")
            seq = self.p_sample(seq, cond, ts, 
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised,**kwargs)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(seq)
            if callback: callback(i)
            if seq_callback: seq_callback(seq, i)

        if return_intermediates:
            return seq, intermediates
        return seq

    @torch.no_grad()
    def sample(self, c, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.seq_len)
        if c is not None:
            if isinstance(c, dict):
                c = {key: c[key][:batch_size] if not isinstance(c[key], list) else
                list(map(lambda x: x[:batch_size], c[key])) for key in c}
            else:
                c = [_[:batch_size] for _ in c] if isinstance(c, list) else c[:batch_size]
        return self.p_sample_loop(c, shape, return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, mask=mask, x0=x0,**kwargs)

    def sample_counterfactual(self, cond, env_prob, batch_size=16, shape=None, **kwargs):
        """
        Sample counterfactual data using fixed environment prb
        """
        if shape is None:
            shape = (batch_size, self.channels, self.seq_len)
            
        self.env_manager.env_info_extractor.set_fixed_env_prob_cf(env_prob)
        
        samples, _ = self.sample_log(c=cond, batch_size=batch_size, ddim=False, **kwargs)
        
        return samples.detach().cpu().numpy().transpose(0,2,1) 

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        weighted_x_start = extract_into_tensor(self.sqrt_alphas_cumprod,         t, x_start.shape) * x_start
        weighted_noise = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        out = weighted_x_start + weighted_noise
        return out

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        return loss

    def apply_model(self, x_noisy, t, cond, cfg_scale=1, cond_drop_prob=None, 
                    sampled_concept= None, sampled_index= None, sub_scale=None, return_env_prob=False, return_env_emb=False, **kwargs):
        if cond_drop_prob is None: # generation process
            if self.task_type == "int":
                x_recon = self.model.int_forward(x_noisy, t, c=cond, EnvManager=self.env_manager, cfg_scale=1.2, sampled_concept = sampled_concept, sampled_index = sampled_index, sub_scale = sub_scale)   
            elif self.task_type == "cf" or self.task_type == "cf_harmonic":
                env_prob_cf = self.env_manager.env_info_extractor.env_prob_cf
                x_recon = self.model.cf_forward(x_noisy, t, c=cond, EnvManager=self.env_manager, cfg_scale=1.2, env_prob_cf=env_prob_cf, sampled_concept = sampled_concept, sampled_index = sampled_index, sub_scale = sub_scale)
            return x_recon
        else:
            # joint training process - both int and cf use the same training process
            if return_env_prob and return_env_emb:
                x_recon, env_prob, env_emb = self.model.forward_with_env(
                    x_noisy, t, c=cond, EnvManager=self.env_manager, cond_drop_prob=cond_drop_prob, 
                    sampled_concept=sampled_concept, sampled_index=sampled_index, sub_scale=sub_scale, 
                    return_env_prob=True, return_env_emb=True
                )
                return x_recon, env_prob, env_emb
            elif return_env_prob:
                x_recon, env_prob = self.model.forward_with_env(
                    x_noisy, t, c=cond, EnvManager=self.env_manager, cond_drop_prob=cond_drop_prob, 
                    sampled_concept=sampled_concept, sampled_index=sampled_index, sub_scale=sub_scale, 
                    return_env_prob=True
                )
                return x_recon, env_prob
            else:
                x_recon = self.model.forward_with_env(
                    x_noisy, t, c=cond, EnvManager=self.env_manager, cond_drop_prob=cond_drop_prob, 
                    sampled_concept=sampled_concept, sampled_index=sampled_index, sub_scale=sub_scale,
                    return_env_prob=True
                )
                return x_recon
                
    def weight_sum_over_env(self, pred_all: torch.Tensor, env_probs: torch.Tensor):
        assert pred_all.dim() >= 2, "pred_all must be (B, K, ...)"
        B, K = env_probs.shape
        assert pred_all.shape[0] == B and pred_all.shape[1] == K, \
            f"Shape mismatch: pred_all[:2]={tuple(pred_all.shape[:2])}, env_probs={tuple(env_probs.shape)}"

        env_probs_safe = env_probs.clamp_min(1e-12)
        S_all = env_probs_safe.log()               # (B, K)

        # Sinkhorn
        Q_global = sinkhorn(S_all, eps=0.05, niters=3)  #  (K, B)
        if Q_global.shape == (K, B):
            Q_global = Q_global.t().contiguous()
        elif Q_global.shape != (B, K):
            raise RuntimeError(f"Unexpected Sinkhorn shape: {tuple(Q_global.shape)}, expect (B,K) or (K,B)")

        Q_global = Q_global.to(device=pred_all.device, dtype=pred_all.dtype)
        expand_shape = (B, K) + (1,) * (pred_all.dim() - 2)
        w = Q_global.reshape(expand_shape)         # (B, K, 1, 1, ...)
        return (w * pred_all).sum(dim=1)           # -> (B, ...)
    
    def p_losses(self, x_start, c, t, noise=None, x_view1=None, c_view1=None, x_view2=None, c_view2=None):
        '''
        p_losses(x, c, t) with optional multi-view support
        use in training to compute the loss
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Check if we're in warmup phase where UNet can be bypassed
        env_prob = None
        model_out = None
        
        # Check if we're in warmup phase (UNet bypassed)
        is_warmup_phase = False
        if self.task_type in ['int', 'cf'] and env_prob is not None:
            _, _, loss_config = self.get_training_phase()
            is_warmup_phase = not loss_config['mse']

        # Multi-view variables for swapped loss
        scores_list = []
        # is_global = []

        phase_name, _, loss_config = self.get_training_phase()
        if not is_warmup_phase:
            model_out, env_prob = self.apply_model(
                x_noisy, t, c, cond_drop_prob=self.cond_drop_prob, 
                task_type=self.task_type, return_env_prob=True
            )

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # Only compute MSE loss if not in warmup phase
        if not is_warmup_phase:
            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()
        
            # model_out is (B*K, D, T), t is (B*K,)
            # Need to handle expanded t for extract_into_tensor
            B, K = env_prob.shape
            t_expanded = t.unsqueeze(1).expand(-1, K).contiguous().view(B*K)
            eps_pred = return_wrap(model_out, extract_into_tensor(self.shift_coef, t_expanded, (B*K, *x_start.shape[1:])))
            # Reshape eps_pred from (B*K, ...) to (B, K, ...) for weight_sum_over_env
            eps_pred = eps_pred.view(B, K, *eps_pred.shape[1:])
            eps_pred = self.weight_sum_over_env(eps_pred, env_prob) 
            loss_simple = self.get_loss(eps_pred, target, mean=False).mean([1, 2])
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
            logvar_t = self.logvar[t.cpu()].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t

            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})
        else:
            # Warmup phase: skip MSE loss computation
            loss = torch.tensor(0.0, device=x_start.device)
            loss_dict.update({f'{prefix}/loss_simple': loss})
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss})
                loss_dict.update({'logvar': self.logvar.data.mean()})


        if env_prob is not None:
            phase_name, phase_code, loss_config = self.get_training_phase()
            
            # Initialize all losses
            mse_loss = torch.tensor(0.0, device=env_prob.device)
            orthogonal_loss = torch.tensor(0.0, device=env_prob.device)
            swapped_loss = torch.tensor(0.0, device=env_prob.device)
            
            # Compute losses based on configuration
            if loss_config['mse']:
                mse_loss = self.l_simple_weight * loss.mean()

            if loss_config['orth']:
                orthogonal_loss = self.compute_orthogonal_loss()

            if (loss_config['swapped'] and x_view1 is not None and x_view2 is not None and
                self.swapped_loss_weight > 0):
                # Multi-view contrastive loss using SwAV
                env_bank = self.env_manager.env_bank

                views_x = [x_start, x_view1, x_view2]
                # Ensure all c_views go through the same embedding process as original c
                views_c = [c, self.prepare_c(c_view1), self.prepare_c(c_view2)]
                
                scores_list = []
                all_is_global = [True, True, True] 
                
                # Process each view through EnvInfer
                for view_x, view_c in zip(views_x, views_c):
                    B, _, T = view_x.shape  # view_x is (B, 1, T)
                    
                    combined = torch.cat([view_x, view_c], dim=1)  # (B, 1+cond_dim, T)
                    # Forward through EnvInfer 
                    env_output = self.env_manager.env_info_extractor(combined, env_bank)
                    scores_list.append(env_output['scores'])
            
                swapped_loss, balance_reg = swav_loss_from_scores(scores_list, all_is_global)
                # L_swav + λ₁ * balance_reg
                swapped_loss = self.swapped_loss_weight * swapped_loss - 0.01 * balance_reg

            elif loss_config['swapped']:
                # SwAV loss requested but multi-view data not available (e.g., during validation)
                swapped_loss = torch.tensor(0.0, device=x_start.device)
            else:
                self.enable_multi_view = False  # Disable multi-view if swapped loss not configured
            
            # Adaptive weighting based on environment distribution
            # Use fixed orthogonal loss weight
            orth_loss_weight = self.orthogonal_loss_weight if loss_config['orth'] else 0.0
            
            # Log all losses and statistics (with original scale and no weighted)
            if loss_config['mse']:
                loss_dict.update({f'{prefix}/mse_loss': mse_loss})
            if loss_config['orth'] and orthogonal_loss.item() > 0:
                loss_dict.update({f'{prefix}/orthogonal_loss': orthogonal_loss})
            if loss_config['swapped'] and swapped_loss.item() > 0:
                loss_dict.update({f'{prefix}/swapped_loss': swapped_loss})

            # Combine all configured losses
            loss = torch.tensor(0.0, device=x_start.device)
            if loss_config['mse']:
                loss += mse_loss
            if loss_config['orth']:
                loss += orth_loss_weight * orthogonal_loss
            if loss_config['swapped']:
                loss += swapped_loss
            
            # Log environment probability statistics (only if available)
            if env_prob is not None:
                p_max = env_prob.max(dim=-1).values.mean()
                loss_dict.update({f'stats/p_env_max': p_max})
            loss_dict.update({f'{prefix}/training_phase': phase_code})
            loss_dict.update({f'{prefix}/unet_bypassed': float(not loss_config['mse'])})  # Log if UNet was bypassed
        else:
            # For non-environment tasks or when env_prob is None
            loss = self.l_simple_weight * loss.mean()
            loss_dict.update({f'{prefix}/training_phase': 4.0})  # normal_no_env = 4
            
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/epoch_num': self.current_epoch})
        loss_dict.update({f'{prefix}/step_num': self.global_step})

        return loss, loss_dict


    def get_training_phase(self):
        """
        Determine current training phase based on global step
        Returns: (phase_name, phase_code, loss_config)
        where loss_config is dict with keys:  'mse', 'orth', 'swapped'
        """
        step = self.global_step
        
        if step < self.warmup_steps:
            # Warmup phase: use configured warmup losses
            loss_config = {
                'mse': False,  # No MSE in warmup phase
                'orth': 'orth' in self.warmup_losses,
                'swapped': 'swapped' in self.warmup_losses,
            }
            return "warmup", 0, loss_config
        else:
            # Normal phase: use configured normal losses
            loss_config = {
                'mse': 'mse' in self.normal_losses,
                'orth': 'orth' in self.normal_losses,
                'swapped': 'swapped' in self.normal_losses,
            }
            return "normal", 1, loss_config

    def compute_orthogonal_loss(self):
        if not hasattr(self, 'env_manager') or self.env_manager is None:
            return torch.tensor(0.0, device=self.device)

        if not hasattr(self.env_manager, 'env_bank'):
            return torch.tensor(0.0, device=self.device)

        # env_bank: (K, D)
        env_bank = self.env_manager.env_bank
        K, D = env_bank.shape

        # Step 1: Normalize to unit length
        env_bank_norm = F.normalize(env_bank, dim=1)  # (K, D)

        # Step 2: Compute similarity (K x K)
        sim_matrix = env_bank_norm @ env_bank_norm.T  # (K, K)

        # Step 3: Off-diagonal loss
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=env_bank.device)
        orth_loss = (sim_matrix[off_diag_mask] ** 2).sum()

        return orth_loss


    def shared_step(self, batch, **kwargs):
        x = self.get_input(batch, self.first_stage_key)
        c = self.get_input(batch, self.cond_stage_key)
        
        multi_view_kwargs = {}
        if self.enable_multi_view:
            multi_view_kwargs['x_view1'] = self.get_input(batch, 'x_view1')
            multi_view_kwargs['c_view1'] = self.get_input(batch, 'c_view1')
            multi_view_kwargs['x_view2'] = self.get_input(batch, 'x_view2')
            multi_view_kwargs['c_view2'] = self.get_input(batch, 'c_view2')
    
        loss, loss_dict = self(x, c, **kwargs, **multi_view_kwargs)
        return loss, loss_dict

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        c = self.prepare_c(c)
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def _setup_embedding_config(self, dataset_name, split_method):
        """Setup embedding configuration based on dataset and split method"""

        from omegaconf import OmegaConf
        import os
        
        # Extract base dataset name from combined dataset_splitmethod name
        # Handle special cases where underscore is part of the dataset name
        if dataset_name.startswith('harmonic_vm'):
            base_dataset_name = 'harmonic_vm'
            split_method='alpha_based'
        elif dataset_name.startswith('harmonic_vp'):
            base_dataset_name = 'harmonic_vp'
            split_method='combination_based'
        else:
            base_dataset_name = dataset_name.split('_')[0] if '_' in dataset_name else dataset_name
        dataset_config_path = f"configs/dataset_config/{base_dataset_name}.yaml"
        print(f"Loading feature embeddings from {dataset_config_path}")
        dataset_config = OmegaConf.load(dataset_config_path)
        
        # Extract feature_embeddings from dataset config
        feature_embeddings_config = OmegaConf.to_container(dataset_config.feature_embeddings)
        
        split_config = getattr(dataset_config, split_method)
        c_var = OmegaConf.to_container(split_config.c_var)
        
        # Build categorical_features structure with dynamic indices
        categorical_features = {}
        for feature_name, config in feature_embeddings_config.items():
            if feature_name in c_var:
                # Add dynamic index based on c_var position
                config_with_index = config.copy()
                config_with_index['index'] = c_var.index(feature_name)
                categorical_features[feature_name] = config_with_index
        
        for feature_name, config in categorical_features.items():
            print(f"  {feature_name}: vocab_size={config['vocab_size']}, embed_dim={config['embedding_dim']}, index={config['index']}")
        
        return {'categorical_features': categorical_features}, len(c_var)
        

    @staticmethod
    def calculate_conditioning_dim(feature_embeddings, dataset_name, split_method, base_cond_channels):
        """Calculate the total conditioning dimension after embedding"""
        if not feature_embeddings:
            return base_cond_channels
            
        dataset_config = feature_embeddings.get(dataset_name, {})
        
        if not dataset_config:
            return base_cond_channels
            
        # Get categorical features directly from dataset config
        categorical_features = dataset_config
        if not categorical_features:
            return base_cond_channels
            
        # Calculate total embedded dimension
        total_embedded_dim = sum(config['embedding_dim'] for config in categorical_features.values())
        num_categorical = len(categorical_features)
        num_numerical = base_cond_channels - num_categorical
        
        total_dim = num_numerical + total_embedded_dim
        print(f"Conditioning dimension: {base_cond_channels} → {total_dim} ({num_numerical} numerical + {total_embedded_dim} embedded)")
        return total_dim
    
    def _apply_categorical_embeddings(self, c):
        """Apply categorical embeddings to conditioning data"""
        if not self.embedding_config.get('categorical_features'):
            return c
            
        processed_features = []
        
        for i in range(c.shape[-1]):
            if i in self.index_to_feature:
                # This is a categorical feature
                feature_name = self.index_to_feature[i]
                categorical_config = self.embedding_config['categorical_features'][feature_name]
                
                feature_values = c[:, :, i]  # (batch, seq_len)
                
                # Convert directly to integer indices (no normalization needed)
                feature_indices = feature_values.round().long()
                feature_indices = torch.clamp(feature_indices, 0, categorical_config['vocab_size'] - 1)
                
                # Apply embedding
                embedded = self.categorical_embeddings[feature_name](feature_indices)
                processed_features.append(embedded)
            else:
                # This is a numerical feature
                numerical_feature = c[:, :, i:i+1]  # (batch, seq_len, 1)
                processed_features.append(numerical_feature)
        
        # Concatenate all processed features
        return torch.cat(processed_features, dim=-1)

    def prepare_c(self, c):
        """Prepare conditioning data with proper shape, categorical embeddings,
        and (optional) time-of-day phase features."""
        if c.dim() != 3:
            raise ValueError(f"c must be 3D, got {tuple(c.shape)}")
        if c.shape[2] == self.seq_len:              # (B, D, T) -> (B, T, D)
            c = c.permute(0, 2, 1)

        c = self._apply_categorical_embeddings(c)

        B, T, _ = c.shape
        t = torch.arange(T, device=c.device, dtype=c.dtype)
        phi = (2.0 * math.pi) * t / float(T)    # [0, 2π)
        tod = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)   # (T, 2)
        tod = tod.unsqueeze(0).expand(B, T, 2)                        # (B, T, 2)
        c = torch.cat([c, tod], dim=-1)                               # (B, T, D_embed+2)

        return c.permute(0, 2, 1)
    
        
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 2:
            x = x[..., None]
        x = rearrange(x, 'b t c -> b c t')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        # Set multi-view dataset configuration on each batch if needed  
        if self.enable_multi_view and hasattr(self.trainer, 'train_dataloader'):
            dataloader = self.trainer.train_dataloader
            if hasattr(dataloader, 'dataset'):
                if hasattr(dataloader.dataset, 'enable_multi_view'):
                    dataloader.dataset.enable_multi_view = True
                elif hasattr(dataloader.dataset, 'datasets') and dataloader.dataset.datasets:
                    # Handle CombinedDataset case - need to find the actual dataset
                    for ds in dataloader.dataset.datasets:
                        if hasattr(ds, 'enable_multi_view'):
                            ds.enable_multi_view = True
                            break
        
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def on_after_backward(self):
        """Normalize env_bank after each optimization step"""
            
        if hasattr(self, 'env_manager') and self.env_manager is not None:
            if hasattr(self.env_manager, 'env_bank') and self.env_manager.env_bank is not None:
                # Only normalize if the norms are getting too large (avoid overriding small gradient updates)
                with torch.no_grad():
                    norms = torch.norm(self.env_manager.env_bank.data, dim=-1)
                    max_norm = norms.max().item()
                    # Only normalize if any vector has norm > 1.5 (allow some drift for better learning)
                    if max_norm > 1.5:
                        self.env_manager.env_bank.data = F.normalize(self.env_manager.env_bank.data, dim=-1)
                        if self.global_step % 50 == 0:  # Log occasionally
                            print(f"[ENV_BANK] Normalized env_bank at step {self.global_step}, max_norm was {max_norm:.4f}")
    

    @torch.no_grad()
    def sample_log(self, c, batch_size, ddim, ddim_steps=20, use_dpm_solver=None, **kwargs):
        """
        Sample from the model using DPM-Solver (default).
        """
        # Determine which sampler to use
        if use_dpm_solver is None:
            use_dpm_solver = getattr(self, 'use_dpm_solver', True)
            
        shape = (self.channels, self.seq_len)
        
        if use_dpm_solver:
            print(f"[DPM-Solver] Sampling with {self.dpm_solver_steps} steps (order {self.dpm_solver_order})")
            
            dpm_ddim_sampler = DPMDDIMSampler(self, use_dpm_solver=True)
            dpm_kwargs = {k: v for k, v in kwargs.items() if k not in ['steps', 'order', 'method']}
            
            samples, intermediates = dpm_ddim_sampler.sample(
                S=ddim_steps,
                batch_size=batch_size,
                shape=shape,
                conditioning=c,
                verbose=False,
                steps=self.dpm_solver_steps,
                order=self.dpm_solver_order,
                method=self.dpm_solver_method,
                **dpm_kwargs
            )
            
        else:
            # Use DDPM sampling
            print(f"[DDPM] Sampling with {self.num_timesteps} steps")
            samples, intermediates = self.sample(
                c=c, 
                batch_size=batch_size,
                return_intermediates=True,
                **kwargs
            )

        return samples, intermediates

    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Check if we're in limited validation mode and enable it in env_manager
        if hasattr(self, 'env_manager') and hasattr(self.trainer, 'limit_val_batches') and self.trainer.limit_val_batches is not None:
            self.env_manager.set_limited_val_mode(True)
        
        original_enable_multi_view = self.enable_multi_view
        if hasattr(self, 'env_manager') and self.env_manager is not None:
            # Set to train stage for consistent loss computation
            self.env_manager.set_stage("train")
        
        # Temporarily disable multi-view during validation since val dataset doesn't have multi-view data
        self.enable_multi_view = False
        
        try:
            loss, loss_dict_no_ema = self.shared_step(batch)
            with self.ema_scope():
                loss_ema, loss_dict_ema = self.shared_step(batch)
                loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            
            if 'val/loss' in loss_dict_no_ema:
                loss_dict_no_ema = {k: v for k, v in loss_dict_no_ema.items() if k != 'val/loss'}
            
            self.log("val/loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        finally:
            # Restore original settings
            self.enable_multi_view = original_enable_multi_view
        
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
            
    def on_train_start(self):
        """Called when training starts - save env_bank immediately and configure dataset"""
        if self.task_type in ['int', 'cf'] and hasattr(self, 'env_manager'):
            self._need_to_save_env_bank = True
        
        # Set up multi-view dataset configuration
        if self.enable_multi_view:
            self._enable_multiview_on_train_dataset()

    def _enable_multiview_on_train_dataset(self):
        ds = None
        if hasattr(self, "_train_set") and self._train_set is not None:
            ds = self._train_set
        else:
            try:
                loader = self.trainer.train_dataloader if hasattr(self.trainer, "train_dataloader") else None
                if callable(loader):
                    ds = loader().dataset
            except Exception:
                ds = None

        def _set_flag(d):
            import torch.utils.data as tud
            if d is None: 
                return False
            ok = False
            if hasattr(d, "enable_multi_view"):
                old = getattr(d, "enable_multi_view", None)
                d.enable_multi_view = True
                ok = True
                print(f"[TRAIN_START] Train dataset multi-view: {old} -> {d.enable_multi_view}")
            if hasattr(tud, "ConcatDataset") and isinstance(d, tud.ConcatDataset):
                for sub in d.datasets:
                    ok |= _set_flag(sub)
            return ok

        if not _set_flag(ds):
            print("[TRAIN_START] Could not set enable_multi_view (set it earlier in your DataModule.setup('fit'))")

    def on_fit_end(self):
        """Called when fit ends"""
        print(f"[FIT_END] Training completed")
            
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Call parent class method if it exists
        super().on_train_epoch_end() if hasattr(super(), 'on_train_epoch_end') else None

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        
        # Add env_manager parameters if trainable
        if self.task_type in ['int', 'cf'] and hasattr(self, 'env_manager'):
            params.extend(list(self.env_manager.parameters()))
        
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


    def set_datasets(self, train_set, val_set, batch_size=None, num_workers=12):
        self._train_set = train_set
        self._val_set = val_set
        self._original_train_set = train_set  # Keep original for normal phase
        self._original_val_set = val_set
        self._batch_size = batch_size if batch_size is not None else 32
        self._num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)


class CaDiffusionWrapper(DiffusionWrapper):
    
    # training
    def forward_with_env(self, x, t, c, EnvManager, cond_drop_prob = 0., mask=None, return_env_prob=False, return_env_emb=False, **kwargs):
        '''forward with environment bank, in training process
        return eps
        '''
        if return_env_prob and return_env_emb:
            c_e, env_prob, env_emb = EnvManager.fuse_e_c(bs=x.shape[0], c=c, x=x, return_env_prob=True, return_env_emb=True)
        elif return_env_prob:
            c_e, env_prob = EnvManager.fuse_e_c(bs=x.shape[0], c=c, x=x, return_env_prob=True)
        else:
            c_e = EnvManager.fuse_e_c(bs=x.shape[0], c=c, x=x)

        B = x.shape[0]
        K = env_prob.shape[1]
        _, _, T_eff, Df = c_e.shape
        ctx_rep = c_e.contiguous().view(B*K, T_eff, Df)

        x = x.unsqueeze(1).expand(-1, K, *x.shape[1:]).contiguous().view(B*K, *x.shape[1:])
        t = t.unsqueeze(1).expand(-1, K).contiguous().view(B*K)

        out = self.diffusion_model(x, t, ctx_rep, mask=mask, cond_drop_prob=cond_drop_prob, **kwargs)
        
        if return_env_prob and return_env_emb:
            return out, env_prob, env_emb
        elif return_env_prob:
            return out, env_prob
        else:
            return out

    def int_forward(self, x, t, c, EnvManager, mask=None, **kwargs):
        '''int generation process, using back-door adjusted guidance'''
        self.diffusion_model.task_type = 'int'
        out = self.diffusion_model.forward_with_bag(x, t, c, EnvManager, mask=mask, **kwargs)
        
        return out

    def cf_forward(self, x, t, c, EnvManager, mask=None, env_prob_cf=None, **kwargs):
        '''cf generation process, using back-door adjusted guidance'''
        self.diffusion_model.task_type = 'cf'
        out = self.diffusion_model.forward_with_bag(x, t, c, EnvManager, mask=mask, env_probs_cf=env_prob_cf, **kwargs)
        return out
