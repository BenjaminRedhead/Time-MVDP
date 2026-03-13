# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import exists, default, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim_time import DDIMSampler
from ldm.modules.diffusionmodules.util import return_wrap
import copy

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 seq_len=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.seq_len = seq_len  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
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

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


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
        self.load_epoch = sd['epoch']
        self.load_step = sd["global_step"]
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        eps_pred = return_wrap(model_out, extract_into_tensor(self.ddim_coef, t, x.shape))

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=eps_pred)
        elif self.parameterization == "x0":
            x_recon = eps_pred
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)
        intermediates = [seq]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            seq = self.p_sample(seq, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(seq)
        if return_intermediates:
            return seq, intermediates
        return seq

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        seq_len = self.seq_len
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, seq_len),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

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
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        eps_pred = return_wrap(model_out, extract_into_tensor(self.shift_coef, t, x_start.shape))

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(eps_pred, target, mean=False).mean(dim=[1, 2])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 2:
            x = x[..., None]
        x = rearrange(x, 'b t c -> b c t')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 cond_drop_prob = None,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        
        # --- Extract multivariate kwargs before passing to DDPM ---
        self._mv_n_variates = kwargs.pop('n_variates', None)
        self._mv_adapter_top_k = kwargs.pop('adapter_top_k', 3)
        self._mv_adapter_d_model = kwargs.pop('adapter_d_model', 64)
        self._mv_adapter_n_heads = kwargs.pop('adapter_n_heads', 4)
        self._mv_copula_d_model = kwargs.pop('copula_d_model', 64)
        self._mv_copula_n_heads = kwargs.pop('copula_n_heads', 4)
        self._mv_corr_loss_weight = kwargs.pop('corr_loss_weight', 0.1)
        
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_drop_prob = cond_drop_prob
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        
        # --- Multivariate adapter initialization ---
        self.multivariate_mode = (self._mv_n_variates is not None and self._mv_n_variates > 1)
        if self.multivariate_mode:
            self._init_multivariate_adapters()

    def _init_multivariate_adapters(self):
        """Initialize the cross-variate and copula adapters for multivariate generation."""
        from ldm.modules.cross_variate_adapter import CrossVariateAdapter
        from ldm.modules.copula_adapter import CopulaAdapter
        
        # Determine number of prototypes from the cond_stage_model
        if hasattr(self.cond_stage_model, 'num_latents'):
            num_latents = self.cond_stage_model.num_latents
        else:
            # Fallback: try to infer from model config
            num_latents = 16
            print(f"[Multivariate] Warning: could not find num_latents on cond_stage_model, defaulting to {num_latents}")
        
        self.cross_variate_adapter = CrossVariateAdapter(
            n_prototypes=num_latents,
            d_model=self._mv_adapter_d_model,
            top_k=self._mv_adapter_top_k,
            n_heads=self._mv_adapter_n_heads,
        )
        self.copula_adapter = CopulaAdapter(
            seq_len=self.seq_len,
            d_model=self._mv_copula_d_model,
            n_heads=self._mv_copula_n_heads,
        )
        self.corr_loss_weight = self._mv_corr_loss_weight
        
        n_adapter_params = sum(p.numel() for p in self.cross_variate_adapter.parameters())
        n_copula_params = sum(p.numel() for p in self.copula_adapter.parameters())
        print(f"[Multivariate] Initialized adapters: CrossVariateAdapter ({n_adapter_params} params), "
              f"CopulaAdapter ({n_copula_params} params)")

    def freeze_base_model(self):
        """Freeze all parameters except the two multivariate adapters."""
        # Freeze UNet (DiffusionWrapper)
        for param in self.model.parameters():
            param.requires_grad = False
        # Freeze first stage (autoencoder)
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        # Freeze PAM / cond_stage_model
        if self.cond_stage_model is not None:
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        # Freeze logvar
        if hasattr(self, 'logvar') and isinstance(self.logvar, nn.Parameter):
            self.logvar.requires_grad = False
        
        # Count trainable
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Multivariate] Froze base model. Trainable: {n_trainable}/{n_total} parameters")

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            if hasattr(self.model.diffusion_model,"scale_factor"):
                del self.scale_factor
                self.register_buffer('scale_factor', self.model.diffusion_model.scale_factor)
                print(f"setting self.scale_factor to {self.scale_factor}")
                print("### USING Pre-Trained STD-RESCALING ###")
            else:
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c, return_mask=False):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            elif self.cond_stage_model is None:
                c, mask = None, None
            else:
                c, mask = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        if return_mask:
            return c, mask
        return c

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_mask=False):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c, mask = self.get_learned_conditioning(xc, return_mask=True)
                else:
                    c, mask = self.get_learned_conditioning(xc.to(self.device), return_mask=True)
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

        else:
            c = None
            xc = None
            mask = None
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if return_mask:
            out.append(mask)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b t c -> b c t').contiguous()

        z = 1. / self.scale_factor * z

        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    # ===================================================================
    # Multivariate: get_input, p_losses, shared_step overrides
    # ===================================================================

    def _get_input_multivariate(self, batch):
        """
        Process multivariate batch through frozen PAM per-variate, then
        enrich masks via CrossVariateAdapter.
        
        Args:
            batch: dict with 'context': (B, T, C_var) from MultivariateDataset
            
        Returns:
            z_all: (B, C_var, Ch_latent, T_latent) - encoded latents per variate
            context_all: (B, C_var, Np, d) - prototype embeddings per variate
            mask_tilde: (B, C_var, Np) - enriched masks from adapter
            A: (B, C_var, C_var) - sparse adjacency graph
        """
        x_mv = batch[self.first_stage_key]  # (B, T, C_var)
        x_mv = x_mv.to(self.device).float()
        B, T, C_var = x_mv.shape
        
        z_list = []
        context_list = []
        mask_list = []
        
        with torch.no_grad():
            for c in range(C_var):
                # Single variate: (B, T) -> (B, 1, T) matching UNet input format
                x_c = x_mv[:, :, c].unsqueeze(1)  # (B, 1, T)
                
                # Encode through frozen first stage
                encoder_posterior = self.encode_first_stage(x_c)
                z_c = self.get_first_stage_encoding(encoder_posterior).detach()
                z_list.append(z_c)  # (B, Ch_latent, T_latent)
                
                # Get conditioning from frozen PAM
                if self.cond_stage_model is not None:
                    ctx_c, mask_c = self.cond_stage_model(x_c)
                    context_list.append(ctx_c)  # (B, Np, d)
                    mask_list.append(mask_c)     # (B, Np)
        
        z_all = torch.stack(z_list, dim=1)              # (B, C_var, Ch_latent, T_latent)
        context_all = torch.stack(context_list, dim=1)   # (B, C_var, Np, d)
        mask_all = torch.stack(mask_list, dim=1)         # (B, C_var, Np)
        
        # Enrich masks via trainable cross-variate adapter
        mask_tilde, A = self.cross_variate_adapter(mask_all)
        
        return z_all, context_all, mask_tilde, A

    def _p_losses_multivariate(self, z_all, context_all, mask_tilde, A, x_mv_raw):
        """
        Compute multivariate training loss with three components:
        
        1. Per-variate denoising loss (through frozen UNet)
        2. Cross-variate adapter loss: mask similarity should reflect real correlation structure
           (direct gradient to adapter — bypasses UNet gradient vanishing)
        3. Copula adapter loss: correlation loss on corrected real data
           (direct gradient to copula adapter)
        
        Args:
            z_all: (B, C_var, Ch_latent, T_latent)
            context_all: (B, C_var, Np, d)
            mask_tilde: (B, C_var, Np)
            A: (B, C_var, C_var) sparse adjacency
            x_mv_raw: (B, T, C_var) raw multivariate input
        """
        B, C_var = z_all.shape[0], z_all.shape[1]
        prefix = 'train' if self.training else 'val'
        
        # ---- Loss 1: Per-variate denoising ----
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        
        total_denoise = 0.0
        for c in range(C_var):
            z_c = z_all[:, c]
            ctx_c = context_all[:, c]
            m_c = mask_tilde[:, c]
            
            noise_c = torch.randn_like(z_c)
            z_noisy_c = self.q_sample(x_start=z_c, t=t, noise=noise_c)
            
            model_out = self.apply_model(z_noisy_c, t, ctx_c, m_c,
                                         cond_drop_prob=self.cond_drop_prob)
            eps_pred = return_wrap(model_out, extract_into_tensor(self.shift_coef, t, z_c.shape))
            
            target = noise_c if self.parameterization == "eps" else z_c
            loss_c = self.get_loss(eps_pred, target, mean=False).mean([1, 2])
            total_denoise = total_denoise + loss_c.mean()
        
        denoise_loss = total_denoise / C_var
        
        # ---- Loss 2: Cross-variate adapter — mask similarity vs real correlation ----
        # Cosine similarity between enriched mask vectors should reflect actual
        # cross-variate correlation. This gives DIRECT gradient to the adapter,
        # bypassing the frozen UNet which causes gradient vanishing.
        mask_normed = mask_tilde / (mask_tilde.norm(dim=-1, keepdim=True) + 1e-8)
        mask_sim = torch.bmm(mask_normed, mask_normed.transpose(1, 2))  # (B, C, C)
        
        # Compute real per-batch correlation matrix
        x_mv = x_mv_raw.to(self.device).float()  # (B, T, C)
        x_perm = x_mv.permute(0, 2, 1)  # (B, C, T)
        x_centered = x_perm - x_perm.mean(dim=-1, keepdim=True)
        x_std = x_centered / (x_centered.std(dim=-1, keepdim=True) + 1e-8)
        real_corr = torch.bmm(x_std, x_std.transpose(1, 2)) / x_mv.shape[1]  # (B, C, C)
        
        adapter_loss = torch.nn.functional.mse_loss(mask_sim, real_corr.detach())
        
        # ---- Loss 3: Copula adapter — learn to restore cross-variate correlations ----
        # Simulate the copula's inference-time job: take data with DESTROYED
        # cross-variate correlations and train it to RESTORE them.
        # We independently shuffle each variate along the batch dimension,
        # which preserves marginal distributions but destroys correlations.
        from ldm.modules.copula_adapter import correlation_loss
        x_shuffled = x_perm.detach().clone()  # (B, C, T)
        for c in range(C_var):
            perm_idx = torch.randperm(B, device=x_shuffled.device)
            x_shuffled[:, c, :] = x_shuffled[perm_idx, c, :]
        
        x_restored = self.copula_adapter(x_shuffled, A.detach())  # (B, C, T)
        copula_loss = correlation_loss(x_restored, x_perm.detach())
        
        # ---- Combined loss ----
        loss = denoise_loss + self.corr_loss_weight * adapter_loss + self.corr_loss_weight * copula_loss
        
        loss_dict = {
            f'{prefix}/denoise_loss': denoise_loss.detach(),
            f'{prefix}/adapter_loss': adapter_loss.detach(),
            f'{prefix}/copula_loss': copula_loss.detach(),
            f'{prefix}/loss': loss.detach(),
        }
        
        return loss, loss_dict

    def shared_step(self, batch, **kwargs):
        if self.multivariate_mode:
            # --- Multivariate path ---
            z_all, ctx_all, mask_tilde, A = self._get_input_multivariate(batch)
            loss, loss_dict = self._p_losses_multivariate(
                z_all, ctx_all, mask_tilde, A,
                x_mv_raw=batch[self.first_stage_key]
            )
            return loss, loss_dict
        else:
            # --- Original univariate path (unchanged) ---
            x, c = self.get_input(batch, self.first_stage_key)
            kwargs['data_key'] = batch['data_key'].to(self.device)
            loss = self(x, c, **kwargs)
            return loss

    # ===================================================================
    # Multivariate sampling
    # ===================================================================

    @torch.no_grad()
    def sample_multivariate(self, few_shot_batch, n_samples=16, ddim_steps=50, eta=1.0):
        """
        Generate multivariate samples using few-shot domain prompts.
        
        Args:
            few_shot_batch: dict with 'context': (K, T, C_var) few-shot examples
            n_samples: number of multivariate samples to generate
            ddim_steps: DDIM sampling steps
            eta: DDIM eta parameter
            
        Returns:
            x_generated: (n_samples, C_var, T) generated multivariate time series
        """
        x_fs = few_shot_batch[self.first_stage_key].to(self.device).float()
        K, T, C_var = x_fs.shape
        
        # Extract domain prompts from all few-shot samples
        all_contexts = []
        all_masks = []
        
        for k_idx in range(K):
            ctx_list, mask_list = [], []
            for c in range(C_var):
                x_c = x_fs[k_idx, :, c].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
                ctx_c, mask_c = self.cond_stage_model(x_c)
                ctx_list.append(ctx_c)    # (1, Np, d)
                mask_list.append(mask_c)  # (1, Np)
            
            ctx_k = torch.stack(ctx_list, dim=1)    # (1, C_var, Np, d)
            mask_k = torch.stack(mask_list, dim=1)  # (1, C_var, Np)
            all_contexts.append(ctx_k)
            all_masks.append(mask_k)
        
        # Stack: (K, C_var, Np, d) and (K, C_var, Np)
        domain_contexts = torch.cat(all_contexts, dim=0)
        domain_masks = torch.cat(all_masks, dim=0)
        
        # Enrich masks via adapter
        domain_masks_tilde, A = self.cross_variate_adapter(domain_masks)
        
        # Generate using DDIM sampler
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.seq_len)
        
        all_generated = []
        
        # NOTE: Skip ema_scope for multivariate sampling because the EMA
        # was built before adapters were added and doesn't track them.
        # The base UNet weights are frozen so EMA vs non-EMA is identical for them.
        for i in tqdm(range(n_samples), desc="Generating multivariate samples"):
            prompt_idx = i % K
            per_variate_samples = []
            
            for c in range(C_var):
                ctx_c = domain_contexts[prompt_idx, c].unsqueeze(0)    # (1, Np, d)
                mask_c = domain_masks_tilde[prompt_idx, c].unsqueeze(0)  # (1, Np)
                
                samples, _ = ddim_sampler.sample(
                    S=ddim_steps, batch_size=1, shape=shape,
                    conditioning=ctx_c, mask=mask_c, eta=eta, verbose=False
                )
                # Decode from latent to original space
                x_decoded = self.decode_first_stage(samples)  # (1, 1, T)
                per_variate_samples.append(x_decoded.squeeze(0))  # (1, T)
            
            # Stack variates: (C_var, T)
            mv_sample = torch.cat(per_variate_samples, dim=0)
            all_generated.append(mv_sample)
        
        # Stack: (n_samples, C_var, T)
        x_generated = torch.stack(all_generated, dim=0)
        
        # Apply copula correction
        # Use adjacency from first prompt (representative of domain structure)
        A_batch = A[0:1].expand(n_samples, -1, -1)
        x_generated = self.copula_adapter(x_generated, A_batch)
        
        return x_generated

    # ===================================================================
    # Original methods (unchanged)
    # ===================================================================

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c, return_mask=True)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, mask, cfg_scale=1, cond_drop_prob=None, 
                    sampled_concept= None, sampled_index= None, sub_scale=None, **kwargs):

        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond, 'mask': mask}
        
        if cond_drop_prob is None:
            x_recon = self.model.cfg_forward(x_noisy, t, cfg_scale=cfg_scale, sampled_concept = sampled_concept, sampled_index = sampled_index, sub_scale = sub_scale, **cond)
        else:
            x_recon = self.model.forward(x_noisy, t, cond_drop_prob=cond_drop_prob, 
                                            sampled_concept = sampled_concept, sampled_index = sampled_index, sub_scale = sub_scale, **cond)

        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_losses(self, x_start, condmask, t, noise=None, data_key=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if condmask is not None:
            cond, mask = condmask
        else:
            cond = None
            mask = None
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond, mask, cond_drop_prob=self.cond_drop_prob)

        eps_pred = return_wrap(model_output, extract_into_tensor(self.shift_coef, t, x_start.shape))

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(eps_pred, target, mean=False).mean([1, 2])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
            
        loss_vlb = self.get_loss(eps_pred, target, mean=False).mean(dim=(1, 2))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/epoch_num': self.current_epoch})
        loss_dict.update({f'{prefix}/step_num': self.global_step})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, m, clip_denoised: bool, return_x0=False, 
                        score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, m,**kwargs)

        eps_pred = return_wrap(model_out,extract_into_tensor(self.ddim_coef, t, x.shape))

        if score_corrector is not None:
            assert self.parameterization == "eps"
            eps_pred = score_corrector.modify_score(self, eps_pred, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=eps_pred)
        elif self.parameterization == "x0":
            x_recon = eps_pred
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, m, clip_denoised=False, repeat_noise=False,
                 return_x0=False, temperature=1., noise_dropout=0., 
                 score_corrector=None, corrector_kwargs=None,**kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, m=m, clip_denoised=clip_denoised,
                                       return_x0=return_x0, score_corrector=score_corrector, 
                                       corrector_kwargs=corrector_kwargs,**kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              seq_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None,**kwargs):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            seq = torch.randn(shape, device=self.device)
        else:
            seq = x_T
        inter_recons = []
        inter_seqs = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            seq, x0_partial = self.p_sample(seq, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,**kwargs)
            if mask is not None:
                assert x0 is not None
                seq_orig = self.q_sample(x0, ts)
                seq = seq_orig * mask + (1. - mask) * seq

            if i % log_every_t == 0 or i == timesteps - 1:
                inter_recons.append(x0_partial)
                inter_seqs.append(seq)
            if callback: callback(i)
            if seq_callback: seq_callback(seq, i)
        return seq, inter_seqs, inter_recons

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, seq_callback=None, start_T=None,
                      log_every_t=None,**kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            seq = torch.randn(shape, device=device)
        else:
            seq = x_T

        intermediates = [seq]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            seq = self.p_sample(seq, cond, ts, mask,
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
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.seq_len)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond, shape, return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, mask=mask, x0=x0,**kwargs)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps=20,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.seq_len)
            samples, intermediates =ddim_sampler.sample(S = ddim_steps,batch_size = batch_size,
                                                    shape = shape,conditioning = cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=8, sample=True, plot_reconstruction=False, 
                   ddim_steps=20, ddim_eta=1., return_keys=None, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, mask = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, return_mask=True)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        if plot_reconstruction:
            log["reconstruction"] = xrec

        if sample:
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta, mask=mask)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            with self.ema_scope("Uncond Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta, cfg_scale=0, mask=mask)
            x_samples = self.decode_first_stage(samples)
            log["uncond_samples"] = x_samples

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        
        if self.multivariate_mode:
            # --- Only train adapter parameters ---
            params = (list(self.cross_variate_adapter.parameters()) +
                      list(self.copula_adapter.parameters()))
            n_params = sum(p.numel() for p in params)
            print(f"[Multivariate] Optimizing {n_params} adapter parameters at lr={lr:.2e}")
        else:
            # --- Original: train everything ---
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.logvar)
        
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
    
    def parameters(self):
        return self.diffusion_model.parameters()

    def forward(self, x, t, c_crossattn: list = None, cond_drop_prob = 0., mask=None, **kwargs):
        
        if (c_crossattn is not None) and (not None in c_crossattn):
            cc = torch.cat(c_crossattn, 1)
        else:
            cc = None
        out = self.diffusion_model(x, t, context=cc, mask=mask, cond_drop_prob=cond_drop_prob)
        
        return out
        
    def cfg_forward(self, x, t, c_crossattn: list = None, mask=None, **kwargs):
        
        if (c_crossattn is not None) and (not None in c_crossattn):
            cc = torch.cat(c_crossattn, 1)
        else:
            cc = None
        out = self.diffusion_model.forward_with_cfg(x, t, context=cc, mask=mask, **kwargs)
        
        return out