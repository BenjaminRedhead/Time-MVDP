# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.diffusionmodules.util import return_wrap, extract_into_tensor


class GuideDDIMSampler(object):

    def __init__(self,
                 model,
                 schedule="linear",
                 guide_type='l2',
                 GDCalculater=None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.guide_type = guide_type
        self.GDC = GDCalculater
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self,
                      ddim_num_steps,
                      ddim_discretize="uniform",
                      ddim_eta=0.,
                      verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[
            0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model
                                                                     .device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev, ddim_coef = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_coef', ddim_coef)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

        self.posterior_variance = self.model.posterior_variance

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        mask=None,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        label=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, W = shape
        size = (batch_size, C, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            cond=conditioning,
            shape=size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            label=label,
            **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self,
                      cond,
                      shape,
                      x_T=None,
                      ddim_use_original_steps=False,
                      callback=None,
                      timesteps=None,
                      quantize_denoised=False,
                      mask=None,
                      x0=None,
                      img_callback=None,
                      log_every_t=100,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      label=None,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1) *
                self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(
            0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[
            0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range,
                        desc=f'DDIM Guided Sampler with dynamic scale',
                        total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b, ), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                x=img,
                c=cond,
                t=ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                label=label,
                **kwargs)
            img_hat, pred_x0 = outs

            # scale = extract_into_tensor(self.model.posterior_variance, ts, img.shape)
            with torch.enable_grad():
                score = self.semantic_scoring(pred_x0, cond, label, ts)
            img = (1 - self.GDC.gd_scale) * img_hat + score * self.GDC.gd_scale
            # img = img_hat
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def semantic_scoring(self, pred_x0, cond, label=None, t=None):
        # cond: [b, n unit, latent dim]
        pred_x0.requires_grad = True
        pred_cond = self.model.get_learned_conditioning(pred_x0)
        # cosine similarity between cond and pred_cond
        if self.guide_type == 'cosine':
            if cond.dim() == 2:
                sim_score = torch.nn.functional.cosine_similarity(
                    cond[:, -32:], pred_cond[:, -32:], dim=-1)
            else:  # assume the last condition is the target condition
                sim_score = torch.nn.functional.cosine_similarity(
                    cond[:, -1], pred_cond[:, -1], dim=-1)
            scale = (1 / max(sim_score.mean(), 0.05))**2
        elif self.guide_type == 'l2':
            if cond.dim() == 2:
                sim_score = -torch.nn.functional.pairwise_distance(
                    cond[:, -32:], pred_cond[:, -32:], p=2)
            else:
                sim_score = -torch.nn.functional.pairwise_distance(
                    cond[:, -1], pred_cond[:, -1], p=2)
            scale = 1 / max(torch.exp(sim_score.mean().detach()), 0.01)
        elif self.guide_type == 'GDC':

            sim_score = self.GDC.compute_gradient(pred_x0, label)
            return sim_score
        score = torch.autograd.grad(sim_score.mean(), pred_x0)[0]
        return score, scale

    @torch.no_grad()
    def p_sample_ddim(self,
                      x,
                      c,
                      t,
                      index,
                      repeat_noise=False,
                      use_original_steps=False,
                      quantize_denoised=False,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      label=None,
                      **kwargs):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, label=label, **kwargs)
            e_t = return_wrap(
                e_t, torch.full((b, 1, 1),
                                self.ddim_coef[index],
                                device=device))
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in,
                                                     t_in,
                                                     c_in,
                                                     label=label).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t -
                                                               e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c,
                                               **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1),
                                       sqrt_one_minus_alphas[index],
                                       device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device,
                                     repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
