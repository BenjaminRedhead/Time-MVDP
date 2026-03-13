# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
DPM-Solver Sampler Integration
"""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


class DPMSolverSampler:
    """
    DPM-Solver sampler for time series diffusion models.
    Provides fast, high-quality sampling with 10-20 function evaluations.
    """
    
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.schedule = schedule
        
        # Extract noise schedule parameters from the model
        if hasattr(model, 'num_timesteps'):
            self.num_timesteps = model.num_timesteps
        else:
            self.num_timesteps = 1000
            
        if hasattr(model, 'betas'):
            self.betas = model.betas
        elif hasattr(model, 'alphas_cumprod'):
            self.alphas_cumprod = model.alphas_cumprod
        else:
            # Default linear schedule
            self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
            
        # Create noise schedule for DPM-Solver
        if hasattr(self, 'betas'):
            self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        elif hasattr(self, 'alphas_cumprod'):
            self.noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod)
        else:
            # Default to continuous linear schedule
            self.noise_schedule = NoiseScheduleVP(schedule='linear', 
                                                continuous_beta_0=0.1, 
                                                continuous_beta_1=20.)
        
        # Initialize DPM-Solver
        self.dpm_solver = None
        
    def _setup_solver(self, model_fn, algorithm_type="dpmsolver++"):
        """Setup the DPM-Solver with the appropriate model function."""
        self.dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.noise_schedule,
            algorithm_type=algorithm_type
        )
        
    def _model_wrapper(self, x, t_continuous, context=None, mask=None, **kwargs):
        """
        Wrapper function to adapt the diffusion model for DPM-Solver.
        """
        # Convert continuous time to discrete timesteps
        t_discrete = (t_continuous * (self.num_timesteps - 1)).round().long()
        t_discrete = torch.clamp(t_discrete, 0, self.num_timesteps - 1)
        
        # Call the original model
        if hasattr(self.model, 'apply_model'):
            # For DDPM-style models
            model_output = self.model.apply_model(x, t_discrete, context, **kwargs)
            # Handle Return objects
            if hasattr(model_output, 'pred'):
                noise_pred = model_output.pred
            else:
                noise_pred = model_output
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'diffusion_model'):
            # For our CaTSG model structure
            noise_pred = self.model.model.diffusion_model(x, t_discrete, context=context, mask=mask, **kwargs)
        else:
            # Generic forward call
            noise_pred = self.model(x, t_discrete, context=context, **kwargs)
            
        return noise_pred
        
    def sample(self, 
               S, 
               batch_size, 
               shape, 
               conditioning=None, 
               callback=None,
               normals_sequence=None, 
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
               steps=20,
               order=3,
               method='singlestep',
               algorithm_type="dpmsolver++",
               skip_type="time_uniform",
               **kwargs):
        """
        Sample from the diffusion model using DPM-Solver.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # Initialize random noise
        if x_T is None:
            img = torch.randn((batch_size, *shape), device=self.model.device, dtype=torch.float32)
        else:
            img = x_T

        # Setup model wrapper with conditioning
        if unconditional_conditioning is not None and unconditional_guidance_scale != 1.:
            # Classifier-free guidance
            def model_fn(x, t):
                # Concatenate conditional and unconditional inputs
                x_in = torch.cat([x, x])
                t_in = torch.cat([t, t])
                
                if conditioning is not None:
                    c_in = torch.cat([conditioning, unconditional_conditioning])
                else:
                    c_in = torch.cat([unconditional_conditioning, unconditional_conditioning])
                    
                noise_pred = self._model_wrapper(x_in, t_in, context=c_in, mask=mask, **kwargs)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return noise_pred
        else:
            # Standard conditioning
            def model_fn(x, t):
                return self._model_wrapper(x, t, context=conditioning, mask=mask, **kwargs)
        
        # Setup DPM-Solver
        self._setup_solver(model_fn, algorithm_type=algorithm_type)
        
        # Sample using DPM-Solver
        if verbose:
            print(f"DPM-Solver sampling with {steps} steps, order {order}, method {method}")
            
        samples = self.dpm_solver.sample(
            x=img,
            steps=steps,
            order=order,
            skip_type=skip_type,
            method=method,
            lower_order_final=True  # Use lower order for final step
        )
        
        # Create intermediates for compatibility
        intermediates = {'x_inter': [samples]}
        
        return samples, intermediates
        
    def ddim_sampling(self, *args, **kwargs):
        """Compatibility method that redirects to DPM-Solver sampling."""
        return self.sample(*args, **kwargs)
        
    def ddpm_sampling(self, *args, **kwargs):
        """Compatibility method that redirects to DPM-Solver sampling."""
        return self.sample(*args, **kwargs)


class DDIMSampler:
    """
    Enhanced DDIM sampler with DPM-Solver as default backend.
    Maintains compatibility with existing code while providing DPM-Solver acceleration.
    """
    
    def __init__(self, model, schedule="linear", use_dpm_solver=True, **kwargs):
        self.model = model
        self.use_dpm_solver = use_dpm_solver
        
        if use_dpm_solver:
            self.dpm_sampler = DPMSolverSampler(model, schedule=schedule, **kwargs)
        
        # Keep track of the number of reverse processes calls
        self.ddim_timesteps = None
        self.ddpm_num_timesteps = getattr(model, 'num_timesteps', 1000)
        
    def register_buffer(self, name, attr):
        """Register buffer for compatibility."""
        setattr(self, name, attr)
        
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """Create sampling schedule."""
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize, 
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose
        )
        
        alphas_cumprod = getattr(self.model, 'alphas_cumprod', None)
        if alphas_cumprod is not None:
            to_torch = partial(torch.tensor, dtype=torch.float32)
            
            self.register_buffer('betas', to_torch(self.model.betas))
            self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
            self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
            
            # Calculate DDIM parameters
            ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
                alphacums=alphas_cumprod.cpu(),
                ddim_timesteps=self.ddim_timesteps,
                eta=ddim_eta,
                verbose=verbose
            )
            self.register_buffer('ddim_sigmas', ddim_sigmas)
            self.register_buffer('ddim_alphas', ddim_alphas)
            self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
            self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
            
    def sample(self, S, batch_size, shape, conditioning=None, callback=None,
               normals_sequence=None, eta=0., mask=None, x0=None, temperature=1.,
               noise_dropout=0., score_corrector=None, corrector_kwargs=None,
               verbose=True, x_T=None, log_every_t=100,
               unconditional_guidance_scale=1., unconditional_conditioning=None,
               dynamic_threshold=None, ucg_schedule=None, **kwargs):
        """
        Main sampling method. Uses DPM-Solver by default for acceleration.
        """
        if self.use_dpm_solver:
            # Use DPM-Solver for fast sampling
            steps = kwargs.get('steps', min(20, S))  # Default to 20 steps or requested S
            order = kwargs.get('order', 3)
            method = kwargs.get('method', 'singlestep')
            
            # Remove conflicting parameters from kwargs
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['steps', 'order', 'method']}
            
            return self.dpm_sampler.sample(
                S=S, batch_size=batch_size, shape=shape, conditioning=conditioning,
                callback=callback, normals_sequence=normals_sequence, eta=eta,
                mask=mask, x0=x0, temperature=temperature, noise_dropout=noise_dropout,
                score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                verbose=verbose, x_T=x_T, log_every_t=log_every_t,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                steps=steps, order=order, method=method, **clean_kwargs
            )
        else:
            # Fall back to traditional DDIM sampling
            return self.ddim_sampling(
                conditioning, batch_size, shape, eta, callback, normals_sequence,
                mask, x0, temperature, noise_dropout, score_corrector,
                corrector_kwargs, verbose, x_T, log_every_t,
                unconditional_guidance_scale, unconditional_conditioning,
                dynamic_threshold, ucg_schedule, **kwargs
            )
            
    def ddim_sampling(self, cond, batch_size, shape, eta, callback, normals_sequence,
                     mask, x0, temperature, noise_dropout, score_corrector,
                     corrector_kwargs, verbose, x_T, log_every_t,
                     unconditional_guidance_scale, unconditional_conditioning,
                     dynamic_threshold, ucg_schedule, **kwargs):
        """Traditional DDIM sampling implementation."""
        device = self.model.betas.device
        b = batch_size
        
        if x_T is None:
            img = torch.randn(shape, device=device)
            img = img * temperature
        else:
            img = x_T
            
        if verbose:
            print(f"Running DDIM Sampling with {len(self.ddim_timesteps)} timesteps")
            
        intermediates = {'x_inter': [img]}
        
        time_range = reversed(range(0, len(self.ddim_timesteps)))
        total_steps = len(self.ddim_timesteps)
        
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=not verbose)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), self.ddim_timesteps[step], device=device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
                
            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
                
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=False,
                                     quantize_denoised=False, temperature=temperature,
                                     noise_dropout=noise_dropout, score_corrector=score_corrector,
                                     corrector_kwargs=corrector_kwargs,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning,
                                     dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
                
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                
        return img, intermediates
        
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, **kwargs):
        """Single DDIM sampling step."""
        b, *_, device = *x.shape, x.device
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, **kwargs).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas if use_original_steps else self.ddim_sigmas
        
        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1), sqrt_one_minus_alphas[index], device=device)
        
        # Current prediction for x_0
        if self.model.parameterization != "eps":
            raise NotImplementedError()
            
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
        if dynamic_threshold is not None:
            raise NotImplementedError()
            
        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0


# Utility functions
def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    """Create timesteps for DDIM sampling."""
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # Assert that the timesteps are unique and in ascending order
    assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # Add one to get the timesteps from the index (as done in ddim.py)
    ddim_timesteps = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for DDIM sampler: {ddim_timesteps}')
    
    return ddim_timesteps


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    """Create sampling parameters for DDIM."""
    # Select alphas for DDIM timesteps
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # According to DDIM paper
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for DDIM sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for DDIM sampler {sigmas}')
    
    return sigmas, alphas, alphas_prev


def noise_like(shape, device, repeat=False):
    """Generate noise tensor."""
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()