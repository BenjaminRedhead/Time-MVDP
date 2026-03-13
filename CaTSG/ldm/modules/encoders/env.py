# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.dilated_conv import DilatedConvEncoder


def l2_normalize_rows(x):
    """L2 normalize along the last dimension"""
    return F.normalize(x, p=2, dim=-1)

class EnvManager(nn.Module):
    def __init__(self, c_dim, hid_dim, num_envs, env_dim, seq_len, train_env=True, initial_temp=2.0, final_temp=0.1, 
                 depth=3):
        super().__init__()
        # ---- Env Bank ----
        self.num_envs = num_envs  # K
        self.env_dim = env_dim  # D_e
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        self.train_env = train_env
        
        # Adaptive temperature parameters
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.current_attention_temperature = initial_temp
        
        # Adaptive temperature schedule (warmup_steps will be set by parent model)
        self.adaptive_temperature_config = {
            'initial_temp': initial_temp,  # From config
            'final_temp': final_temp,      # From config
            'warmup_steps': None,          # Will be set from parent model's warmup_steps
            'enabled': True                # Enable adaptive temperature
        }
        self.env_bank = nn.Parameter(torch.empty(self.num_envs, self.env_dim), requires_grad=train_env)
        nn.init.orthogonal_(self.env_bank)
        with torch.no_grad():
            self.env_bank.data = F.normalize(self.env_bank.data, dim=-1)
        
        c_dim = c_dim+2  # include time feature
        self.c_encoder = DilatedConvEncoder(in_channels=c_dim, channels=[c_dim] * depth + [hid_dim], kernel_size=3)
        
        self.env_info_extractor = EnvInfer(H=c_dim +1, D_z=self.hid_dim, K=self.num_envs, topk_peaks=8, temp=self.initial_temp)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hid_dim + env_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim * 2, hid_dim)
        )

        # for log the prob of env
        self._current_stage = "train"  
        self._limited_val_mode = False  # Skip env_prov saving during limited validation
        self._epoch_env_stats = {}
        self._sample_env_stats = {}

    def set_stage(self, stage):
        self._current_stage = stage
    
    def set_limited_val_mode(self, enabled=True):
        """Enable/disable limited validation mode (skip env_prov saving for speed)"""
        self._limited_val_mode = enabled
    
    
    def get_current_temperature(self):
        """Get current attention temperature."""
        return self.current_attention_temperature
    
    def update_adaptive_temperature(self, current_step, total_steps):
        """Update adaptive temperature based on training progress."""
        if not self.adaptive_temperature_config['enabled']:
            return
        
        config = self.adaptive_temperature_config
        warmup_steps = config['warmup_steps']
        temperature_steps = 2 * warmup_steps  # Temperature annealing over 2x warmup duration
        
        if current_step <= temperature_steps:
            # Linear annealing from initial_temp to final_temp over 2x warmup period
            progress = current_step / temperature_steps
            temp_range = config['initial_temp'] - config['final_temp']
            new_temp = config['initial_temp'] - progress * temp_range
            
            self.current_attention_temperature = max(new_temp, config['final_temp'])
            
            if hasattr(self, 'env_info_extractor'):
                self.env_info_extractor.temp = self.current_attention_temperature
        
    def env_prob_infer(self, bs, c, x):
        combined_features = torch.cat([c, x], dim=1)  # (B, c_dim + x_dim,seq_len)
        env_output = self.env_info_extractor(combined_features, self.env_bank)

        return env_output
    
    def fuse_e_c(self, 
                 bs,
                 c, # (B, T, c_dim)
                 x,  # (B, T, x_dim)
                 return_env_prob=False, 
                 return_env_emb=False,
                 cf=False):
        B, _, _ = c.shape
        
        # Initialize env_prob to None (will be set based on mode)
        env_prob = None
        
        if cf:
            print(f"[CF] Using fixed environment prob.")
            env_prob = self.env_info_extractor.env_prob_cf
        else:
            env_prob_result = self.env_prob_infer(bs, c=c, x=x)
            env_prob = env_prob_result['p'] if env_prob_result is not None else None
            env_emb = torch.Tensor(env_prob @ self.env_bank).to(c.device) if env_prob is not None else torch.zeros(bs, self.env_dim, device=c.device)  # [B, D_e]

        h_c = self.c_encoder(c).permute(0,2,1) # (B,T,hid_dim)
        
        # use K dimension for multiple environments
        c_exp   = h_c.unsqueeze(1).expand(B, self.num_envs, self.seq_len, self.hid_dim)     # (B,K,T,hid_dim)
        bank_exp = self.env_bank.unsqueeze(0).unsqueeze(2).expand(B, self.num_envs, self.seq_len, self.env_dim)
        
        cat     = torch.cat([c_exp, bank_exp], dim=-1)                        # (B,K,T,Dc+De)
        BK      = B * self.num_envs
        cat     = cat.reshape(BK,  self.seq_len, self.hid_dim + self.env_dim)
        c_e_all = self.fusion_mlp(cat)
        c_e = c_e_all.reshape(B, self.num_envs, self.seq_len, -1)      # (B,K,T,Df)
        
        if return_env_prob and return_env_emb:
            return c_e, env_prob, env_emb
        elif return_env_prob:
            return c_e, env_prob
        elif return_env_emb:
            return c_e, env_emb
        return c_e

    def compute_eps_env_all(self, model, x, timesteps, context, mask=None, y=None, cf=False):
        """
        in generating process
        """
        B, K = context.shape[0], self.num_envs

        context_env_all, env_probs = self.fuse_e_c(B, context, x, return_env_prob=True, cf=cf) 
        
        # flatten into (B*K, ...) for batch inference
        x_rep = x.unsqueeze(1).expand(-1, K, *x.shape[1:]).reshape(B*K, *x.shape[1:])
        t_rep = timesteps.unsqueeze(1).expand(-1, K).reshape(B*K)
        ctx_rep = context_env_all.reshape(B*K, context.shape[2], -1)

        # expand for K environments
        if mask is not None:
            mask_rep = mask.unsqueeze(1).expand(-1, K, -1).reshape(B*K, *mask.shape[1:])
        else:
            mask_rep = None
        y_rep = None if y is None else y.unsqueeze(1).expand(-1, K).reshape(B*K)

        # model inference
        eps_all = model._forward(x_rep, timesteps=t_rep, context=ctx_rep, mask=mask_rep, y=y_rep).pred
        # reshape from (B*K, ...) to (B, K, ...)
        eps_env_all = eps_all.reshape(B, K, *eps_all.shape[1:])  

        return eps_env_all, env_probs


class EnvInfer(torch.nn.Module):
    '''EnvInfer q_\phi(e|x,c) in the paper'''
    def __init__(self, H, D_z, K, topk_peaks=8, temp=0.7, depth=3):
        super().__init__()
        self.infer_prob_encoder = DilatedConvEncoder(in_channels=H, channels=[H] * depth + [D_z], kernel_size=3)
        self.topk_peaks = topk_peaks
        self.temp = temp
        self.K = K
        self.D_z = D_z
        self.env_prob_cf = None  # For fixed env prob in cf mode

        self.att_fc = torch.nn.Linear(D_z, 1)

        D_raw = 3*D_z + D_z + (D_z + topk_peaks)      # stat(3H)+att(H)+spec(H+Kp)
        self.env_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(D_raw),
            torch.nn.Linear(D_raw, 2*D_z),
            torch.nn.GELU(),
            torch.nn.Linear(2*D_z, D_z)
        )

    def set_fixed_env_prob_cf(self, env_prob_cf):
        """Set fixed environment probabilities for counterfactual generation."""
        self.env_prob_cf = env_prob_cf

    def forward(self, h, env_bank=None):  # h: (B, H, T), env_bank: (K, D_z)
        h=self.infer_prob_encoder(h)
        # B, H, T = h.shape

        # stat
        mean_t = h.mean(-1)
        std_t  = h.std(-1, unbiased=False)
        max_t  = h.amax(-1)
        stat_feat = torch.cat([mean_t, std_t, max_t], dim=1)  # (B, 3H)

        # att
        att_w = self.att_fc(h.transpose(1,2))                 # (B, T, 1)
        att_w = torch.softmax(att_w, dim=1)
        att_feat = (h * att_w.transpose(1,2)).sum(-1)         # (B, H)

        # spectral
        psd = torch.fft.rfft(h.float(), dim=-1)
        psd = (psd.real**2 + psd.imag**2)                     # (B, H, T_fft)
        freq = torch.linspace(0, 1, psd.shape[-1], device=h.device).view(1,1,-1)
        psd_sum = psd.sum(-1, keepdim=True) + 1e-8
        centroid = (psd * freq).sum(-1) / psd_sum.squeeze(-1) # (B, H)
        topk_vals, _ = torch.topk(psd.mean(dim=1), k=min(self.topk_peaks, psd.shape[-1]), dim=-1)  # (B, Kp)
        spec_feat = torch.cat([centroid, topk_vals], dim=1)   # (B, H + Kp)

        env_feat = torch.cat([stat_feat, att_feat, spec_feat], dim=1)
        z_env = torch.tanh(self.env_proj(env_feat))           # (B, D_z)

        # L2 normalize for dot product
        z_env = l2_normalize_rows(z_env)                      
        C = l2_normalize_rows(env_bank)

        scores = (z_env @ C.t()) / self.temp                   # logits (B, K)
        p = torch.softmax(scores, dim=-1)

        return {"z_env": z_env, "scores": scores, "p": p}