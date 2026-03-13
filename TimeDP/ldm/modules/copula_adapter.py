# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Created for internship coding assessment
"""
Copula Adapter for Multivariate TimeDP.

After the frozen UNet generates C independent marginal time series,
this module applies a lightweight cross-variate correction to impose
the proper joint dependency structure.

Inspired by Sklar's theorem (any joint distribution = marginals + copula),
this adapter learns a small residual correction that adjusts cross-variate
correlations without destroying the per-variate temporal quality produced
by the pretrained diffusion model.

The adapter reuses the sparse adjacency graph A from the CrossVariateAdapter
so that only variates deemed relevant to each other exchange information.

Only this module's parameters are trained; the base model stays frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalEncoder(nn.Module):
    """
    Lightweight 1D-conv encoder that maps a single variate's time series
    into a fixed-size feature vector for cross-variate attention.

    Input:  (B*C, 1, T)
    Output: (B*C, d_model)
    """

    def __init__(self, seq_len, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (B*C, 64, 1)
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        """x: (B*C, 1, T) → (B*C, d_model)"""
        h = self.net(x).squeeze(-1)  # (B*C, 64)
        return self.proj(h)  # (B*C, d_model)


class TemporalDecoder(nn.Module):
    """
    Maps a per-variate feature vector back to a temporal residual correction.

    Input:  (B*C, d_model)
    Output: (B*C, 1, T)
    """

    def __init__(self, seq_len, d_model=64):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, seq_len),
        )
        # Initialize final layer near zero so initial residual is tiny
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        """h: (B*C, d_model) → (B*C, 1, T)"""
        return self.net(h).unsqueeze(1)  # (B*C, 1, T)


class CopulaAdapter(nn.Module):
    """
    Post-hoc joint distribution correction via sparse cross-variate attention.

    Architecture:
        1. Encode each variate's generated sequence into a feature vector.
        2. Run sparse cross-variate attention (reusing adjacency A from
           CrossVariateAdapter) to exchange information between related variates.
        3. Decode a small residual correction per variate.
        4. Add residual to original sequences (gated, starts near zero).

    Args:
        seq_len: Length of time series T.
        d_model: Hidden dimension for cross-variate attention.
        n_heads: Number of attention heads.
    """

    def __init__(self, seq_len, d_model=64, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.encoder = TemporalEncoder(seq_len, d_model)
        self.decoder = TemporalDecoder(seq_len, d_model)

        # Cross-variate attention projections
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1)
        )

        # Residual gate — moderate init since copula is trained directly via correlation loss
        self.gate = nn.Parameter(torch.tensor(0.1))

    def _sparse_cross_variate_attn(self, features, A):
        """
        Sparse multi-head attention across the variate dimension.

        Args:
            features: (B, C, d_model) per-variate feature vectors.
            A: (B, C, C) sparse adjacency from CrossVariateAdapter.

        Returns:
            out: (B, C, d_model) attention output.
        """
        B, C, _ = features.shape
        h = self.n_heads

        q = self.to_q(features)
        k = self.to_k(features)
        v = self.to_v(features)

        # Multi-head reshape: (B*H, C, d_head)
        q = rearrange(q, 'b c (h d) -> (b h) c d', h=h)
        k = rearrange(k, 'b c (h d) -> (b h) c d', h=h)
        v = rearrange(v, 'b c (h d) -> (b h) c d', h=h)

        # Attention scores: (B*H, C, C)
        scale = self.d_head ** -0.5
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale

        # Apply adjacency mask (shared across heads)
        # A is (B, C, C), expand to (B*H, C, C)
        A_expanded = A.unsqueeze(1).expand(-1, h, -1, -1)
        A_expanded = rearrange(A_expanded, 'b h i j -> (b h) i j')

        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(A_expanded == 0, neg_inf)

        attn = F.softmax(masked_scores, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = torch.bmm(attn, v)  # (B*H, C, d_head)
        out = rearrange(out, '(b h) c d -> b c (h d)', h=h)
        out = self.to_out(out)

        return out

    def forward(self, x_marginals, A):
        """
        Apply copula correction to independently generated marginals.

        Args:
            x_marginals: (B, C, T) stacked per-variate generated sequences.
                         Each x_marginals[:, c, :] was denoised independently.
            A: (B, C, C) sparse adjacency graph from CrossVariateAdapter.

        Returns:
            x_corrected: (B, C, T) joint-distribution-corrected sequences.
        """
        B, C, T = x_marginals.shape

        # 1. Encode each variate into a feature vector
        x_flat = rearrange(x_marginals, 'b c t -> (b c) 1 t')
        features = self.encoder(x_flat)  # (B*C, d_model)
        features = rearrange(features, '(b c) d -> b c d', b=B)  # (B, C, d_model)

        # 2. Sparse cross-variate attention
        attn_out = self._sparse_cross_variate_attn(features, A)  # (B, C, d_model)

        # Residual connection on features
        enriched = features + attn_out  # (B, C, d_model)

        # 3. Decode to temporal residual correction
        enriched_flat = rearrange(enriched, 'b c d -> (b c) d')
        delta = self.decoder(enriched_flat)  # (B*C, 1, T)
        delta = rearrange(delta, '(b c) 1 t -> b c t', b=B)  # (B, C, T)

        # 4. Gated residual addition
        x_corrected = x_marginals + self.gate * delta

        return x_corrected


def correlation_loss(x_generated, x_real):
    """
    Computes the Frobenius norm between the cross-variate correlation matrices
    of generated and real multivariate time series. Use as an auxiliary loss
    to train the CopulaAdapter.

    Args:
        x_generated: (B, C, T) generated multivariate samples.
        x_real: (B, C, T) real multivariate samples.

    Returns:
        loss: Scalar Frobenius norm of correlation matrix difference.
    """
    def _corr_matrix(x):
        # x: (B, C, T) → compute correlation across variates (C x C)
        # Pool over batch: concatenate all samples along time
        x_flat = rearrange(x, 'b c t -> c (b t)')  # (C, B*T)
        # Standardize each variate
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        x_normed = x_centered / (x_centered.std(dim=1, keepdim=True) + 1e-8)
        # Correlation = (1 / N) * X @ X^T
        N = x_normed.shape[1]
        corr = torch.mm(x_normed, x_normed.t()) / N
        return corr

    corr_gen = _corr_matrix(x_generated)
    corr_real = _corr_matrix(x_real)

    loss = torch.norm(corr_gen - corr_real, p='fro')
    return loss