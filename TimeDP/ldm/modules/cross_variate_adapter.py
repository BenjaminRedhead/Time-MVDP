# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Created for internship coding assessment
"""
Cross-Variate Conditioning Adapter for Multivariate TimeDP.

Takes per-variate PAM mask vectors M ∈ R^{B x C x Np} and enriches them
via top-k sparse self-attention across the variate dimension, producing
M_tilde ∈ R^{B x C x Np}. Each variate's conditioning is informed by its
most relevant neighbors.

The sparse adjacency graph A ∈ {0,1}^{B x C x C} learned here is also
returned for reuse by the CopulaAdapter.

Designed to slot into the TimeDP pipeline between the frozen PAM and the
frozen UNet — only this module's parameters are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CrossVariateAdapter(nn.Module):
    """
    Sparse self-attention adapter operating on stacked per-variate
    prototype assignment vectors (PAM masks).

    Args:
        n_prototypes: Number of prototypes Np (dimension of each mask vector m_c).
        d_model: Internal projection dimension for attention.
        top_k: Number of neighbor variates each variate attends to.
        n_heads: Number of attention heads.
    """

    def __init__(self, n_prototypes, d_model=64, top_k=3, n_heads=4):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.d_model = d_model
        self.top_k = top_k
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Project each variate's mask (Np,) into query, key, value space
        self.to_q = nn.Linear(n_prototypes, d_model, bias=False)
        self.to_k = nn.Linear(n_prototypes, d_model, bias=False)
        self.to_v = nn.Linear(n_prototypes, d_model, bias=False)

        # Project attention output back to mask space
        self.to_out = nn.Sequential(
            nn.Linear(d_model, n_prototypes),
            nn.Dropout(0.1)
        )

        # Learnable residual gate — initialized at 1.0 to ensure the adapter's
        # contribution to the mask is large enough for gradients to flow back
        # through the frozen UNet's cross-attention layers. A small gate (e.g. 0.01)
        # causes gradient vanishing through the deep frozen network.
        self.gate = nn.Parameter(torch.tensor(1.0))

    def _top_k_mask(self, scores):
        """
        Build a binary mask keeping only top-k entries per row.

        Args:
            scores: (B*H, C, C) raw attention scores.

        Returns:
            mask: (B*H, C, C) binary mask with 1s at top-k positions per row.
        """
        C = scores.shape[-1]
        k = min(self.top_k, C)  # handle case where C < top_k
        _, topk_indices = scores.topk(k, dim=-1)  # (B*H, C, k)

        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def forward(self, M):
        """
        Args:
            M: (B, C, Np) stacked per-variate PAM mask vectors.

        Returns:
            M_tilde: (B, C, Np) enriched mask vectors.
            A: (B, C, C) sparse adjacency graph (averaged over heads),
               for reuse by the CopulaAdapter.
        """
        B, C, Np = M.shape
        h = self.n_heads

        # Project to Q, K, V — each (B, C, d_model)
        q = self.to_q(M)
        k = self.to_k(M)
        v = self.to_v(M)

        # Reshape to multi-head: (B*H, C, d_head)
        q = rearrange(q, 'b c (h d) -> (b h) c d', h=h)
        k = rearrange(k, 'b c (h d) -> (b h) c d', h=h)
        v = rearrange(v, 'b c (h d) -> (b h) c d', h=h)

        # Compute attention scores: (B*H, C, C)
        scale = self.d_head ** -0.5
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale

        # Top-k sparse masking with straight-through gradient
        with torch.no_grad():
            sparse_mask = self._top_k_mask(scores)  # binary (B*H, C, C)

        # Apply mask: zero out non-top-k positions before softmax
        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(sparse_mask == 0, neg_inf)

        attn = F.softmax(masked_scores, dim=-1)  # (B*H, C, C)
        # Replace any NaN rows (can happen if a variate has no neighbors) with zeros
        attn = attn.nan_to_num(0.0)

        # Weighted combination of values
        out = torch.bmm(attn, v)  # (B*H, C, d_head)

        # Reshape back: (B, C, d_model)
        out = rearrange(out, '(b h) c d -> b c (h d)', h=h)

        # Project back to mask space and apply gated residual
        delta = self.to_out(out)  # (B, C, Np)
        M_tilde = M + self.gate * delta

        # Compute adjacency graph for CopulaAdapter:
        # Average raw scores across heads, then apply top-k for a clean binary graph
        scores_per_head = rearrange(scores, '(b h) i j -> b h i j', h=h)
        avg_scores = scores_per_head.mean(dim=1)  # (B, C, C)
        A = self._top_k_mask(avg_scores)  # (B, C, C) binary

        return M_tilde, A