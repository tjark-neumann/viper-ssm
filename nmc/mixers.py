"""
token mixers:
the one component that viper-ssm swaps

both mixers take (batch, length, d_model) and return (batch, length, d_model),
so the surrounding architecture (norm, MLP, residuals, embeddings, head) is
identical regardless of which one you pick. that is deliberate: it makes the
attention-vs-SSM comparison a controlled experiment. the only thing that changes
between runs is how tokens talk to each other.

  * CausalSelfAttention: standard multi-head causal attention. O(L^2) compute,
    O(L) state at inference (the KV cache grows with context).

  * SelectiveSSM: a minimal Mamba-style selective state-space mixer (S6):
    input-dependent (B, C, delta), a learned decay A, a short causal conv, and a
    SiLU gate. O(L) compute, O(1) state per step at inference (fixed-size hidden
    state, independent of context length).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import get_scan


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, l, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, l, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, l, self.n_head, self.d_head).transpose(1, 2)
        # flash / fused causal attention when available
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        return self.proj(y)


class SelectiveSSM(nn.Module):
    """Minimal Mamba (S6) mixer. No fused CUDA kernel — the scan is plain PyTorch
    (see nmc/scan.py). Faithful to the architecture, optimised for readability."""

    def __init__(self, cfg):
        super().__init__()
        self.d_inner = cfg.expand * cfg.d_model
        self.d_state = cfg.d_state
        self.dt_rank = cfg.dt_rank if cfg.dt_rank > 0 else math.ceil(cfg.d_model / 16)
        self.scan = get_scan(cfg.scan)

        # project into the inner space, twice: one path is the SSM input, one the gate
        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)

        # short depthwise causal conv mixes a little local context before the scan
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )

        # produce the *selective* (input-dependent) parameters delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is learned in log space and kept negative (stable decay). D is a skip.
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        xz = self.in_proj(x)                              # (b, l, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                     # each (b, l, d_inner)

        # depthwise causal conv (trim the right padding to stay causal)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :l]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)

        # selective parameters
        A = -torch.exp(self.A_log.float())               # (d_inner, d_state), negative
        dbc = self.x_proj(x_in)                           # (b, l, dt_rank + 2*d_state)
        dt, B, C = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))             # (b, l, d_inner), positive

        y = self.scan(x_in, delta, A, B, C, self.D)      # (b, l, d_inner)
        y = y * F.silu(z)                                # gate
        return self.out_proj(y)


def make_mixer(kind, cfg):
    if kind == "attention":
        return CausalSelfAttention(cfg)
    if kind == "ssm":
        return SelectiveSSM(cfg)
    raise ValueError(f"unknown mixer '{kind}'")
