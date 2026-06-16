"""
The model.

A plain decoder-only language model with one twist: every block's token-mixer is
chosen by config, so the *same* class is an attention Transformer, a Mamba-style
SSM, or a hybrid, depending on one flag. Norm -> mixer -> residual,
Norm -> MLP -> residual. Pre-norm, RMSNorm, untied... actually weight-tied head,
because at this scale tying helps and it is one fewer thing to explain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .mixers import make_mixer


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg, mixer_kind):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = make_mixer(mixer_kind, cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # SSMs are inherently sequential (the recurrence carries position), so the
        # positional embedding is only added for layers that need it. We keep a
        # learned pos-emb and let attention rely on it; it is harmless for SSMs.
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(
            Block(cfg, kind) for kind in cfg.layer_mixers()
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.numel()  # tied head already not double-counted

    def forward(self, idx, targets=None):
        b, l = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :l]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
