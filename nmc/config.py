"""
Configuration — and the single dial.

Everything about the model size is derived from one number, `dial`, the same way
nanochat derives a whole family of models from `depth`. Turn the dial up, spend
more compute, get a bigger model; every other dimension follows a fixed rule so
you are never tuning a dozen knobs by hand. You are choosing a point on a curve.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- the dial ---
    dial: int = 6                 # sets n_layer; d_model and n_head follow

    # --- vocab / context (set by the data or task) ---
    vocab_size: int = 256
    seq_len: int = 256

    # --- token mixer ---
    mixer: str = "ssm"            # "attention" | "ssm" | "hybrid"
    hybrid_period: int = 4        # in hybrid mode, 1 of every N layers is attention

    # --- derived transformer dims (filled by from_dial) ---
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    mlp_mult: int = 4

    # --- ssm-specific ---
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 0              # 0 -> auto (ceil(d_model/16))
    scan: str = "sequential"     # "sequential" | "parallel"

    @classmethod
    def from_dial(cls, dial, **overrides):
        """Map the dial to a concrete architecture. The rule is intentionally
        simple and legible: width grows with the dial, heads keep head_dim=64."""
        n_layer = dial
        d_model = 64 * dial                 # 64, 128, ... width tracks depth
        n_head = max(1, d_model // 64)      # head_dim fixed at 64
        cfg = cls(dial=dial, n_layer=n_layer, d_model=d_model, n_head=n_head)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def layer_mixers(self):
        """Per-layer mixer assignment. 'hybrid' interleaves attention into an
        otherwise-SSM stack (Jamba-style): cheap global mixing where it helps,
        linear-time everywhere else."""
        if self.mixer in ("attention", "ssm"):
            return [self.mixer] * self.n_layer
        if self.mixer == "hybrid":
            return [
                "attention" if (i % self.hybrid_period == self.hybrid_period - 1) else "ssm"
                for i in range(self.n_layer)
            ]
        raise ValueError(f"unknown mixer '{self.mixer}'")
