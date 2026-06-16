"""
speedrun

sweeps context length and measures forward+backward throughput for each mixer.
attention's cost curves up quadratically, the SSM stays roughly linear.

    python speedrun.py
    python speedrun.py --lengths 128 256 512 1024 2048 --mixers attention ssm

this is the cheap demo. the full time-to-target-loss
leaderboard run goes on real data via train.py (see the README).
"""

import argparse
import time
import torch

from nmc import Config, LM


def bench(mixer, seq_len, dial, batch, device, iters=10):
    cfg = Config.from_dial(dial, mixer=mixer, seq_len=seq_len, vocab_size=256)
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (batch, seq_len), device=device)
    y = torch.randint(0, 256, (batch, seq_len), device=device)

    # warmup
    for _ in range(3):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    toks = batch * seq_len
    return toks / dt  # tokens/sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    p.add_argument("--mixers", nargs="+", default=["attention", "ssm", "hybrid"])
    p.add_argument("--dial", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"\nspeedrun  (dial={args.dial}, batch={args.batch}, device={args.device})")
    print("throughput in tokens/sec — higher is better\n")
    header = f"{'seq_len':<10}" + "".join(f"{m:>12}" for m in args.mixers)
    print(header)
    print("-" * len(header))
    for L in args.lengths:
        row = f"{L:<10}"
        for mixer in args.mixers:
            try:
                tps = bench(mixer, L, args.dial, args.batch, args.device)
                row += f"{tps:>12.0f}"
            except RuntimeError as e:  # e.g. OOM at long context for attention
                row += f"{'oom':>12}" if "memory" in str(e).lower() else f"{'err':>12}"
        print(row)
    print()


if __name__ == "__main__":
    main()
