"""
Diagnostics

Three synthetic tasks, each probing a capability the architecture-vs-architecture
debate actually turns on:

  * recall   — associative recall. Key/value pairs, then a query key; predict its
               value. Attention nails this (it can look the pair up directly); a
               fixed-state SSM has to have *stored* the right binding.

  * copy     — copy a random sequence verbatim after a delimiter. Pure long-range
               exact memory.

  * needle   — needle in a haystack. One key/value pair hidden in a long stream of
               filler, queried at the very end. Retrieval over distance.

Each task trains a small model per mixer for a fixed budget and reports accuracy
on held-out samples. Run all three across all mixers:

    python diagnostics.py                       # full sweep, prints a table
    python diagnostics.py --task recall         # one task
    python diagnostics.py --mixers ssm attention

"""

import argparse
import torch

from nmc import Config, LM

# task generators
# Every generator returns (x, y) of shape (batch, L). Positions we don't score
# are set to -1 in y (ignored by cross-entropy). Vocab layout is per-task.


def gen_recall(batch, n_pairs=8, n_symbols=16, device="cpu"):
    """[k0 v0 k1 v1 ... | q]  -> predict value bound to q at the last step."""
    K = n_symbols                      # keys:   0..K-1
    V = n_symbols                      # values: K..2K-1
    SEP = 2 * n_symbols                # separator
    vocab = 2 * n_symbols + 1
    L = 2 * n_pairs + 2                # pairs + SEP + query
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -1, dtype=torch.long)
    for b in range(batch):
        keys = torch.randperm(K)[:n_pairs]
        vals = torch.randint(0, V, (n_pairs,))
        seq = []
        for k, v in zip(keys, vals):
            seq += [k.item(), K + v.item()]
        seq.append(SEP)
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq.append(keys[qi].item())
        x[b] = torch.tensor(seq)
        y[b, -1] = K + vals[qi].item()  # predict the queried value at final pos
    return x.to(device), y.to(device), vocab


def gen_copy(batch, length=16, n_symbols=16, device="cpu"):
    """[s0..s_{T-1} COPY 0..0] -> reproduce s at the second half."""
    DELIM = n_symbols
    BLANK = n_symbols + 1
    vocab = n_symbols + 2
    L = 2 * length + 1
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -1, dtype=torch.long)
    for b in range(batch):
        s = torch.randint(0, n_symbols, (length,))
        seq = s.tolist() + [DELIM] + [BLANK] * length
        x[b] = torch.tensor(seq)
        y[b, length + 1:] = s            # score the copied-out region
    return x.to(device), y.to(device), vocab


def gen_needle(batch, fill=64, n_symbols=16, device="cpu"):
    """filler...filler KEY VALUE filler...filler SEP KEY -> predict VALUE."""
    KEY = n_symbols
    SEP = n_symbols + 1
    vocab = n_symbols + 2
    L = fill + 4
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -1, dtype=torch.long)
    for b in range(batch):
        seq = torch.randint(0, n_symbols, (fill,)).tolist()
        val = torch.randint(0, n_symbols, (1,)).item()
        pos = torch.randint(0, fill - 1, (1,)).item()
        seq[pos] = KEY
        seq[pos + 1] = val               # the needle
        seq += [SEP, KEY]
        x[b] = torch.tensor(seq)
        y[b, -1] = val                   # predict the needle's value
    return x.to(device), y.to(device), vocab


TASKS = {
    "recall": gen_recall,
    "copy": gen_copy,
    "needle": gen_needle,
}

# train/eval one (task, mixer)


def run(task, mixer, steps, dial, device):
    gen = TASKS[task]
    _, _, vocab = gen(1, device=device)
    sample_x, _, _ = gen(1, device=device)
    seq_len = sample_x.shape[1]
    cfg = Config.from_dial(dial, mixer=mixer, seq_len=seq_len, vocab_size=vocab)
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    for _ in range(steps):
        x, y, _ = gen(64, device=device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            x, y, _ = gen(128, device=device)
            logits, _ = model(x)
            pred = logits.argmax(-1)
            mask = y != -1
            correct += (pred[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all"] + list(TASKS))
    p.add_argument("--mixers", nargs="+", default=["attention", "ssm", "hybrid"])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dial", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    print(f"\ndiagnostics  (dial={args.dial}, steps={args.steps}, device={args.device})")
    print("accuracy on held-out samples — higher is better\n")
    header = f"{'task':<10}" + "".join(f"{m:>12}" for m in args.mixers)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = f"{task:<10}"
        for mixer in args.mixers:
            acc = run(task, mixer, args.steps, args.dial, args.device)
            row += f"{acc:>12.2%}"
        print(row)
    print()


if __name__ == "__main__":
    main()
