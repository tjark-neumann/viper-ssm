"""
Pretrain a char-level model.

    python train.py --mixer ssm        --dial 6 --steps 2000
    python train.py --mixer attention  --dial 6 --steps 2000
    python train.py --mixer hybrid     --dial 6 --steps 2000

Same command, one flag changes the architecture. That is the experiment.
"""

import argparse
import time
import torch

from nmc import Config, LM, CharData


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--mixer", default="ssm", choices=["attention", "ssm", "hybrid"])
    p.add_argument("--dial", type=int, default=6)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--scan", default="sequential", choices=["sequential", "parallel"])
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--out", default="model.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data = CharData(args.data, args.seq_len, device=args.device)
    cfg = Config.from_dial(
        args.dial, mixer=args.mixer, seq_len=args.seq_len,
        vocab_size=data.vocab_size, scan=args.scan,
    )
    model = LM(cfg).to(args.device)
    print(f"[{args.mixer}] dial={args.dial}  params={model.num_params()/1e6:.2f}M  "
          f"vocab={data.vocab_size}  device={args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = data.batch("val", args.batch_size)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = data.batch("train", args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == 1:
            vl = evaluate()
            dt = time.time() - t0
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  "
                  f"({dt:.1f}s)")

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "stoi": data.stoi, "itos": data.itos}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
