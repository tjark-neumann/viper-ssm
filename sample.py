"""
Sample from a trained checkpoint.

    python sample.py --ckpt model.pt --prompt "ROMEO:" --tokens 400
"""

import argparse
import torch

from nmc import Config, LM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="model.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = LM(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    stoi, itos = ck["stoi"], ck["itos"]

    prompt = [c for c in args.prompt if c in stoi]
    if not prompt:
        prompt = [next(iter(stoi))]  # fall back to first known char
        print("(prompt had no in-vocab characters; starting from a default token)")
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[int(i)] for i in out[0]))


if __name__ == "__main__":
    main()
