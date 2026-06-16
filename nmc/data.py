"""
Char-level data. The whole point is to stay dependency-light: a text file in,
integer tensors out, no tokenizer library. (A real BPE tokenizer is a roadmap
item; char-level is enough to train a coherent model and to run every diagnostic.)
"""

import os
import torch


class CharData:
    def __init__(self, path, seq_len, device="cpu"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        self.device = device
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train = data[:n]
        self.val = data[n:]

    def batch(self, split, batch_size):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i + self.seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.seq_len] for i in ix])
        return x.to(self.device), y.to(self.device)

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, t):
        return "".join(self.itos[int(i)] for i in t)
