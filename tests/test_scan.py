"""
The one test that matters: the fast parallel scan must agree with the slow,
obviously-correct sequential scan. If this passes, the SSM you train with
--scan parallel computes the same thing as --scan sequential.

    python -m pytest tests/ -q      # or just: python tests/test_scan.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nmc.scan import selective_scan_sequential, selective_scan_parallel


def _random_inputs(b=2, l=64, d=8, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, l, d, generator=g)
    delta = torch.nn.functional.softplus(torch.randn(b, l, d, generator=g))
    A = -torch.exp(torch.randn(d, n, generator=g))      # negative -> stable decay
    B = torch.randn(b, l, n, generator=g)
    C = torch.randn(b, l, n, generator=g)
    D = torch.randn(d, generator=g)
    return u, delta, A, B, C, D


def test_parallel_matches_sequential():
    inp = _random_inputs()
    y_seq = selective_scan_sequential(*inp)
    y_par = selective_scan_parallel(*inp)
    max_err = (y_seq - y_par).abs().max().item()
    assert max_err < 1e-3, f"scans disagree, max err {max_err}"
    print(f"ok: parallel matches sequential (max err {max_err:.2e})")


if __name__ == "__main__":
    test_parallel_matches_sequential()
