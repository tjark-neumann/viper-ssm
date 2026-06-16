"""
The selective scan — the heart of a state-space model.

A linear recurrence over a sequence:

    h_t = a_t * h_{t-1} + b_t
    y_t = c_t . h_t   (+ D * u_t)

where, for the selective (input-dependent) SSM, the coefficients are produced
*from the input* at every step:

    a_t = exp(delta_t * A)           (the "decay", always in (0, 1) since A < 0)
    b_t = delta_t * B_t * u_t        (the discretised input)

Two implementations live here, and they must agree:

  * `selective_scan_sequential` — the obvious O(L) loop. The reference. Always
    correct. This is what you read to understand what an SSM *is*.

  * `selective_scan_parallel`   — the same recurrence in closed form, computed
    with cumulative products/sums so a whole chunk is done in parallel, then
    sequentially carried across chunks. This is where the "linear-time but still
    fast on a GPU" claim actually lives. Validated against the sequential version
    in tests/test_scan.py.

The closed form: unrolling the recurrence gives

    h_t = sum_{j<=t} ( prod_{k=j+1..t} a_k ) * b_j

Let P_t = prod_{k<=t} a_k = exp(cumsum(log a)). Then prod_{j+1..t} a_k = P_t / P_j
and so h_t = P_t * cumsum( b_j / P_j ). That is the parallel form.

THE CATCH: P_j is a cumulative product that decays toward zero, so b_j / P_j
overflows on any non-trivial sequence. The fix (and what the parallel scan below
actually does) is to *reset* the cumulative product at every chunk boundary and
carry the hidden state across chunks by hand — bounding the dynamic range to one
chunk's worth of decay. Parallel within a chunk, sequential across them. Default
everywhere is `sequential`; switch to `parallel` for speed once you trust it.
"""

import torch


def selective_scan_sequential(u, delta, A, B, C, D):
    """Reference scan. Loops over time. Correct by construction.

    Shapes (b=batch, l=length, d=inner channels, n=state dim):
        u, delta : (b, l, d)
        A        : (d, n)        negative
        B, C     : (b, l, n)
        D        : (d,)
    Returns y : (b, l, d)
    """
    b, l, d = u.shape
    n = A.shape[1]
    # discretise
    dA = torch.exp(delta.unsqueeze(-1) * A)              # (b, l, d, n)
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, l, d, n)

    h = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(l):
        h = dA[:, t] * h + dBu[:, t]                     # (b, d, n)
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t])     # (b, d)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                           # (b, l, d)
    return y + u * D


def selective_scan_parallel(u, delta, A, B, C, D, chunk=8):
    """Chunked closed-form scan. Parallel *within* a chunk, sequential *across*
    chunks. Same signature/return as the sequential version, and it matches it to
    tolerance (see tests/test_scan.py).

    Why chunked: the naive one-shot closed form h_t = P_t * cumsum(b_j / P_j)
    divides by a cumulative product P_j that decays toward zero, so it overflows
    on any non-trivial sequence. Resetting the cumulative product at each chunk
    boundary bounds its dynamic range to `chunk` steps, which is stable, while
    still doing the expensive part (the within-chunk scan) in parallel. This is
    the same idea behind the chunked/SSD algorithms used in real implementations.
    """
    b, l, d = u.shape
    log_a = delta.unsqueeze(-1) * A                              # (b, l, d, n)
    b_term = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, l, d, n)

    n = A.shape[1]
    h_carry = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    for s in range(0, l, chunk):
        e = min(s + chunk, l)
        la = log_a[:, s:e]                                       # (b, lc, d, n)
        bt = b_term[:, s:e]
        Cc = C[:, s:e]                                           # (b, lc, n)
        P = torch.exp(torch.cumsum(la, dim=1))                  # within-chunk, in (0, 1]
        h_fresh = P * torch.cumsum(bt / P, dim=1)               # zero initial state
        h = h_fresh + P * h_carry.unsqueeze(1)                  # add carried state
        ys.append(torch.einsum("bldn,bln->bld", h, Cc))
        h_carry = h[:, -1]                                       # state leaving chunk
    y = torch.cat(ys, dim=1)
    return y + u * D


SCANS = {
    "sequential": selective_scan_sequential,
    "parallel": selective_scan_parallel,
}


def get_scan(name):
    if name not in SCANS:
        raise ValueError(f"unknown scan '{name}', choose from {list(SCANS)}")
    return SCANS[name]
