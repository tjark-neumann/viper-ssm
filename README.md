# viper-ssm

A minimal, hackable, single-codebase playground for the question every "is attention all you need?" thread eventually reaches:

**where does a state-space model actually beat a Transformer, and where does it lose?**

Every nano-style repo in the lineage (minGPT → nanoGPT → nanochat) is a Transformer. viper-ssm keeps the same readable, dependency-light spirit but makes the **token-mixer a single flag**. The *exact same* model — same embeddings, norms, MLPs, residuals, training loop — runs as causal attention, as a Mamba-style selective SSM, or as a hybrid of the two. Then it hands you the two things you need to form an opinion: a **diagnostic suite** that probes the capabilities the architectures genuinely differ on, and a **speedrun** that shows the linear-vs-quadratic scaling on your own GPU.

It's small enough to read in an afternoon (~700 lines of core code, torch + numpy only) and cheap enough to run the experiments for the price of coffee.

```bash
pip install -r requirements.txt
python data/prepare.py                              # ~1MB of Shakespeare

python train.py --mixer attention --dial 6 --steps 2000
python train.py --mixer ssm       --dial 6 --steps 2000
python train.py --mixer hybrid    --dial 6 --steps 2000   # one flag — that's the experiment

python sample.py --ckpt model.pt --prompt "ROMEO:"
python diagnostics.py                               # the comparison table
python speedrun.py                                  # the scaling table
```

## The one idea

A decoder-only LM where each block is `norm → mixer → residual, norm → MLP → residual`, and `mixer` is chosen per layer:

| `--mixer`     | what it is                                              | compute | inference state |
|---------------|---------------------------------------------------------|---------|-----------------|
| `attention`   | multi-head causal attention                             | O(L²)   | KV cache grows with context |
| `ssm`         | Mamba-style selective state-space (S6) + short conv + gate | O(L) | fixed-size hidden state |
| `hybrid`      | mostly SSM, 1-in-N layers attention (Jamba-style)       | ~O(L)   | small |

Holding everything else fixed makes the comparison a controlled experiment: the only variable is how tokens talk to each other.

## The diagnostics are the point

`diagnostics.py` runs three synthetic tasks, each chosen because it's where the architecture debate actually lives:

- **recall** — associative recall. See key/value pairs, then a query key, predict its value. Direct lookup favors attention; a fixed-state SSM has to have *stored* the right binding.
- **copy** — reproduce a random sequence verbatim after a delimiter. Pure long-range exact memory.
- **needle** — one key/value pair hidden in a long stream of filler, queried at the end. Retrieval over distance.

```
$ python diagnostics.py
diagnostics  (dial=4, steps=400, device=cuda)
accuracy on held-out samples — higher is better

task         attention         ssm      hybrid
--------------------------------------------------
recall            ...          ...         ...
copy              ...          ...         ...
needle            ...          ...         ...
```

The headline isn't "X wins." It's the **shape** of where each architecture is strong — and watching `hybrid` recover attention's strengths at a fraction of the cost. That table, reproducible in minutes, is the contribution.

## The scan is the teeth

The SSM's whole pitch is linear-time sequence mixing, and that lives in the **selective scan** (`nmc/scan.py`). Two implementations are included and are tested to agree:

- `sequential` — the obvious O(L) loop. The reference. Read this to understand what an SSM *is*.
- `parallel` — the same recurrence in closed form, done with cumulative products/sums **chunked** so the cumulative product can't underflow (parallel within a chunk, carried across chunks — the same trick the real SSD kernels use). No custom CUDA.

```bash
python tests/test_scan.py        # asserts parallel == sequential
python train.py --mixer ssm --scan parallel ...
```

## The speedrun

`speedrun.py` sweeps context length and reports throughput per mixer. On a GPU at long context you watch attention's curve bend upward while the SSM stays flat — the linear-time claim, with your own eyes, no $100 run required.

> Note: on CPU at short context, attention's fused kernel wins; the SSM advantage is a GPU-at-long-context phenomenon. Use `--scan parallel` and a real GPU to see the crossover.

## The dial

Model size is set by one number, like nanochat's `depth`. `--dial D` gives `n_layer = D`, `d_model = 64·D`, heads at `head_dim = 64`. You're not tuning a dozen knobs; you're picking a point on a curve.

## What's in here

```
nmc/
  config.py      the dial + per-layer mixer assignment
  scan.py        sequential + chunked-parallel selective scan  (the teeth)
  mixers.py      CausalSelfAttention | SelectiveSSM            (the swap)
  model.py       embeddings, blocks, head, generate
  data.py        char-level loader, no tokenizer dependency
train.py         pretrain on a text file
sample.py        generate from a checkpoint
diagnostics.py   recall / copy / needle, swept across mixers   (the point)
speedrun.py      throughput vs context length                  (the scaling)
tests/test_scan.py
```

## Roadmap → the full viper-ssm

This v0 is the comparison engine and a char-level pretrainer. The path to a full nanochat-style assistant, in order:

- [ ] BPE tokenizer (drop-in replacement for `data.py`)
- [ ] midtraining + SFT on a small instruction set → an actual chat model
- [ ] a minimal RL stage (GRPO) on a reasoning task
- [ ] a tiny web UI to talk to it — and to *feel* the long-context, constant-memory inference the SSM gives you
- [ ] a proper associative (Blelloch) parallel scan as a third `--scan` option
- [ ] a "time-to-target" speedrun leaderboard

## Credit

Inspired by Andrej Karpathy's nanoGPT / nanochat, and by the Mamba line of work (Gu & Dao). Built to be read, forked, and argued with.

## License

MIT.
