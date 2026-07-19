# Why the whitebox submission failed in the sandbox while `--dry` passed

Diagnosed 2026-07-19/20. All nine whitebox failures (`whitebox_probe.ipynb` /
`sonic_v1.ipynb`, Jul 13–17, every one "failed on Dataset Metis" at a uniform
~38–48 s) had one deterministic root cause, reproduced locally and fixed.

## Root cause

The leaderboard sandbox wraps notebooks in a **Landlock policy that denies
`/proc`**. Torch lazily initializes its `cpuinfo` library from `/proc/cpuinfo`,
and the CPU **fp16 → fp32 cast kernel** hard-fails without it:

```
RuntimeError: Failed to initialize cpuinfo!
```

The notebook downloads activations from NDIF as float16 and cast them with
`.to(torch.float32)` in a cell without a try/except → the notebook died → the
leaderboard reported the generic, redacted "failed to run in the sandbox"
error on the first dataset in run order.

Key facts that explain the confusing symptoms:

- **`--dry` cannot see this failure class.** The dry run executes the same
  pipeline but with `confine=False` — no Landlock/seccomp/egress — so the
  fp16 cast works locally and only breaks on the Space.
- **"Dataset Metis" is the first dataset in run order**: the instructed-
  deception task, whose first combo is **Nemotron-120B with no LoRA** (visible
  in a passing run's per-dataset breakdown from `/api/me`). The ~38 s uniform
  runtime = venv/pip/imports, then death at the first sizeable torch CPU op —
  before any real remote work.
- **The gated-LoRA-403 theory (Jul 17) is disproven**: the three hidden repos
  (`g-st-gemma-3-27b-it-1/-3`, `a-mo-qwen3.5-27b-2`) appear in *none* of the
  29 datasets a submission actually runs, and the failing dataset has no LoRA
  at all. Also ruled out: the NDIF team key (single traces *and* multi-trace
  sessions verified working on both Qwen-27B and Nemotron-120B), Nemotron
  deployment status (running, pinned), zip packaging (`*.pt` probe weights are
  included), and `torch.load` weights-only incompatibilities.
- **The try/except "armor" masks failures in `--dry`**: when a step fails, the
  notebook writes fallback 0.5 scores and the dry run still prints "✓ OK" with
  blank metrics and a ~20 s runtime. Blank metrics + tiny runtime = red flag;
  check score spread, not the checkmark.

## Reproduction

The runner in `leaderboard/` was driven locally with the real confinement
stack (`sandbox=True, confine=True, enforce_egress=True` — WSL2's kernel has
Landlock and seccomp; the project `.venv` added to the Landlock RO paths).
The unmodified notebook crashed with exactly the cpuinfo error. An op-by-op
audit inside the sandbox on a **CPU-only torch wheel** (2.13.0+cpu, the same
install path as the Space image) showed the fp16→fp32 cast is the *only*
failing operation in the notebook's client-side inventory — probe
construction, the full TransformerEncoder forward, fp32 cat/copy/matmul, and
numpy conversions all pass.

## Fix (applied to `submission/sonic_v1.ipynb`, 2026-07-20)

1. Convert downloaded fp16 activations via **numpy** —
   `flat.cpu().numpy().astype(np.float32)` + `np.clip` + `torch.from_numpy` —
   never a torch CPU cast. fp16 is kept remote-side only (halves the download).
2. Removed the now-redundant fp32 cast in the scoring cell.
3. Armored the remaining unwrapped cells (probe config load, probe build,
   scoring) so any surprise degrades to a valid fallback CSV instead of a
   crashed run.
4. Fixed the notebook's nbformat version declaration (4.4 → 4.5).

## Validation

- **Confined pipeline** (real Landlock + seccomp + egress proxy + rlimits):
  `ok=True` on both the Nemotron and Qwen a-mo-1 dev datasets with real,
  distinct probe scores (not the fallback).
- **Signal quality on full dev sets**: Qwen a-mo-1 → AUROC 1.0,
  balanced accuracy @0.5 = 0.995; Nemotron → AUROC 0.999, balanced
  accuracy = 0.969. The default 0.5 threshold is fine.

Residual risk: the Space's exact image can't be byte-matched locally, so final
certainty only comes from the next real submission — but the known crash is
deterministically fixed under the real confinement stack, and the worst case
is now a scored run rather than an error. Leaderboard rate limit as of
2026-07-19: one submission per 30-minute window.
