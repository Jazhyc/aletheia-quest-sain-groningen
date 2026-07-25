# The dev datasets are ordered honest-first

Found on 2026-07-24 while evaluating sonic_v2.3.7. It invalidates any evaluation
that takes a prefix of a dev dataset.

## The fact

Every dev dataset has 400 rows and is exactly 50% deceptive. The rows are not
shuffled. The honest rows come first.

| dataset | first 20 | first 80 | whole set |
| --- | ---: | ---: | ---: |
| `dev-instructed-deception-Qwen3.5-27B-None` | 0/20 | 0/80 | 0.500 |
| `dev-varied-deception-Qwen3.5-27B-None` | 0/20 | 0/80 | 0.500 |
| `dev-instructed-deception-gemma-3-27b-it-None` | 0/20 | 0/80 | 0.500 |

Reproduce:

```bash
.venv/bin/python -c "
import numpy as np
d = np.load('results/whitebox/activations/'
            'aletheias-quest__dev-instructed-deception-Qwen3.5-27B-None.tokens.npz')
y = d['deceptive'].astype(int)
print(y[:80].sum(), '/80 deceptive,', y.mean(), 'overall')"
```

## Why it matters

A prefix of a dev dataset contains one class only. Every metric that needs two
classes then breaks, and it breaks quietly:

- **AUROC is undefined.** The tools print `—` or `nan`, not an error.
- **Balanced accuracy is undefined** for the same reason.
- **A judge looks broken when it is correct.** On 80 honest rows a good judge
  returns `positive_rate = 0.000`. That looks identical to a judge whose output
  never parsed.
- **The gate degenerates.** All-honest judge verdicts are single-class, so the
  gate takes its `w = 0.5` fallback and the threshold falls back to `0.5`.

## What this broke

Two sonic_v2.3.7 measurements, both of which looked like product faults and were
not:

1. `python submit.py --dry --limit 20` reported `Bal.Acc — AUROC — Recall —`
   on all 21 datasets. The notebook was fine. `--limit` forwards
   `ALETHEIA_LIMIT`, the notebook does `ds.select(range(LIMIT))`, and 20 honest
   rows cannot produce a metric.
2. A judge prompt A/B with `--limit 80` returned `positive_rate = 0.000` on both
   arms. Both arms were correct. They were judging honest rows.

## The rule

**Never take a prefix of a dev dataset to measure anything.** Sample across the
whole dataset.

`experiments/ensemble_gate_eval/run_judge_local.py` now does this: its `--limit`
takes an evenly spaced sample rather than the first N rows. On
`dev-instructed-deception-Qwen3.5-27B-None` a limit of 80 gives 40/80 deceptive
instead of 0/80.

`submit.py --dry --limit N` still takes a prefix, because the limit is applied
inside the notebook by `ds.select(range(LIMIT))` and that code is part of the
submission. Treat a limited dry run as a **smoke test only** — it tells you the
notebook runs, the sandbox is healthy and the detectors fire. It cannot tell you
whether accuracy regressed. For that, run `--dry` with no limit.

## Related

The same shape of trap appears elsewhere in this repo's evaluation: see
`experiments/contrastive_probe/EVALUATION.md` for the separate finding that
every local leave-one-organism-out fold is easier than every real Notus unit.
Both are cases where a convenient local proxy silently answers a different
question than the one asked.
