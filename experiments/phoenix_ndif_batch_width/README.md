# Phoenix NDIF short-batch-width sweep

This deployment experiment varies only Phoenix 6.2's short-prompt batch width:
`48`, `52`, or `56`. The medium and long tiers remain frozen at `32` above
600 tokens and `16` above 900 tokens. Every arm uses the exact Phoenix 6.2
Qwen-27B-soft adapter, byte-matched binary renderer, direct normalized `0|1`
margin, and `logits_to_keep=1`.

The sweep scores all 822 local-validation rows through competition NDIF. It
records forward-trace shapes, padded-token work, peak padded tokens, wall time,
per-dataset macro AUROC, and score drift relative to batch 48. Batch 64 is not
retested because it already OOMed against the hosted process allowance.

```bash
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
HF_DATASETS_OFFLINE=1 \
  python experiments/phoenix_ndif_batch_width/run.py
```

Artifacts are written under
`results/blackbox/phoenix_ndif_batch_width_v1/` and remain ignored. A wider
batch should be promoted only if it completes without OOM, provides a
repeatable material wall-time gain, and preserves macro AUROC within ordinary
reported batch-shape drift.

The follow-up replaces the tiers with a dynamic dual-cap scheduler. It takes
the largest next length-sorted batch satisfying both a row cap (`48` or `56`)
and `batch rows × longest prompt ≤ 28,800` padded tokens:

```bash
HF_DATASETS_OFFLINE=1 \
  python experiments/phoenix_ndif_batch_width/run_dynamic.py
```

The row cap is necessary even for prefill-only inference because Qwen3.5's
Gated DeltaNet maintains batch-shaped recurrent state. The padded-token cap
controls token activations. Token count alone is not treated as sufficient
because batch 64 previously OOMed while smaller-row shapes above 30,000 padded
tokens completed.

## Results

Both sweeps completed on all 822 validation rows:

| Scheduler | Time | Speedup vs fixed 48 | Traces | Padded tokens | Peak/trace | Macro AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 48/32/16 | 22.163s | 1.000x | 18 | 267,994 | 24,240 | 0.95375 |
| Fixed 52/32/16 | 20.747s | 1.068x | 17 | 266,438 | 26,728 | 0.95238 |
| Fixed 56/32/16 | 20.952s | 1.058x | 16 | 268,962 | 29,400 | 0.95387 |
| Dynamic, row cap 48 | 22.440s | 0.988x | 18 | 267,378 | 28,395 | 0.95327 |
| Dynamic, row cap 56 | 21.266s | 1.042x | 16 | 270,222 | 28,196 | 0.95387 |

The dynamic scheduler is viable, but it did not reduce trace count relative to
the matching fixed-width arm. Dynamic 56 was also slightly slower than fixed
56 despite producing the same reported AUROC, which is within the expected
noise of a single hosted timing. Fixed 52 and fixed 56 gained only 5.8--6.8%
over the frozen batch-48 reference, with the usual batch-shape score drift.

Do not change the Phoenix submission schedule from `48/32/16` based on this
run. If dynamic scheduling is revisited for a materially different private
length distribution, retain both the row and padded-token caps; do not use a
token-only budget.
