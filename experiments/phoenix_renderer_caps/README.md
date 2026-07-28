# Phoenix renderer-cap sweep

This experiment asks whether Phoenix 4.0's strong private transfer still leaves
useful context on the floor. It holds base `Qwen/Qwen3.5-9B`, the exact factual
deception rubric, no-thinking direct `Prediction:` margins, label verbalizers,
and rows fixed while changing only renderer character caps and the maximum
model window.

The four preregistered conditions are:

| condition | per-message cap | context cap | output cap | token window |
| --- | ---: | ---: | ---: | ---: |
| `current_3k_2k` | 1,500 | 3,000 | 3,000 | 2,048 |
| `double_caps_2k` | 3,000 | 6,000 | 6,000 | 2,048 |
| `double_caps_4k` | 3,000 | 6,000 | 6,000 | 4,096 |
| `quad_caps_8k` | 6,000 | 12,000 | 12,000 | 8,192 |

The second arm isolates character-cap effects under the existing model window;
the third and fourth permit genuinely longer inference. All conditions preserve
Phoenix 4.0's message boundaries, head/tail retention, and explicit truncation
marker.

The existing `48/32/16` length tiers remain. Because a batch of sixteen
8,000-token prompts would be unsafe, the sweep additionally caps padded input
work at 32,768 tokens per request, reducing very-long batches below sixteen
while preserving the established short `48×600` tier.
The 822-row local validation audit found **zero** character-cap or token-window
hits in every arm: its prompt lengths are 232--844 tokens. It is therefore a
renderer-parity check, but cannot measure the effect of retaining more context.

The GPU sweep instead uses the public Liars' Bench Insider Trading corpus. Its
long conversations exercise the caps and make the input-token/runtime tradeoff
observable. This corpus is a structural long-context diagnostic, not a proxy
label for private Notus and not a new route or tuning set. The report includes
cap-hit counts, untruncated/effective token quantiles, batch shapes, total
padded tokens, AUROC, source-model AUROC, score uniqueness, and scoring-only
wall time.

Run the CPU/tokenizer audit before allocating a GPU:

```bash
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
python experiments/phoenix_renderer_caps/run.py --audit-only
python experiments/phoenix_renderer_caps/run.py \
  --corpus insider-trading --audit-only
```

Then run all four scoring arms in one persistent vLLM process:

```bash
sbatch experiments/phoenix_renderer_caps/run.sh
```

For the hosted leaderboard backend, first exercise the maximum-padded batch
from each arm:

```bash
python experiments/phoenix_renderer_caps/run.py \
  --corpus insider-trading --backend ndif --ndif-smoke-only \
  --output-dir results/blackbox/phoenix_renderer_caps_ndif_smoke_v1
```

If all four memory smokes pass, remove `--ndif-smoke-only` for the full run.
The NDIF path uses the same normalized literal-label logits and
`logits_to_keep=1`; it groups 32 padded forward traces per remote session so
the 430--444 batches do not become one oversized intervention graph.

The selected fast diagnostic uses one organism and 400 balanced rows:

```bash
python experiments/phoenix_renderer_caps/run.py \
  --corpus insider-trading --backend ndif \
  --source-model llama-v3.3-70b-instruct --sample-size 400 \
  --output-dir results/blackbox/phoenix_renderer_caps_ndif_llama400_v1
```

The sample is frozen with seed `20260728` and contains 200 rows per label.

## Result

All peak-work NDIF smokes passed up to 32,768 padded tokens. The frozen
Llama-3.3 sample then produced:

| condition | input tokens | NDIF seconds | AUROC |
| --- | ---: | ---: | ---: |
| current 3K chars / 2K window | 390,768 | 26.4 | 0.75839 |
| 6K chars / 2K window | 686,375 | 48.1 | 0.54149 |
| 6K chars / 4K window | 686,375 | 41.5 | 0.54149 |
| 12K chars / 8K window | 707,760 | 42.9 | 0.49574 |

All 400 selected prompts remained below 2,048 tokens. The two 6K arms were
therefore byte-identical and returned exactly identical scores; their runtime
spread estimates NDIF load/queue noise. Relative to current, 6K increased
input work by 75.7% and took 57--82% longer, while 12K increased input work by
81.1% and took 62% longer.

The AUROC delta against current was `-0.21690` for 6K (paired stratified
bootstrap 95% interval `[-0.26786, -0.16706]`) and `-0.26265` for 12K
(`[-0.31514, -0.21118]`). Longer context is actively harmful on this public
action-deception diagnostic. This does not identify private Notus composition,
but it rejects raising Phoenix's caps merely because more context is available.

Artifacts are written under
`results/blackbox/phoenix_renderer_caps_validation_v1/` and
`results/blackbox/phoenix_renderer_caps_insider_trading_v1/`; both remain
ignored.
