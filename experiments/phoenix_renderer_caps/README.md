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

Artifacts are written under
`results/blackbox/phoenix_renderer_caps_validation_v1/` and
`results/blackbox/phoenix_renderer_caps_insider_trading_v1/`; both remain
ignored.
