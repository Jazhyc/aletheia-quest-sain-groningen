# Phoenix adapter reason-then-score validation

This experiment measures corrected post-reasoning AUROC for the migrated
GPT-OSS privileged-information and GPT-5.6 Luna privileged-information
students on the complete local validation split.

For each adapter it:

1. renders the training-matched reasoning-summary prompt;
2. generates a deterministic judge reasoning summary and binary prediction;
3. removes the final generated `0|1` token;
4. reads normalized literal `0|1` logits at that position; and
5. compares the result with a fresh direct-boundary score from the same
   migrated adapter.

Generation uses Phoenix's length-aware `32/24/16` tiers, a 512-token allowance,
and a 32,768 padded input-plus-output token ceiling. Direct and
post-reasoning scoring share a second remote session and use `48/32/16` tiers
with `logits_to_keep=1`.

```bash
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
python experiments/phoenix_adapter_reasoning_ndif/run.py
```

Ignored artifacts are written under
`results/blackbox/phoenix_adapter_reasoning_ndif_v1/`.

## Full-validation result

The 2026-07-28 run scored all 822 rows across 21 validation units:

| adapter | boundary | macro AUROC | instructed | varied | pooled AUROC | unique scores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS PI | direct | **`0.94911`** | `0.99604` | `0.88653` | `0.96208` | `37/822` |
| GPT-OSS PI | post-reasoning | `0.92381` | `0.98812` | `0.83806` | `0.91134` | `46/822` |
| Luna PI | direct | **`0.94821`** | `0.98948` | `0.89319` | `0.96111` | `53/822` |
| Luna PI | post-reasoning | `0.92470` | `0.98104` | `0.84958` | `0.90740` | `90/822` |

Post-reasoning lost `0.02530` macro AUROC for GPT-OSS PI and `0.02351`
for Luna. It improved/tied/lost `3/5/13` GPT-OSS units and `7/1/13` Luna
units. The larger losses were in varied deception.

Generation used one optimized remote session per adapter, followed by one
session containing both direct and post-reasoning score arms. GPT-OSS
generation took 396.9 seconds and produced two 512-token completions without a
terminal label; both used the declared canonical-boundary fallback. Luna took
220.0 seconds and had no fallback. The one-token paired score passes took 49.9
and 48.5 seconds. Generation prompts were 230--842 tokens, post-reasoning
prompts were at most 1,150 tokens for GPT-OSS and 950 for Luna, and no prompt
was truncated.

Direct scoring remains selected. The reasoning chain increases score
granularity but worsens the ordering that determines competition AUROC.
