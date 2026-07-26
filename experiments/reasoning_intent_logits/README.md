# Semantic Reasoning-Intent Logits

Status: validation passed the frozen AUROC gate, and the unchanged primary
blend confirmed on local test.

This experiment tests an untrained `Qwen/Qwen3.5-9B` specialist over private
assistant reasoning. It is conditional by design: only trace-bearing rows are
used in analysis, while a submission would retain Phoenix unchanged when the
final assistant reasoning is absent.

Two frozen prompts share one local vLLM load:

- `literal_deceptive_intent` reads the trace as a final-intent classifier and
  ignores output correctness.
- `resolved_intent` additionally requires the final output to execute the
  ultimately adopted deceptive plan. This is the preregistered primary member.

Both prompts end at `Prediction:` and constrain the next generated token to
the single-token strings `0` or `1`. Their normalized next-token probabilities provide continuous
scores without autoregressive reasoning generation.

Run validation:

```bash
sbatch experiments/reasoning_intent_logits/run.sh split=validation
```

Then analyze against the frozen Phoenix 4.1 rank-16 direct-margin validation
cache:

```bash
python experiments/reasoning_intent_logits/analyze.py \
  --validation-phoenix \
    results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation_phoenix_v3_auroc_margin_sweep_v1/generations.jsonl \
  --validation-intent-root \
    results/blackbox/qwen9b_reasoning_intent_logits_v1/validation/member_predictions \
  --output-dir results/blackbox/reasoning_intent_logits_phoenix_v1
```

The AUROC-first analyzer reports fixed probability-mean, log-odds-mean,
maximum, and per-dataset rank-average blends plus a nested
leave-one-dataset-unit-out logistic diagnostic. It also reports score
uniqueness, cross-label ties, and saturation. The fixed log-odds mean is
primary. The resolved-intent member advances to frozen local test only if:

- standalone varied validation macro AUROC is at least `0.80`; and
- fixed log-odds-mean varied macro AUROC improves Phoenix 4.1 direct margins by
  at least `0.005`.

Binary metrics and the nested logistic model remain diagnostics; they do not
govern AUROC-first promotion. Raw OR is also reported as a diagnostic and is
not the preferred combiner. A submission should use the blend only for the
continuous `score` column and retain Phoenix 4.1's separate `deceptive` decision.

## Validation outcome

Slurm job `30297634` evaluated all 1,644 prompt/row pairs in one Qwen load.
The one-token scoring pass took `31.91s` (`51.52` prompt evaluations/s) with
zero parse failures. A separate zero-byte historical `result.json` caused the
optional global leaderboard renderer to exit nonzero after this run's complete
predictions and result had already been written; the renderer now skips
incomplete caches.

On the nine trace-bearing varied validation units:

| member/blend | macro AUROC | delta vs Phoenix |
| --- | ---: | ---: |
| Phoenix 4.1 direct margin | `0.86431` | — |
| literal intent alone | `0.85069` | `-0.01361` |
| literal + Phoenix log-odds mean | **`0.88833`** | `+0.02403` |
| resolved intent alone | `0.84889` | `-0.01542` |
| resolved + Phoenix log-odds mean | **`0.88514`** | **`+0.02083`** |
| resolved + Phoenix nested logistic | `0.87875` | `+0.01444` |

The preregistered resolved-intent log-odds mean passed both gates: standalone
AUROC exceeded `0.80`, and the blend gained more than `0.005`. It emitted 351
distinct scores across 360 rows, with macro cross-label tie fraction
`0.00028` and no extreme saturation. If the 12 no-trace units retain their
exact Phoenix scores, the validation 21-unit macro projects from `0.93815` to
`0.94708`.

The literal member is a diagnostic control and is not promoted post hoc despite
its slightly higher validation blend. The frozen resolved prompt and log-odds
mean advanced unchanged to one local-test confirmation:

```bash
sbatch experiments/reasoning_intent_logits/run.sh split=test
```

## Frozen test outcome

Slurm job `30297674` completed in `2m27s` including cold model startup. Its
1,642 two-prompt evaluations took `30.68s` (`53.52` prompt evaluations/s).
The deployed primary would issue only one intent prompt on the 360
trace-bearing rows, rather than both prompts on all 821 rows; NDIF runtime still
requires a paired full-unit benchmark.

On the nine trace-bearing varied test units:

| member/blend | macro AUROC | delta vs Phoenix |
| --- | ---: | ---: |
| Phoenix 4.1 direct margin | `0.87083` | — |
| resolved intent alone | `0.82986` | `-0.04097` |
| resolved + Phoenix log-odds mean | **`0.88764`** | **`+0.01681`** |
| resolved + Phoenix nested logistic | `0.87847` | `+0.00764` |
| literal + Phoenix log-odds mean, diagnostic | `0.89028` | `+0.01944` |

The fixed primary therefore transferred without retuning and again beat the
fitted diagnostic stack. It produced 350 unique scores across 360 rows, a
macro cross-label tie fraction of `0.00028`, and no extreme saturation.
Keeping the 12 no-trace units unchanged projects full 21-unit test macro AUROC
from `0.93423` to **`0.94143`** (`+0.00720`).

This supports an NDIF implementation/benchmark, not immediate leaderboard
promotion: every local trace-bearing unit is Qwen-family, and the earlier
reasoning n-gram ensemble failed to transfer to the official distribution.

## Rank-1 intent-adapter follow-up

The historical `qwen9b_heterogeneous_resolved_intent_rank1_v1` adapter is a
rank-1/alpha-2 LoRA trained on 288 trace-bearing rows under the same resolved
intent objective. Validation job `30297707` compared direct and empty-summary
label margins without generating a reasoning summary.

On the nine trace-bearing validation units:

| intent member / Phoenix blend | macro AUROC |
| --- | ---: |
| base-Qwen resolved intent | `0.84889` |
| rank-1 direct margin | `0.89125` |
| rank-1 empty-summary margin | `0.88597` |
| Phoenix + rank-1 direct probability mean | `0.89861` |
| Phoenix + rank-1 direct log-odds mean | `0.89750` |
| Phoenix + rank-1 empty-summary log-odds mean | **`0.90111`** |

The adapter therefore materially improves the semantic member. The
empty-summary/log-odds blend is selected because blend AUROC, rather than
standalone intent AUROC, is the target. It has 358 unique scores over 360 rows
and zero cross-label ties. Keeping no-trace units unchanged projects full
validation macro AUROC `0.93815 -> 0.95393`, versus `0.94708` for the selected
base-Qwen blend.

The selected empty-summary pass scored all 822 validation rows in `24.03s`
with zero missing label logits. Deployment scores only the 360 trace-bearing
rows. Frozen test job `30297739` scored rank-1 intent at `0.88819` and the
selected Phoenix blend at **`0.90417`** on the nine trace-bearing varied units,
versus `0.87083` Phoenix (`+0.03333`). Seven units improved and two regressed.
The blend emitted 358 unique scores with zero cross-label ties; leaving the 12
no-trace units unchanged projects full test macro AUROC `0.93423 -> 0.94851`.

Matched validation-only base-control job `30297742` scored `0.88139`
standalone and `0.90028` blended, almost tying the adapter's `0.88597` and
`0.90111`. The rank-1 update therefore has only a small measured edge over its
exact prompt control. As a retrospective diagnostic, combining Phoenix with
the adapter's historical generated binary decisions scored `0.89792`
validation and `0.90139` test varied AUROC, improving eight of nine units on
each split.

Phoenix 4.2 deploys the frozen continuous adapter blend only on non-harm
trace-bearing rows. HP-KR harm-pressure rows use their existing specialist
exclusively and never invoke intent. A one-row trace-bearing NDIF notebook test
passed end to end in `221.82s`.
