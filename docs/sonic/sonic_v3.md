# sonic v3 — direct-logit judge, fixed blend, no batch statistics

> Status: **submitted 2026-07-26 00:41.** Official headline AUROC `0.862417` /
> BA `0.621250`, runtime 29m57s. **Regressed** against the pure judge
> (phoenix_wright_v4 at `0.893817`). Superseded by `sonic_v3_1.ipynb` at
> `0.902083`. See `docs/official_submissions.md` for the counted-unit breakdown.

## Why v3 exists

On 2026-07-25 the organizers made two changes:

- **The prize metric changed to mean per-dataset AUROC.** Balanced accuracy
  stops counting from Monday. The threshold in the notebook no longer scores.
- **The organizers discouraged batch-derived statistics.** A method must not
  use the test set's class balance to decide its threshold. The rules did not
  explicitly forbid batch statistics — that is why many submissions were not
  disqualified — but the preference is for per-sample scoring.

sonic v2.3.7 used batch statistics. Its gate ranked the detectors within the
batch. It measured the probe/judge agreement on the batch. It estimated the
positive rate from the judge. It cut at a quantile of the batch's own scores.
Under AUROC, none of this earns anything.

v3 removes all batch-derived quantities. It spends the freed complexity on
ranking quality.

## The judge

The judge is `Qwen3.5-9B` with a frozen rank-16 deception LoRA adapter
(`Jazhyc/aletheias-phoenix-wright-v2-adapter`). This is the same adapter that
the Phoenix v2 and v3 official submissions used.

The prompt is the same prompt used by Phoenix v3, Phoenix v4, and sonic
v2.3.7: a deception-judge instruction, the conversation as `ROLE: content`,
and the final assistant reply in `<output>` tags. The prompt is
chat-templated with thinking disabled. It has never changed.

The judge does not generate text. It appends `Prediction:` to the prompt. It
reads the next-token logits of the single tokens `0` and `1` in one forward
pass. v2.3.x generated up to 512 tokens of reasoning and parsed the verdict
from the text. v3 skips that step.

The judge runs on every dataset through NDIF. It runs in the same session as
the probe. One retry is allowed if the remaining time budget permits it.

The judge keeps the raw margin `logit_1 - logit_0`. The margin does not pass
through a sigmoid. A sigmoid saturates and ties confident rows. AUROC is
computed from the ranking, so ties are a direct cost.

## The probes

v3 inherits the v2.3.7 probes. The probes do not change.

Each base model has one probe. The probe is a transformer token probe at layer
46. It reads the final-token activation of the assistant's response.

Nemotron and qwen share a trunk. The trunk was trained across all three
families. Gemma keeps the v2.3.6 standalone probe.

See `docs/sonic/sonic_v2_3_7.md`.

## The blend

Every row is scored by a function of itself only:

    score = sigmoid((1 - λ) · probe_logit / PROBE_SD + λ · judge_margin / JUDGE_SD)

The constants:

| constant | value | source |
| --- | --- | --- |
| λ | 0.35 | fitted on dev (offline) |
| PROBE_SD (qwen) | 4.186 | shipped probe output, measured offline |
| PROBE_SD (gemma) | 6.466 | shipped probe output, measured offline |
| PROBE_SD (nemotron) | 2.194 | shipped probe output, measured offline |
| JUDGE_SD | 1.200 | local judge cache, measured offline |

The probe logit and the judge margin have different scales. Raw summed, the
probe dominates because its numbers are larger. λ would no longer express a
preference. Dividing each by its spread puts them on the same scale.

The means are not subtracted. A constant added to every row in a dataset
cannot change AUROC. Only the relative scale matters.

The blend uses no batch statistics. It does not rank the detectors within the
batch. It does not measure probe/judge agreement. It does not estimate the
positive rate. It does not cut at a quantile. It does not branch on the
dataset name.

## How the blend was fitted

`fit_direct_blend.py` used 20 leave-one-organism-out folds from the frozen
80/10/10 dev splits (`dev_splits/`). For each base model, the probe was refit
on the train rows of every organism *except* the held-out one. It then scored
the held-out rows. No fold's probe saw its own rows.

The judge cache is fixed. The judge model and adapter do not change. The same
cache serves every fold.

Mean per-dataset AUROC across the 20 folds:

| method | mean | worst fold |
| --- | ---: | ---: |
| judge alone (λ = 1) | 0.9509 | 0.7938 |
| probe alone (λ = 0) | 0.9646 | 0.8377 |
| **blend (λ = 0.35)** | **0.9685** | **0.8510** |

The blend beats both detectors on the mean and on the worst fold.

The blend is below the better single detector on 5 of 20 folds. The largest
loss is `0.0014`. All five folds are in the saturated instructed-deception
regime where every method scores above `0.996`.

The gains are in the hard regime: `+0.013` on both `varied/a-mo-1` and
`varied/a-mo-3`, the two worst folds.

A per-regime λ was measured. It gave `0.40` for base-model rows and `0.35`
for LoRA rows. It moved the overall mean by less than `1e-4`. Two free
parameters bought nothing. v3 ships one weight.

The optimum is flat. Mean AUROC varies by less than `0.005` across
λ ∈ [0.10, 0.60]. The exact value is not delicate.

## Per-dataset AUROC — all 21 dev units

Shipped probes (trained on all dev rows) and local judge cache. These numbers
are in-sample. The probes saw every row during training. The numbers describe
the best the method can attain. They are not a forecast of the official score.

| dataset | n | probe | judge | blend | regime |
| --- | ---: | ---: | ---: | ---: | --- |
| instr-nemotron-base | 216 | 0.9856 | 0.9905 | **0.9944** | base |
| instr-qwen-base | 400 | 0.9992 | 0.9948 | **0.9985** | base |
| instr-gemma-base | 400 | 0.9516 | 0.9973 | **0.9964** | base |
| instr-qwen-a-mo-1 | 400 | 1.0000 | 0.9968 | **1.0000** | LoRA |
| instr-qwen-a-mo-3 | 400 | 1.0000 | 0.9994 | **1.0000** | LoRA |
| instr-qwen-a-mo-4 | 400 | 0.9996 | 0.9984 | **0.9999** | LoRA |
| instr-qwen-a-mo-5 | 400 | 1.0000 | 0.9976 | **1.0000** | LoRA |
| instr-qwen-a-mo-6 | 400 | 0.9998 | 0.9985 | **0.9999** | LoRA |
| instr-qwen-a-mo-7 | 400 | 0.9999 | 0.9994 | **1.0000** | LoRA |
| instr-qwen-b-mo | 400 | 1.0000 | 0.9989 | **1.0000** | LoRA |
| instr-qwen-c-mo | 400 | 0.9956 | 0.9937 | **0.9959** | LoRA |
| instr-gemma-s-mo | 400 | 0.9178 | 0.9990 | **0.9995** | LoRA |
| varied-qwen-base | 400 | 0.9914 | 0.9521 | **0.9874** | base |
| varied-qwen-a-mo-1 | 400 | 0.9388 | 0.7938 | **0.9288** | LoRA |
| varied-qwen-a-mo-3 | 400 | 0.9600 | 0.7994 | **0.9506** | LoRA |
| varied-qwen-a-mo-4 | 400 | 0.9778 | 0.8730 | **0.9718** | LoRA |
| varied-qwen-a-mo-5 | 400 | 0.9824 | 0.9566 | **0.9884** | LoRA |
| varied-qwen-a-mo-6 | 400 | 0.9662 | 0.9005 | **0.9673** | LoRA |
| varied-qwen-a-mo-7 | 400 | 0.9723 | 0.8730 | **0.9653** | LoRA |
| varied-qwen-b-mo | 400 | 0.9838 | 0.9417 | **0.9822** | LoRA |
| varied-qwen-c-mo | 400 | 0.9871 | 0.9550 | **0.9885** | LoRA |
| **mean** | | 0.9814 | 0.9528 | **0.9864** | |
| **worst** | | 0.9178 | 0.7938 | **0.9288** | |

The blend beats every single detector on the mean and on the worst unit. The
two largest complementary gains: `instr-gemma-s-mo` where the judge rescues a
weak probe, and `varied-qwen-a-mo-1` where the probe rescues a confused judge.

## NDIF benchmark — comparison with Phoenix Wright 4.0

Both notebooks ran through the same harness (`benchmark_ndif.py`) on the same
dataset and row count. These are remote NDIF numbers, not leaderboard results.

| dataset | rows | method | AUROC | BA | time | unique |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| varied-qwen-base | 400 | sonic_v3 | **0.9837** | 0.915 | 63.5s | 400 |
| varied-qwen-base | 400 | phoenix_wright_v4.0 | 0.9394 | 0.865 | 22.5s | 83 |
| varied-qwen-a-mo-1 | 400 | sonic_v3 | 0.9064 | 0.700 | 68.7s | 400 |
| varied-qwen-c-mo | 400 | sonic_v3 | 0.9757 | 0.918 | 54.3s | 400 |
| instr-qwen-base | 400 | sonic_v3 | 0.9980 | 0.985 | 48.9s | 400 |
| instr-qwen-a-mo-1 | 400 | sonic_v3 | 0.9998 | 0.943 | 43.8s | 400 |
| instr-gemma-base | 400 | sonic_v3 | 0.9985 | 0.995 | 33.4s | 400 |
| instr-nemotron-base | 216 | sonic_v3 | 0.9941 | 0.906 | 49.1s | 216 |

The probe lifts the judge by `+0.044` on varied-qwen-base, the only dev unit
with the Notus profile (varied deception, no LoRA). Phoenix v4 produced 83
unique scores. Its sigmoid tied confident rows. The blend retires every tie
because the probe score is continuous.

## Thresholds

The `deceptive` column must still exist. v3 freezes one threshold per score
path. The thresholds are chosen offline for mean balanced accuracy across the
same dev folds. None is derived from the scored batch:

| path | threshold |
| --- | --- |
| blend | 0.50 |
| probe alone | 0.50 |
| judge alone | 0.53 |

The notebook asserts each threshold is finite before writing. The null
threshold trap that cost v2.3 `0.078` headline BA cannot fire.

## Fallbacks

Every path is monotone in the detector that survived. No path can collapse:

- No probe weights for the base model, or extraction failed → judge alone.
- Judge failed after its one retry → probe alone.
- Neither → a strictly increasing ramp. The column is never constant.

## Notus 

The gap to rank 1 is on Notus (base model, no LoRA). Our Notus AUROC is
`0.7577`. Tiresias's is `0.9032`.

The only Notus-shaped dev unit is `varied-qwen-base`. The local judge alone
scores `0.952`. The blend scores `0.987` in-sample and `0.980` leakage-free.

This is not a forecast. Counted Notus is one third Qwen, one third Nemotron,
and one third Gemma. The dev unit is a Qwen organism scored by a Qwen judge —
the most favourable case. The only official evidence for this judge across all
three families is Phoenix v2.0's Notus rows: `0.7475 / 0.7075 / 0.7450`
(binary, so AUROC equalled BA). Going from binary to a continuous margin is a
real gain. The honest central estimate for counted Notus is well below `0.95`.

## Reproducing

```bash
# 1. judge logits for every dev row (local GPU, ~25 min on a 4090)
python experiments/ensemble_gate_eval/run_direct_judge_local.py

# 2. leakage-free blend fit -> results/ensemble_gate_eval/sonic_v3_constants.json
python experiments/ensemble_gate_eval/fit_direct_blend.py

# 3. build the notebook from sonic_v2.3.7 plus the constants
python experiments/ensemble_gate_eval/build_sonic_v3_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_notebook.py

# 4. one dataset on NDIF
python experiments/ensemble_gate_eval/benchmark_ndif.py \
    --notebook submission/sonic_v3.ipynb --limit 400 \
    --output-dir results/sonic_v3/bench_varied_qwen_base

# 5. full 21-dataset AUROC table (local, no NDIF)
python experiments/ensemble_gate_eval/score_full_eval.py
```

Cells 2 and 4–9 of the notebook are byte-identical to
`legacy_submissions/sonic_v2.3.7.ipynb`. The probe pipeline is inherited, not
reimplemented.
