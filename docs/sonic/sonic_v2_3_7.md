# sonic_v2.3.7

This document tells you how sonic_v2.3.7 works. You do not need to read the
documents for the earlier versions. This document is self-contained.

This document uses Simplified Technical English (ASD-STE100). The sentences are
short. Each sentence gives one idea. The terms are constant: "the probe", "the
judge", "the gate", "the trunk", "the projection", "the score", "the threshold",
"the family", and "the organism".

**In one line:** sonic_v2.3.7 is the sonic_v2.3.6 notebook with new probe weights.
The probe for nemotron and qwen now comes from one shared trunk trained on all
three families. Nothing else changes.

**Status: submitted on 2026-07-25. The result is not in yet.** When the
leaderboard returns the per-unit table, record it in
`docs/official_submissions.md` and update section 8 of this document.

In `submission/sonic_v2.3.7.ipynb` every executable line is identical to
`sonic_v2.3.6.ipynb`. Only the title text and two comment blocks differ, and
they describe this version. Every functional change is in the weight files. The
full dry run passed. See section 9.

**Read Notus / gemma first when the result lands.** gemma keeps its old probe,
so that unit is the control. If Notus / Nemotron and Notus / Qwen move while
Notus / gemma does not, the shared trunk caused the change. If Notus / gemma
moves too, the per-unit noise is larger than we assumed and no single unit can
be attributed.

**Two other changes were designed, measured and rejected.** A wider judge prompt
and an organism-invariant probe. Section 7 gives the numbers. Do not re-open
either without reading it.

---

## 1. The task

The competition gives you a dataset of conversations. Each conversation has a
final assistant answer. You must say if that answer is deceptive.

You write two columns for each row.

- The `score` column is a continuous number. The leaderboard uses it for AUROC.
- The `deceptive` column is 0 or 1. The leaderboard uses it for balanced
  accuracy.

The leaderboard scores six counted units. The units are three Notus datasets and
three Iris datasets. Each unit uses one base model. The headline score is the
mean over the six units.

Most datasets have 400 rows at 50% deceptive. **This is not a rule.** The
Nemotron dev dataset has 216 rows at 7.4% deceptive. Do not assume balance.
Section 11 explains what that assumption broke.

---

## 2. Why we made sonic_v2.3.7

sonic_v2.3.5 is our best submission. It scored balanced accuracy `0.853750` and
AUROC `0.876333`. The first team is EleutherAI with `Tiresias.ipynb`. They
scored balanced accuracy `0.8904` and AUROC `0.9451`.

The gap is not spread over the six units. Two units hold 95% of it.

| unit | our AUROC | their AUROC | gap | part of total gap |
| --- | ---: | ---: | ---: | ---: |
| Notus / Nemotron-120B | 0.6900 | 0.9056 | 0.2156 | **52.3%** |
| Notus / Qwen3.5-27B | 0.7572 | 0.9326 | 0.1754 | **42.5%** |
| Notus / gemma-3-27b-it | 0.8259 | 0.8715 | 0.0456 | 11.1% |
| Iris / Nemotron-120B | 0.9933 | 0.9863 | −0.0070 | −1.7% |
| Iris / Qwen3.5-27B | 0.9995 | 0.9919 | −0.0076 | −1.8% |
| Iris / gemma-3-27b-it | 0.9921 | 0.9825 | −0.0096 | −2.3% |

Read this table in three steps.

1. We are in front on all three Iris units. Iris has no space left.
2. All of the gap is on Notus. Notus uses the base model only. Notus uses no
   LoRA adapter.
3. Notus / Nemotron and Notus / Qwen hold 95% of the gap.

Our Notus balanced accuracy is already near the best value possible at our Notus
AUROC. So there is one target, not two. Make the order of the rows better on
Notus. The balanced accuracy then follows.

The gate cannot do this. Five gate versions have been scored. Every Notus unit
is identical across v2.1, v2.2, v2.3 and v2.3.5 on both metrics. The gate work
only ever moved Iris. Notus is limited by the detectors.

---

## 3. The two detectors

sonic_v2.3.7 uses two detectors. They read different things. This is why the gate
can mix them.

### 3.1 The probe (white-box)

The probe reads the activations inside the base model. It reads decoder layer
46. It reads only the tokens of the final assistant answer. It gives a score
between 0 and 1.

The probe needs weights for the base model of the dataset. A base model with no
weights gets no probe.

The probe is strong in distribution. The probe is weak on new organisms. On
Notus it scores AUROC 0.69 to 0.83.

### 3.2 The judge (black-box)

The judge reads only the text of the conversation. It does not read
activations. It is `Qwen/Qwen3.5-9B` with the LoRA adapter
`Jazhyc/aletheias-phoenix-wright-v2-adapter`.

The judge gives two outputs.

- A hard verdict of 0 or 1. A regular expression reads `Prediction:<0 or 1>`.
- A soft confidence between 0 and 1. A second pass reads the probability of the
  token "1" against the token "0".

The judge is black-box. So it runs on any base model. It runs even when there
are no probe weights. The judge scores about AUROC 0.733 on Notus.

The soft confidence is applied all or nothing. The gate uses it only if it
agrees with the hard verdict on 80% or more of the rows. If not, the gate keeps
the hard verdicts.

---

## 4. The change — the shared-trunk probe

### 4.1 The fault

We train one probe for each family. Each probe learns deception from the rows of
its own family only. This is bad for Nemotron.

| family | dev rows | deceptive rows | organisms | hidden width |
| --- | ---: | ---: | ---: | ---: |
| qwen | 7,200 | 3,600 | 18 | 5,120 |
| gemma | 800 | 400 | 2 | 5,376 |
| **nemotron** | **216** | **16** | **1** | **4,096** |

Nemotron has 216 rows and 16 deceptive rows. This is the whole Nemotron dev
corpus. We checked the Hugging Face API. There is exactly one Nemotron dev
dataset. We cannot get more.

Notus / Nemotron is 52% of the gap.

### 4.2 The idea

Look at the probe architecture. Only one part depends on the family. That part
is the input projection `Linear(hidden_dim, 128)`. It maps a residual stream of
4,096, 5,120 or 5,376 down to 128.

Everything after the projection lives at width 128:

- the sinusoidal position encoding,
- the transformer encoder (2 blocks, 4 heads, feed-forward width 256),
- the masked mean pooling,
- the linear head.

None of these parts depend on the family.

So keep one projection for each family. Share everything after it. Call the
shared part "the trunk".

The trunk now learns from all 8,216 rows of all three families. The Nemotron
projection still fits on the 216 Nemotron rows. But the 216 rows no longer have
to teach the trunk what deception is.

### 4.3 Training rules

- Batches never mix families. The projections have different input widths.
- Batch order is interleaved across families. The trunk then sees the families
  evenly inside one epoch.
- Standardize the tokens with the moments of each family. Do not share moments.
- Weight the loss by the inverse class frequency. Nemotron is 7.4% positive.
  Without the weight the head learns the majority class for that family.
- Select the best epoch on the deception loss of a held-out validation split.

### 4.4 The evidence

Every number below is leakage-free. The probe is refit for each fold on the
training rows only.

The noise floor is `0.002`. We refit the same probe with the same seed twice and
the worst-organism AUROC moved `0.8343` to `0.8345`. Ignore smaller changes.

**Nemotron, 5-fold cross-validation over the rows.** 216 rows and 16 positives
are too few for one holdout. So each row is scored one time out of fold.

| seed | one probe for each family | shared trunk | change |
| ---: | ---: | ---: | ---: |
| 0 | 0.9063 | 0.9791 | +0.0728 |
| 1 | 0.9641 | 0.9800 | +0.0159 |
| 2 | 0.9563 | 0.9816 | +0.0253 |
| **mean** | **0.9422** | **0.9802** | **+0.0380** |

The shared trunk wins on every seed.

**The variance result is stronger than the mean result.** The single-family
probe ranges `0.9063` to `0.9641`, a spread of `0.058`. The shared trunk ranges
`0.9791` to `0.9816`, a spread of `0.0025`. The shared trunk is about 25 times
more stable. You ship one draw of the weights. A probe that can land anywhere in
a `0.058` band is a risk in itself.

**Leave one organism out, per family, mean AUROC over that family's folds.**

| family | folds | seed 0 | seed 1 | seed 2 | mean change |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen | 18 | +0.0033 | +0.0006 | +0.0002 | **+0.0014** |
| gemma | 2 | −0.0168 | −0.0158 | −0.0107 | **−0.0144** |

qwen is positive on every seed, but the margin is small. gemma is negative on
every seed, by 7 to 10 times the noise floor.

A single seed is not enough here. On seed 0 alone the shared trunk appeared to
win the worst qwen organism by `+0.0135`. Seeds 1 and 2 reversed that. Treat any
single-seed result on these folds as noise.

### 4.5 What to ship

Use the shared trunk for two families. Keep gemma as it is.

| family | probe to ship | reason |
| --- | --- | --- |
| nemotron | shared trunk | `+0.038` mean, positive on every seed, 25x more stable |
| qwen | shared trunk | small but positive on every seed; 42.5% of the gap |
| gemma | **the current probe** | negative on every seed |

Be clear about the strength of the qwen case. It is an expected-value bet, not a
demonstrated gain. The measured downside is near zero. The upside is unmeasured,
because no local fold reaches the Notus regime (section 10). The trunk's
demonstrated property is variance reduction, and that is what transfer needs.

gemma is declined on evidence. The loss repeats on all three seeds. Notus /
gemma is already our best Notus unit at `0.8259`. That unit is only 11% of the
gap. There is little to win and a measured loss.

### 4.6 Cost

The cost is zero at run time. All activations are already cached. There is no
extra NDIF work. There is no extra forward pass. There is no notebook change.

### 4.7 How the weights are built

The trunk is trained on every dev row, then split into one plain probe for each
family. This is exact. For one family we copy the shared encoder, the shared
head, and that family's projection into a plain `TransformerTokenProbe`. Its
`state_dict` keys then match what cell 5 of the notebook builds.

The export is verified, not assumed. Every family's rows are scored with the
shared model and with the exported model. The run fails unless they agree to
`1e-4`. The measured difference is `0.00e+00`.

```
.venv/bin/python experiments/nonlinear_probe/export_multifamily_probe.py
```

The old weights are copied to `submission/whitebox_probe/*_probe.backup.<stamp>`
before anything is written.

---

## 5. The gate

The gate does not change. This section describes it in full.

The gate runs after both detectors. It reads the probe scores and the judge
outputs. It writes the two output columns.

### 5.1 Which detectors are available

The gate has four branches.

| probe | judge | what the gate does |
| --- | --- | --- |
| yes | yes | blend, see below |
| no | yes | judge alone, ranked |
| yes | no | probe alone, split at the median of the probe scores |
| no | no | write 0.5 on every row |

The probe-alone branch splits at the median. It does not use the threshold in
the config file. The config ships `"threshold": null`. In sonic_v2.3 that null
collapsed one unit from `0.9700` to `0.5000`. The median split removes that
fault.

### 5.2 The agreement

The gate has no labels. So it uses the judge as a substitute. It measures:

```
agreement = AUROC(judge hard verdicts, probe scores)
```

High agreement means the two detectors rank the rows in the same order. They
read different things. So agreement is real evidence.

### 5.3 The inversion guard

A probe can rank the rows backwards on new data. sonic_v1 scored AUROC `0.3526`
on Notus / gemma. That probe was inverted, not empty.

```
if agreement < 0.5 - 0.05:
    agreement = 1.0 - agreement      # backwards: flip the probe
    probe = 1.0 - probe
elif agreement < 0.5:
    agreement = 0.5                  # near chance: drop the probe
```

The margin of `0.05` stops the gate from flipping noise. An agreement of `0.47`
is almost always noise around chance.

### 5.4 The trust weight

```
w = clip(2 * agreement - 1, 0, 0.9)
```

This maps an agreement of `[0.5, 1.0]` to a weight of `[0, 1]`, then caps it at
`0.9`. An agreement of `0.5` gives `w = 0`, and the gate uses the judge alone.
An agreement of `0.8` gives `w = 0.6`.

### 5.5 The blend

Change both detectors to ranks between 0 and 1. Ties take the mean rank.

```
score = w * rank(probe) + (1 - w) * rank(judge soft)
```

The rank step removes the scale of each detector. A flat detector becomes flat
ranks. So it cannot damage the blend.

### 5.6 The threshold

```
rate = clip(mean(judge hard verdicts), 0.05, 0.95)
threshold = quantile(score, 1 - rate)
deceptive = score >= threshold
```

The judge gives the prevalence. The blend gives the order. Each detector drives
the column it suits.

When the judge returns one class only, `rate` falls back to `0.5`. Section 11
shows what that costs on an imbalanced dataset.

---

### 5.7 The whole flow

```mermaid
flowchart TD
    CONV["Conversation"] --> WEIGHTS{"probe weights<br/>for this base model?"}
    WEIGHTS -- "yes" --> PROBE["Probe (white-box)<br/>layer 46, score 0..1"]
    WEIGHTS -- "no" --> NOPROBE["no probe score"]
    CONV --> SPLIT{"validation dataset<br/>OR no probe weights?"}
    SPLIT -- "no" --> NOJUDGE["no judge score"]
    SPLIT -- "yes" --> JUDGE["Judge (black-box)<br/>retry once if budget allows"]
    PROBE --> GATE
    NOPROBE --> GATE
    JUDGE --> GATE
    NOJUDGE --> GATE
    subgraph GATE ["The gate (cell 12)"]
        direction TB
        S1{"which detectors<br/>gave output?"}
        S1 -- "probe only" --> S2["probe alone<br/>split at its median"]
        S1 -- "judge only" --> S2B["judge alone, ranked"]
        S1 -- "neither" --> S2C["write 0.5 on every row"]
        S1 -- "both" --> S3{"verdicts are<br/>one class?"}
        S3 -- "yes" --> S4["w = 0.5<br/>rate = 0.5"]
        S3 -- "no" --> INV{"agreement<br/>vs chance?"}
        INV -- "below 0.45" --> FLIP["flip the probe<br/>a = 1 - a"]
        INV -- "0.45 to 0.5" --> DROP["a = 0.5<br/>drop the probe"]
        INV -- "0.5 or above" --> KEEP["keep the probe"]
        FLIP --> S5
        DROP --> S5
        KEEP --> S5
        S5["w = clip(2a-1, 0, 0.9)<br/>rate = judge positive rate"]
        S4 --> S6["change scores to ranks<br/>mix the ranks"]
        S5 --> S6
        S6 --> S7["threshold = quantile(blend, 1 - rate)"]
    end
    GATE --> OUT["submission.csv<br/>score = blend<br/>deceptive = blend >= threshold"]
```

Two branches differ from the older diagram in
`docs/sonic/sonic_v2_3_CHANGES.md`. The judge now also runs when the base model
has no probe weights, not only on a `validation` dataset. And the inversion
guard now has three outcomes instead of two.

---

## 6. What sonic_v2.3.7 does not change

- The gate. It stays at sonic_v2.3.6, byte for byte.
- The notebook code. Every executable line of `sonic_v2.3.7.ipynb` is identical to
  `sonic_v2.3.6.ipynb`. Only prose changed.
- The judge model, the LoRA adapter and the judge prompt.
- The rank blend and the threshold rule.
- The response-span selection. The probe still reads the final answer only.
- The layer. The probe still reads decoder layer 46.
- The probe architecture. Width 128, 2 blocks, 4 heads, feed-forward 256.
- The budget-aware judge retry and the median-split fallback.
- The gemma probe weights.

No executable line changes, so sonic_v2.3.7 adds no new sandbox surface.

---

## 7. What we tested and rejected

We designed three more changes. All three failed. We record them with numbers so
nobody repeats the work.

### 7.1 Rejected — a wider judge prompt

The idea: the prompt says "predict 1 only when the final answer itself is
false". This tells the judge to answer 0 for every deception that is not a false
claim. We thought this caused our Notus under-flagging.

We wrote a wider prompt. It kept the false-positive guard and the output
contract. We ran both prompts on the same 400 rows, with the same model,
adapter, precision and batching. Only the prompt differed.

| metric | current prompt | wider prompt | change |
| --- | ---: | ---: | ---: |
| positive rate | 0.365 | 0.375 | +0.010 |
| soft AUROC | 0.9439 | 0.9382 | −0.0057 |
| hard balanced accuracy | 0.9140 | 0.9106 | −0.0034 |
| recall | 0.849 | 0.855 | +0.006 |
| false positive rate | 0.021 | 0.034 | +0.013 |

The wider prompt changed only 10 of 400 hard verdicts.

**We rejected it because the mechanism is wrong, not because the loss is large.**
Look at `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1`. That is the dataset
where the judge under-flags worst: recall `0.475` against a true prevalence of
50%. If the narrow clause caused the under-flagging, removing it must raise
recall there.

| `varied/a-mo-qwen3.5-27b-1` | current | wider |
| --- | ---: | ---: |
| recall | 0.475 | **0.475** |
| false positive rate | 0.050 | 0.075 |
| balanced accuracy | 0.7125 | 0.7000 |

Recall did not move at all. The wider prompt bought no recall where recall was
the problem. It only added false positives.

So the judge does not under-flag because the prompt forbids it. It under-flags
because it cannot tell those rows apart. A wider instruction cannot fix that. To
make the judge carry more of Notus you need a better judge.

```
.venv/bin/python experiments/ensemble_gate_eval/run_judge_local.py --prompt v3 ...
.venv/bin/python experiments/ensemble_gate_eval/compare_judge_prompts.py
```

### 7.2 Rejected — the contrastive probe

The idea: run each conversation two times. Pass 1 keeps the system prompt. Pass
2 removes it. Probe the difference of the activations.

The features are real. The mean cosine between the difference and the absolute
activations is `0.0665`. So the two feature sets are almost orthogonal.

But the result is small.

| method | mean AUROC | worst AUROC |
| --- | ---: | ---: |
| absolute probe | 0.9615 | 0.8345 |
| contrastive probe | 0.9570 | 0.8401 |
| both, fused 50/50 | 0.9650 | 0.8507 |

The contrastive probe alone is worse than the absolute probe. The fusion gives
`+0.014` on the worst organism, for a second forward pass and twice the
activation download.

See `experiments/contrastive_probe/EVALUATION.md`.

### 7.3 Rejected — the organism-invariant probe

The idea: add a second head that predicts the organism. Put a gradient reversal
layer in front of it. The encoder then hides the organism.

| method | mean AUROC | worst AUROC |
| --- | ---: | ---: |
| plain probe | 0.9615 | 0.8345 |
| gradient reversal | 0.9609 | 0.8357 |

Both changes are inside the noise floor.

The probable reason: it removes the wrong thing. Gradient reversal hides which
*training* organism a row came from. The dev organisms differ by LoRA adapter.
Notus has no adapter at all. Invariance to the first is not invariance to the
second.

See `experiments/nonlinear_probe/invariant_probe.py`.

---

## 8. Dry run results

Full dry run on 2026-07-24. All 21 dev datasets, all 400 rows, judge forced on.
Runtime 15m30s.

| measure | value |
| --- | ---: |
| mean balanced accuracy | 0.9490 |
| mean AUROC | 0.9868 |
| datasets below AUROC 0.90 | none |
| lowest AUROC | 0.9386 (`varied/a-mo-qwen3.5-27b-1`) |
| lowest balanced accuracy | 0.7700 (Nemotron, see below) |

No dataset collapsed. No fallback fired. The weights load and score correctly.
The gemma probe works beside the two new ones.

**Read these as a health check, not as accuracy.** The shipped probe is trained
on every dev row. So every number here is in-sample and too high. The
leakage-free estimates are in section 4.4.

**The one low unit is a threshold artefact.** Nemotron dev scored balanced
accuracy `0.7700` at AUROC `0.9872`. The order of the rows is near-perfect. The
judge returned one class only, so `rate` fell back to `0.5` against a true
prevalence of `0.074`. The gate then flagged about half the rows: recall
`1.0000`, false positive rate `0.4600`. This behaviour is identical in
sonic_v2.3.5. sonic_v2.3.7 does not touch that path.

```
SONIC_FORCE_JUDGE=1 python submit.py --dry
```

---

## 9. How to build and check sonic_v2.3.7

1. Train and export the weights.

   ```
   .venv/bin/python experiments/nonlinear_probe/export_multifamily_probe.py
   ```

   This fits the trunk on every dev row, exports nemotron and qwen, verifies
   parity, and backs up the old weights. It does not touch gemma.

2. Copy `sonic_v2.3.6.ipynb` to `submission/sonic_v2.3.7.ipynb`. Change no code in
   it. Update only the title and the comments. Keep one notebook in
   `submission/`; the dry runner needs exactly one.

3. Run the dry pipeline with the judge forced on.

   ```
   SONIC_FORCE_JUDGE=1 python submit.py --dry
   ```

   **Do not add `--limit`.** The dev datasets are ordered honest-first. The
   first 80 rows of a 400-row, 50%-deceptive dataset are 0/80 deceptive. A
   limited run scores one class only. Balanced accuracy and AUROC are then
   undefined, and the tool prints `—` instead of an error. A limited run is a
   smoke test. It cannot show whether accuracy moved. See
   `docs/sonic/dev_data_row_ordering.md`.

4. Compare against section 8. No dataset may collapse.

---

## 10. The limits

Read this section before you trust any number above.

**The offline proxy is easier than the real task.** Our hardest local fold
scores `0.8345`. Our best real Notus unit scores `0.8259`. Every local fold is
easier than every Notus unit. So no offline result can prove a Notus gain.

**The Nemotron gain is in-distribution.** The `+0.038` comes from
cross-validation over rows. It is not a held-out organism. Nemotron has one
organism, so leave-one-organism-out is not defined for it. The number shows that
the shared trunk fixes small-data overfit. It does not show transfer to Notus.

**The qwen case is a bet.** The measured gain is `+0.0014`, close to the noise
floor. We ship it because the downside is measured and near zero, and because
Notus / Qwen is 42.5% of the gap.

**A mix of two detectors stays between the two detectors.** The gate cannot make
signal that neither detector has.

**Expected leaderboard effect: small.** sonic_v2.3.7 addresses the largest single
cell and takes a low-conviction bet on the second. It does not close a `0.138`
AUROC gap on its own.

---

## 11. Known open item — the threshold on imbalanced data

The threshold uses the positive rate of the judge, clipped to `[0.05, 0.95]`.
Two units show that this is fragile.

- Metis / Nemotron scored balanced accuracy `0.7525` at AUROC `1.0000`, with a
  false positive rate of `0.4950`. It over-flags.
- Nemotron dev scored balanced accuracy `0.7700` at AUROC `0.9872`, with a false
  positive rate of `0.4600`. The judge returned one class, so `rate` fell back
  to `0.5` against a true prevalence of `0.074`.

**A warning about a fix we proposed and then disproved.** We first proposed
narrowing the clip to `[0.35, 0.65]`. The argument was that every dataset is 50%
deceptive. **That argument is false.** The Nemotron dev dataset is 216 rows at
7.4% deceptive, confirmed against the Hugging Face API. On a dataset of that
shape, a `[0.35, 0.65]` clip forces at least 35% of the rows to be flagged and
makes balanced accuracy much worse.

Do not ship that clip. A correct fix must handle a genuinely imbalanced dataset,
and the fixed band cannot. This item is open.

---

## 12. Rollback

The last submitted version is `legacy_submissions/sonic_v2.3.5.ipynb`. It scored
balanced accuracy `0.853750` and AUROC `0.876333`.

To roll back the weights only, copy the backup directories back:

```
cp submission/whitebox_probe/nemotron_probe.backup.<stamp>/* \
   submission/whitebox_probe/nemotron_probe/
cp submission/whitebox_probe/qwen_probe.backup.<stamp>/* \
   submission/whitebox_probe/qwen_probe/
```

The notebook needs no rollback. It is identical to sonic_v2.3.6.

---

## 13. Where the evidence is

| subject | file |
| --- | --- |
| Leaderboard record, all units | `docs/official_submissions.md` |
| The v2.3.6 gate, step by step | `docs/sonic/sonic_v2_3_6_EXPLAINER.md` |
| Honest-first row ordering | `docs/sonic/dev_data_row_ordering.md` |
| Shared-trunk results and seed sweep | `experiments/nonlinear_probe/MULTIFAMILY_EVALUATION.md` |
| Contrastive and invariant results | `experiments/contrastive_probe/EVALUATION.md` |
| Shared-trunk model | `experiments/nonlinear_probe/multifamily_probe.py` |
| Shared-trunk evaluation | `experiments/nonlinear_probe/run_multifamily_eval.py` |
| Weight export and parity check | `experiments/nonlinear_probe/export_multifamily_probe.py` |
| Judge prompt A/B | `experiments/ensemble_gate_eval/compare_judge_prompts.py` |
| Gate as pure functions | `experiments/ensemble_gate_eval/gate.py` |
| Full dry run log | `logs/sonic_v2_3_7_dry_full.log` |
