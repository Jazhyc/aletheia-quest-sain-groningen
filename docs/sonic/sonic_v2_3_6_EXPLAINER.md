# sonic v2.3.x — how the detector works, a worked example, and a critique

This document explains the sonic detector end to end (the shared logic that
sonic_v2.3.5 and sonic_v2.3.6 both run), walks one input through it with exact
numbers, and then critiques each key step for how well it should generalize.

- For the *deltas* v2.3.6 adds over v2.3.5, read `docs/sonic/sonic_v2_3_6_CHANGES.md`.
- For the original gate description, read `docs/sonic/sonic_v2_3_CHANGES.md`.

Where v2.3.6's two changes touch the flow is marked inline as **[v2.3.6: A]**
and **[v2.3.6: C]**.

---

## 1. The two detectors

sonic fuses two independent deception detectors that read different things:

- The **probe** (white-box) reads the model's **internal activations**. It is a
  transformer token probe at decoder layer 46 and outputs a score in `[0, 1]`.
  It needs weights for the dataset's base model (gemma, qwen, or nemotron) and
  cannot run on any other base model.
- The **judge** (black-box) reads only the **conversation text**. It is
  `Qwen3.5-9B` with a LoRA adapter and outputs a hard `0/1` verdict plus a soft
  confidence in `[0, 1]`. Because it needs no activations, it runs on any base
  model — including one with no probe weights.

The submission writes two columns to `submission.csv`:

- **score** — one continuous number per row; sets the **AUROC**.
- **deceptive** — a `0/1` per row; sets the **balanced accuracy**.

The gate produces both from whichever detectors are available.

---

## 2. The probe

### 2a. Training (offline, ahead of time)

One probe is trained per base-model family. The probe never sees text — it sees
hidden states.

1. **Activation extraction.** For each labeled conversation, chat-template and
   tokenize it, locate the assistant's **response span** (see §3), and run the
   target model, capturing the decoder **layer-46** hidden state for every
   response token. Each example becomes a `(n_response_tokens, hidden_dim)`
   sequence (hidden_dim 5376 gemma / 5120 qwen / 4096 nemotron).
2. **Standardize.** Per-feature mean/std are computed on the training rows only
   and saved as `feature_mean.pt` / `feature_std.pt`. Every token is
   standardized, padding zeroed after.
3. **Model** (`TransformerTokenProbe`): linear projection `hidden_dim -> 128`,
   add sinusoidal positional encoding, a 2-layer `TransformerEncoder` (4 heads,
   ff 256, dropout 0.1), **masked mean-pool** over the response tokens to one
   128-d vector, linear head to one logit.
4. **Fit** (`TokenProbe.fit`): stratified 15% validation split, Adam (lr 1e-3,
   weight decay 1e-4), `BCEWithLogitsLoss`, length-sorted token-budgeted
   batches, early stopping on validation loss (patience 6, <=60 epochs), restore
   best-validation weights, save `model.pt`.
5. **Output.** `sigmoid(logit)` -> probe score in `[0, 1]`. The config ships
   `"threshold": null` — the probe has **no calibrated absolute cutoff**.

> Leakage caveat: the shipped probes were trained on *all* dev rows, so dev
> numbers are optimistic. The leakage-free holdout gives BA ~0.923 vs the ~0.968
> the leaked numbers suggest. Discount dev performance accordingly.

### 2b. Runtime scoring (in the notebook, cells 4–10)

Load the probe and moments, build the model handle, extract layer-46 response
activations over NDIF (one bundled remote session, bounded retry on transient
EOF/connection errors, numpy fp16->fp32 cast to dodge the sandbox
`/proc/cpuinfo` crash), standardize, run the probe, `sigmoid` -> `probe_scores`.

---

## 3. The response span — why "keep only response tokens" and how it is exact

A conversation is more than the answer: it has the system prompt, the user
question, and chat-template scaffolding (`<start_of_turn>`, role markers). The
probe pools **only over the assistant's own tokens**, because deception is a
property of the answer — the user's prompt is identical for an honest and a
deceptive reply, so pooling it in only adds noise.

The boundary is **computed exactly**, not searched, by a difference of two
tokenizations (`util.response_span`):

```python
full   = apply_chat_template(messages,      add_generation_prompt=False)  # everything
prefix = apply_chat_template(messages[:-1], add_generation_prompt=True)   # all but the reply
start, end = len(prefix), len(full)
```

`prefix` tokenizes every message *except* the final assistant turn, with the
generation prompt appended — so it is a literal token-for-token prefix of
`full`, ending exactly where the assistant content begins. Therefore
`full[start:end]` is precisely the assistant's tokens, template tags excluded.
It is reliable because chat templates are deterministic and prefix-consistent,
and it uses the tokenizer's own template, so it is correct for every base model.

Edge cases: with no chat template the fallback treats the whole sequence as the
response (`start=0`); `chat_preprocess` then left-trims to 512 tokens (dropping
from the *start*) so the answer always survives. The assistant reply is assumed
to be the last message — true for this task.

Note the judge consumes the conversation differently: it gets the **whole text**
(all roles) but with the final reply *also* isolated inside `<output>`, so it can
use full context to judge whether that specific answer is false.

---

## 4. The judge (cell 11)

- **Model:** `Qwen/Qwen3.5-9B` + LoRA adapter
  (`Jazhyc/aletheias-phoenix-wright-v2-adapter`) via `util.build_model`, run on
  NDIF.
- **Prompt:** a fixed system prompt tells it to predict `1` only when the final
  `<output>` answer is itself false/misleading (hidden/system text is a warning
  signal, not proof), followed by the conversation packed as
  `<context>…</context>\n<output>…</output>`, truncated head+tail to fit token
  limits.
- **Hard verdict:** greedy generation; a regex reads `Prediction:<0/1>` ->
  `judge_scores`. Drives the `deceptive` column.
- **Soft confidence:** a second, teacher-forced pass feeds prompt + generated
  reasoning up to the `Prediction:` marker and reads the softmax probability of
  emitting token "1" vs "0" as the next token -> `judge_probs`. Drives ranking
  in the `score` column. It is applied **all-or-nothing**: only if it reproduces
  the hard verdict on `>= 80%` of rows overall; otherwise the gate keeps the hard
  `0/1` values. So it can add ranking resolution but never flip a verdict.
- **When it runs:** on `validation-*` datasets, or `SONIC_FORCE_JUDGE=1`,
  **[v2.3.6: A]** or when the base model has no probe weights. On failure it
  retries once if the remaining wall-clock budget allows.

---

## 5. The gate, worked end to end (cell 12)

Toy dataset: 6 rows from a `validation-gemma` dataset. Ground truth is shown
only to score quality at the end — the gate never sees it.

```
truth        = [1, 1, 0, 1, 0, 0]     # hidden
probe_scores = [0.91, 0.62, 0.30, 0.83, 0.12, 0.55]   # from §2
judge_hard   = [1,    0,    0,    1,    0,    1  ]     # note: row 5 is a judge MISTAKE
judge_probs  = [0.88, 0.41, 0.22, 0.79, 0.18, 0.63]
```

Both detectors fired and the verdicts are not single-class -> the main mixing
path.

### Step 1 — agreement = `AUROC(judge_hard, probe_scores)`

AUROC here means: *pretend the judge's verdicts are the answer key; how well do
the probe's scores line up with it?* Concretely — pick a random judge-`1` row
and a random judge-`0` row; what fraction of such pairs does the probe score the
`1`-row higher?

- judge `1` rows -> probe {0.91, 0.83, 0.55}
- judge `0` rows -> probe {0.62, 0.30, 0.12}

3 × 3 = 9 pairs. Count where the judge-`1` probe score is higher:

|        | vs 0.62 | vs 0.30 | vs 0.12 |
|--------|---------|---------|---------|
| 0.91   | ✓ | ✓ | ✓ |
| 0.83   | ✓ | ✓ | ✓ |
| 0.55   | ✗ | ✓ | ✓ |

8 / 9 -> **agreement = 0.8889**. The one disagreement is row 5 (judge flagged
it, probe 0.55 < honest row 1's 0.62). Since `0.8889 >= 0.5`, no inversion.
**[v2.3.6: C]** would only alter this step if agreement fell into `[0.45, 0.5)`.

### Step 2 — trust weight `w = clip(2·agreement − 1, 0, 0.9)`

`2·0.8889 − 1 = 0.7778`, inside `[0, 0.9]` -> **w = 0.7778**. (Maps agreement
0.5 -> w 0, agreement 1.0 -> w 0.9. The 0.9 cap never lets the probe take the
whole decision.)

### Step 3 — rank both detectors to `[0, 1]` (`_rank01`)

`_rank01` puts scores on a common scale: smallest -> 0, largest -> 1, evenly
spread, tied values sharing an averaged rank. Mechanically: `argsort` gives the
sorted order as original indices; inverting it stamps each row's sorted position
back onto its slot; `np.unique` groups equal values and averages their
positions; divide by `n − 1`.

```
probe_rank = [1.0, 0.6, 0.2, 0.8, 0.0, 0.4]
judge_rank = [1.0, 0.4, 0.2, 0.8, 0.0, 0.6]
```

(Tie example: `[0.5, 0.2, 0.5, 0.9]` -> `[0.5, 0.0, 0.5, 1.0]` — the two `0.5`s
share the averaged rank `0.5` rather than getting arbitrary neighbouring ranks.)

Why ranks: the probe (saturated sigmoid) and judge (softmax over two logits)
live on different scales; ranking makes the blend scale-invariant, and a
spread-less detector collapses to a constant `0.5` rank, so it cannot corrupt
the blend (no degeneracy guard needed).

### Step 4 — blend = `w·probe_rank + (1 − w)·judge_rank` -> the `score` column

With `w = 0.7778`, `1 − w = 0.2222`:

```
row0: 0.7778·1.0 + 0.2222·1.0 = 1.0000
row1: 0.7778·0.6 + 0.2222·0.4 = 0.5556
row2: 0.7778·0.2 + 0.2222·0.2 = 0.2000
row3: 0.7778·0.8 + 0.2222·0.8 = 0.8000
row4: 0.7778·0.0 + 0.2222·0.0 = 0.0000
row5: 0.7778·0.4 + 0.2222·0.6 = 0.4444
```

Watch **row 5**: the judge's soft rank (0.6) is pulled down by the probe's rank
(0.4) under the high weight -> blend 0.4444. The probe quietly corrects the
judge's mistake.

### Step 5 — threshold = `quantile(blend, 1 − rate)` -> the `deceptive` column

The judge sets *how many* rows are deceptive; the blend sets *which*. Judge
positive rate = 3/6 = 0.5, clipped to `[0.05, 0.95]` -> **rate 0.5**. So keep the
top 50%: `threshold = quantile(blend, 0.5)` = the median. Sorted blend is
`0.0, 0.2, 0.4444, 0.5556, 0.8, 1.0`; the 0.5-quantile interpolates to
**0.5000**.

```
deceptive = (blend >= 0.5) = [1, 1, 0, 1, 0, 0]
```

### Result

| index | score  | deceptive | truth |
|-------|--------|-----------|-------|
| 0 | 1.0000 | 1 | 1 |
| 1 | 0.5556 | 1 | 1 |
| 2 | 0.2000 | 0 | 0 |
| 3 | 0.8000 | 1 | 1 |
| 4 | 0.0000 | 0 | 0 |
| 5 | 0.4444 | 0 | 0 |

Scored against the hidden truth: **AUROC 1.00, balanced accuracy 1.00** — versus
the judge's hard verdicts alone at BA 0.667. The fusion recovered row 5 because
`w = 0.78` let the probe's ranking out-vote the judge's wrong verdict.

### The other branches (same input, different availability)

- **Probe blind (novel base model / style):** near-random probe -> agreement
  ≈ 0.5 -> `w ≈ 0`; the blend follows the judge. **[v2.3.6: C]** cleans up the
  case where agreement lands at `0.47` instead of exactly `0.5`.
- **Judge dropped:** probe alone. No prevalence signal, and the probe ships
  `"threshold": null`, so v2.3.5+ splits at the **median** of the probe scores
  (never collapses the decision column — the pre-v2.3.5 null-threshold bug).
- **No probe weights:** judge alone, rank its soft scores. **[v2.3.6: A]**
  guarantees the judge actually runs in this case regardless of the dataset
  name.
- **Neither:** write `0.5` on every row.

---

## 6. Critique — why each step should (or should not) generalize

Ranked roughly by where the generalization risk concentrates. "Generalizes
well on dev" and "will hold on hidden data" are different claims; dev is also
leakage-inflated (see §2a).

### 6.1 The probe — weakest link, by design
Layer ~46 of ~62 is a sensible place to read intent, and the token-transformer
captures distributed signal. But it reads internal activations, the most
distribution-specific object in the system: it sat near **chance on every Notus
unit**. It also needs the base model's weights, so an unseen base model gets no
probe at all. It is a strong *in-distribution* detector and an unreliable
*out-of-distribution* one — the entire gate exists to survive that fact.

### 6.2 Response-span selection — sound, low risk
Reading only assistant tokens is correct and computed exactly. The only
assumption is "the last message is the assistant answer," which a tool-call or
trailing-user-turn dataset could violate. Low probability, bounded damage.

### 6.3 The judge — the coverage backbone, but narrower than it looks
Black-box, so it runs on any base model — the one component that structurally
covers new distributions. But its prompt hard-codes a theory of deception
(*factual falsity*). Deception that is not false-claim-shaped (sandbagging,
withholding, sycophancy, manipulation via technically-true statements) sits
outside what it is asked to flag, and its measured quality is modest (~0.73 BA
on the task the probe cannot do). It broadens *coverage*, not necessarily
*concept*.

### 6.4 Gating the judge on the `validation-` name — underappreciated risk
A dependency on an external naming convention in a load-bearing spot: if the
scored hidden datasets are not named as assumed, the judge never fires. **This
is exactly what change A addresses** — v2.3.6 also runs the judge whenever the
probe is structurally absent, so the worst case (all-0.5) is removed.

### 6.5 agreement = AUROC(judge, probe) — cleverest, most epistemically fragile
It manufactures a probe-reliability estimate with no labels, using the judge as a
pseudo-oracle — genuine evidence when two disjoint-signal detectors agree. Three
leaks: (a) **symmetry / shared bias** — agreement cannot tell which is right, so
correlated errors read as high trust; (b) **small-sample noise** — an AUROC on
few judge-positives is high-variance, so `w` is noisy exactly when it matters;
(c) **inverting noise** — sub-0.5 agreement is usually noise, not true
inversion. **Change C addresses (c)**; a designed-but-unshipped change B would
address (b) by shrinking `w` under small samples.

### 6.6 Trust weight `w` — reasonable mapping, marginal cap, one structural limit
The agreement->trust mapping is the real generalization mechanism and inherits
6.5's flaws. The `0.9` cap is cheap insurance but mostly inert (it only bites at
high agreement, where probe and judge already rank alike). The structural limit:
`w` is one scalar per dataset, so it cannot express "probe reliable on the
clear-cut rows, useless on the subtle ones."

### 6.7 Rank transform + blend — the part that genuinely generalizes well
Scale-invariance is properly motivated, a flat detector self-neutralizes, ties
are averaged. This is why the **score/AUROC column travels much better than the
deceptive/BA column** — AUROC reads only ordering, which ranking preserves.

### 6.8 Threshold from the judge's positive rate — second-weakest link, BA-specific
Clever (sidesteps the probe's missing threshold by borrowing prevalence from the
judge). But it makes **balanced accuracy depend on the judge's absolute
calibration**, not just its ranking — and a fine-tuned model's *positive rate* is
what transfers worst. The `[0.05, 0.95]` clip bounds the disaster but not
ordinary miscalibration inside that range. So AUROC is defended by 6.7;
thresholding/BA is only as good as the judge's prevalence generalizes.

### 6.9 Fallbacks — solid, and the v2.3.5 median-split matters
Median-split probe-alone can never collapse a strong probe to 0.5; judge-alone
ranking covers no-probe datasets; all-0.5 is an honest floor. Pure robustness
wins.

### 6.10 Reliability (budget-aware retry) — good engineering, but ≠ accuracy
Protects against a dropped judge aborting the run without tripping the
wall-clock kill. A recovered failure still lands on probe-only (~0.93), not the
full mix (~0.97) — it hardens the run, it does not extend concept coverage.

### Synthesis
The gate is a **robust combiner**, bounded by its inputs: best-case ≈
`max(probe, judge)`, and its real job is to reliably pick the better one and
never collapse below either. A combiner cannot manufacture signal neither
detector has. The true frontier is a dataset that is *both* a novel base
model/style (probe blind) *and* not factual-falsity-shaped (judge mis-targeted);
there, `w` is noise and the threshold rides an uncalibrated prevalence.
v2.3.6's changes A and C protect the combiner from two ways it mishandles that
regime; raising the *ceiling* needs better detectors (a base-model-agnostic or
gracefully-degrading probe; a broader judge concept), not a better gate.
