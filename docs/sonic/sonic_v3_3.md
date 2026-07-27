# sonic v3.3

**Built 2026-07-26. Submitted 2026-07-27 00:20. Official AUROC `0.9017` /
BA `0.8275`, runtime 24m38s.** Just below v3.2 (`0.9031` / `0.8312`):
`-0.0014` AUROC, `-0.0037` BA. The loss is all on Iris. MAX_CAP is the likely
cause, not the probe. See "Official result". An earlier attempt on 2026-07-26
19:30 failed in the sandbox and scored nothing.

The gate is the v3.2 agreement-modulated cap. The formula is the same. The
threshold is the same. The judge is the same. Three things changed from v3.2:

| | v3.2 | v3.3 |
| --- | ---: | ---: |
| the probe | per-family weights, gemma on its own probe | shared trunk, balanced recipe, all three families together |
| BASE_CAP | 0.208 (2 bf16 steps) | 0.104 (1 step) |
| MAX_CAP | 0.417 (4 steps) | 0.625 (6 steps) |

The values AGREEMENT_SCALE, PROBE_GAIN, and the threshold stay at 3.0, 1.0, and
0.20. The judge is Phoenix v4 direct-logit. The gate formula is byte-identical
to v3.2.

## Design

The judge emits a standardized margin. The probe emits a standardized logit.

```
STEP            = 0.10418792638565248   # one bf16 step of the judge logit
BASE_CAP        = 1 × STEP              # floor: the probe gets at most 1 step
MAX_CAP         = 6 × STEP              # ceiling: the probe gets up to 6 steps
AGREEMENT_SCALE = 3.0

agreement  = clip(judge_z × probe_z / AGREEMENT_SCALE, 0, 1)
cap        = BASE_CAP + agreement × (MAX_CAP − BASE_CAP)
score      = sigmoid(judge_z + cap × tanh(probe_z))
```

The score is the output. The threshold `0.20` converts the score to a binary
label.

**BASE_CAP and MAX_CAP.** The cap is the coefficient on `tanh(probe_z)`. It
controls how much the probe can move the judge's ranking. The unit is bf16 steps
of the judge token logit. One step can break a tie. Six steps can reorder rows
up to six quanta apart.

BASE_CAP is the disagreement floor. When the probe and the judge point opposite
ways, the probe gets at most 1 step. Its signal on those rows is noise. The
probe scored AUROC 0.5586 on Notus. It scored below chance on the gemma unit
(0.4808).

MAX_CAP is the agreement ceiling. When the probe and the judge point the same
way, the probe gets up to 6 steps. Its signal on those rows is strong. The probe
scored AUROC 0.9918 on Iris. The v3.2 offline sweep measured a worst-case Notus
loss of −0.0013 at 6 steps. That is inside the judge's run-to-run noise.

**AGREEMENT_SCALE.** `judge_z` and `probe_z` are standardized. Their product
usually falls in ±1. Without the divisor the product would saturate `clip(…,0,1)`
to 0 or 1 for nearly every row. The agreement modulation would become a hard
binary switch. A weak-agreement row and a strong-agreement row would get the
same cap. Dividing by 3 spreads the product across [0,1]. Weak agreement gives
a partial cap. Strong agreement gives the full cap. The value 3.0 is the smallest
divisor that keeps most rows unsaturated. It is fitted on dev data, not tuned
per submission.

## Rationale

**Lower floor.** The probe does not transfer to Notus. Its AUROC there, measured
by v3.3-mini, is 0.5586. On gemma it is 0.4808 — below chance. Any probe
influence on disagreement rows is random. The floor cap should be as low as
possible. One step is the minimum. It keeps the ability to break ties within the
judge's quantization.

**Higher ceiling.** The probe transfers cleanly to Iris. Its AUROC there is
0.9918 across all three families. The v3.2 offline sweep measured a Notus penalty
of −0.0013 at 6 steps. Six steps is the conservative option. Eight steps is the
fallback.

> This part did not hold. Iris lost `-0.0034`. Eight steps is not the fallback
> any more. See "Official result".

**The probe.** The v3.3-mini submission ran the shared trunk on the private
split. It served all three families. Gemma was included despite local evidence
that the trunk was worse. The counted rows did not single gemma out. The balanced
recipe was sign-consistent across all four dev cells. Swapping these weights
into the blended scoring rule carries low risk.

## Relationship to v3.3-mini

`sonic_v3_3_mini` was a probe-only diagnostic. The judge was removed. The score
was `sigmoid(probe_z)` at threshold 0.5. It isolated the probe's per-unit
ranking on the private split. Its headline AUROC was 0.7752. Its Notus AUROC was
0.5586. Its Iris AUROC was 0.9918.

`sonic_v3_3` re-enables the judge and the v3.2 gate. It uses the same probe
weights that the mini validated. The calibrated threshold and the agreement gate
should absorb the BA artefacts the mini exposed. Iris/gemma had BA 0.5000 at
AUROC 0.9856. That was a threshold problem, not a ranking problem.

## Expected behaviour, and what happened

| regime | Notus AUROC | Notus BA | Iris AUROC | Iris BA | headline AUROC | headline BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the probe alone (v3.3-mini) | 0.5586 | 0.5383 | 0.9918 | 0.7767 | 0.7752 | 0.6575 |
| the judge alone (Phoenix v4) | 0.8640 | 0.7875 | 0.9236 | 0.8433 | 0.8938 | 0.8154 |
| v3.2 (old probe, 2 / 4 steps) | 0.8634 | 0.7942 | 0.9427 | 0.8683 | 0.9031 | 0.8312 |
| v3.3 projected | ≥ 0.8640 | ≥ 0.790 | ≥ 0.950 | ≥ 0.875 | ≥ 0.907 | ≥ 0.833 |
| **v3.3 official** | **0.8642** | **0.7900** | **0.9393** | **0.8650** | **0.9017** | **0.8275** |

All numbers are means over the three counted units per dataset (Nemotron, Qwen,
Gemma). AUROC is the leaderboard metric. BA is the balanced accuracy at the
frozen threshold 0.20.

The Notus projection was right. The Iris projection was wrong in sign.

## Official result

Submitted 2026-07-27 00:20. AUROC `0.9017` / BA `0.8275`, runtime 24m38s.
Against v3.2: `-0.0014` AUROC, `-0.0037` BA.

| dataset | model / lora | v3.3 AUROC | v3.2 AUROC | Δ | v3.3 BA | v3.2 BA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Notus | `Nemotron-120B` / `None` | 0.8946 | 0.8913 | +0.0033 | 0.8325 | 0.8200 |
| Notus | `Qwen3.5-27B` / `None` | 0.8446 | 0.8494 | −0.0048 | 0.7775 | 0.7925 |
| Notus | `gemma-3-27b-it` / `None` | 0.8533 | 0.8494 | +0.0039 | 0.7600 | 0.7700 |
| Iris | `Nemotron-120B` / `g-st` | 0.9541 | 0.9578 | −0.0037 | 0.8850 | 0.8925 |
| Iris | `Qwen3.5-27B` / `g-st` | 0.9491 | 0.9556 | −0.0065 | 0.8950 | 0.8975 |
| Iris | `gemma-3-27b-it` / `g-st-2` | 0.9146 | 0.9148 | −0.0002 | 0.8150 | 0.8150 |
| **Notus mean** | | **0.8642** | 0.8634 | **+0.0008** | 0.7900 | 0.7942 |
| **Iris mean** | | 0.9393 | **0.9427** | **−0.0034** | 0.8650 | 0.8683 |
| **headline** | | 0.9017 | **0.9031** | **−0.0014** | 0.8275 | 0.8312 |

**The lower floor worked.** Notus `0.8642` is the best Notus mean of any sonic
run. It is the first one above the pure judge (`0.8640`). BASE_CAP at 1 step
does what it was designed to do. The probe is noise on Notus. It now barely
touches it.

**The loss is all Iris.** Iris/Nemotron `-0.0037` and Iris/Qwen `-0.0065`.
Iris/gemma does not move (`0.9146` vs `0.9148`). That unit did not move between
v3.1 and v3.2 either. Judge and probe never agree strongly enough there for the
product to clear AGREEMENT_SCALE, so the cap stays near its floor.

**Probably MAX_CAP. Not proven.** The probe and both caps changed in one run.
Nothing here isolates either one. Two facts point at the cap. Both are indirect.

- The uncounted references are flat. Metis `0.9880` against v3.2's `0.9894`.
  Eunomia `0.8918` against `0.8915`. These are the probe's strong regimes. A
  degraded probe would show there first. This is the better argument. The cap
  also changed on those rows.
- The probe ranks Iris well alone. v3.3-mini measured these weights at Iris
  AUROC `0.9918`. But standalone AUROC does not settle blend behaviour. What
  matters in the gate is whether the probe's confident errors line up with the
  judge's. A probe can rank better alone and blend worse.

The unit pattern is **not** evidence. Where the cap is shut the probe barely
enters the score, so a probe regression predicts the same shape: flat on gemma,
loss on the other two. The pattern cannot tell the two causes apart.

One thing holds either way. The v3.2 sweep priced 6 steps at a worst-fold Notus
penalty of `-0.0013` and called it noise. It only asked whether a bigger cap
hurt Notus. The probe is in-sample on every dev fold, so probe-judge agreement
is inflated there and the Iris side was never priced. We raised the ceiling with
no Iris estimate behind it.

**Next.** `sonic_v3_4` is built for this. It puts both caps back to v3.2 and
keeps the new probe. One thing moves, so the result is readable. If Iris
recovers, MAX_CAP caused this and 6 steps is dead. If Iris does not recover, the
probe caused it and 6 steps is still untested. BASE_CAP at 1 step is not carried
into v3.4 even though Notus liked it. Holding it would move two things again.
See `sonic_v3_4.md`.
