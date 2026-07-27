# sonic v3.6 — the cap raised to 12 judge steps

**Built 2026-07-27, submitted 2026-07-27. Scored AUROC `0.9068` / BA `0.7975` — peak AUROC to date. Moved to `legacy_submissions/`.**

This document uses Simplified Technical English. Sentences are short. Each
sentence gives one idea. The terms are constant: "the probe", "the judge",
"the gate", "the cap", "the step", "the edge", "the score", "the threshold",
and "the organism".

## 1. The change

One constant moves. The cap ceiling goes from 4 steps to 12 steps.

```
MAX_CAP = 12 steps = 1.2502551166278297   # v3.6
MAX_CAP =  4 steps = 0.41675170554260993  # v3.1 to v3.5

agreement = (judge_z × probe_z > 0)
cap   = BASE_CAP + agreement × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × tanh(PROBE_GAIN × probe_z))
```

`BASE_CAP` stays at 2 steps. `PROBE_GAIN` stays at 1.0. The gate stays a sign
test. The probe, the judge, the prompt and both thresholds are v3.5's.

Cells 1 to 11 are byte-identical to `sonic_v3_5.ipynb`. The scoring cell differs
in two executable lines. One of those two lines is a log string.

## 2. Why the cap was the limit on Iris

The cap bounds the probe in absolute judge-z units. The bound is a fixed number.

The edge is not a fixed number. The edge is how much the probe beats the judge.
The edge grows when the probe gets better.

So the share of the edge that the cap admits falls as the edge grows. A fit run
on a small edge chooses a cap that is too tight for a large edge.

This table gives the capture. Capture is the share of the edge that reaches the
score. The gate is held fixed. Only the cap moves.

| edge | cap2 | **cap4** | cap6 | cap8 | **cap12** | cap20 | cap40 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 82% | **89%** | 93% | 96% | **100%** | 104% | 105% |
| 0.045 | 73% | **80%** | 84% | 87% | **90%** | 93% | 94% |
| **0.068** | 61% | **69%** | 73% | 76% | **80%** | 84% | 86% |
| 0.090 | 56% | **64%** | 69% | 73% | **78%** | 82% | 83% |

Real Iris sits in the 0.068 row. `sonic_v3_3_mini` measured the probe at AUROC
`0.9918` on Iris. The judge is at `0.9236` on Iris. The edge is `+0.068`.

The judge figure is derived. It is derived two ways. The first way is the
`+0.068` edge in `sonic_v3_5.md`. The second way is the v3.2 decomposition in
`official_submissions.md`, which gives Iris `+0.0191`, Notus `−0.0006` and
headline `+0.0092` against `phoenix_wright_v4`. The two ways agree to `0.0002`.

At 4 steps the cap binds on Iris. At 12 steps it binds less.

## 3. Why the raised cap is safe on Notus

The safety comes from the direction test. It does not come from the cap size.
This is the core claim of this version.

A probe that is noise opens the cap on about half the rows. It opens them at
random. A bounded contribution in random directions cannot reorder rows in a
systematic way. A larger bound does not make random reordering worse.

This table gives the score on Notus. The probe is blunted. The judge is healthy.

| probe quality | cap2 | cap4 | cap8 | cap12 | cap20 | cap40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5586 | 0.9246 | 0.9243 | 0.9242 | 0.9241 | 0.9241 | 0.9241 |
| 0.48 inverted | 0.9132 | 0.9128 | 0.9126 | 0.9125 | 0.9125 | 0.9125 |

The cap moves by a factor of 20. The cost is `−0.0003`.

`0.5586` is the probe's real Notus quality. `sonic_v3_3_mini` measured it.

## 4. The net result

This table gives the change against the shipped cap of 4 steps. Iris is at the
real edge. Notus is at the measured `0.5586`.

| cap | Iris | Notus | headline |
| ---: | ---: | ---: | ---: |
| 6 | +0.0036 | −0.0001 | +0.0018 |
| 8 | +0.0062 | −0.0002 | +0.0030 |
| **12** | **+0.0090** | **−0.0002** | **+0.0044** |
| 20 | +0.0112 | −0.0002 | +0.0055 |
| 40 | +0.0121 | −0.0003 | +0.0059 |

## 5. Why 12 steps and not 20 or 40

12 steps takes 80% of the edge. 40 steps takes 86%. The gain from 20 to 40 is
`+0.0004`.

4 steps is the only cap that any private run has tested. 12 steps is a smaller
move away from it than 20 or 40. The extra capture is not worth the extra
distance.

## 6. Why the earlier fits missed this

Three fits looked at the cap. All three missed it. Each missed it for its own
reason.

**`fit_bounded_refine_v3_2.py`.** It swept 4, 6 and 8 steps. It scored safety
against a probe blunted to AUROC `0.76`. The real Notus figure is `0.5586`. It
scored gain on folds where the edge was `+0.0276`. The real Iris edge is
`+0.068`. Both errors push toward a tighter cap.

**`fit_sign_gate_v3_5.py`.** It varied the shape of the gate. It held the cap at
2 and 4 steps. It never raised the cap. Its "flat, always open" row is a
different change: that row removes the direction test at 4 steps. That is why
that row costs `−0.0137` on Notus. Keeping the direction test and raising the
ceiling was never tested.

**`test_saturation_v3_6.py`.** It swept the cap and found `−0.0001`. It built
its Iris regime by sharpening the probe. Sharpening saturates near `+0.025`. The
cap does not bind at `+0.025`. That arm was uninformative. It was not negative.

`test_cap_binding_v3_6.py` fixes the last one. It builds the edge by blunting the
judge. Blunting does not saturate. So it reaches the real edge.

## 7. Risks

**The Notus proxy is easier than Notus.** The local folds have the judge at
`0.938`. Real Notus has the judge at about `0.864`. A strong judge resists a
bounded nudge. A weak judge resists it less. So the near-zero cost in section 3
is measured on the wrong difficulty. This is the standing blind spot: no dev cell
reproduces the Notus failure. It is the least trustworthy number here.

**Part of the Iris shortfall is not explained.** The harness says 4 steps should
capture 69% on real Iris. `sonic_v3_5` captured `30.5%`. That is `0.9444` against
a judge at `0.9236` and a probe at `0.9918`. Raising the cap addresses the part
of the gap that the model explains. It does not address the rest. So read the
`+0.0044` as an upper bound.

**The probe is in-sample on every dev fold.** So the absolute levels in sections
2 and 3 are too high. Only the comparison across cap values is used.

## 8. Projection

If the gain lands in full: Iris about `0.953`, Notus about `0.865`, headline
about `0.909`. v3.5 measured `0.9046`.

That is about three times v3.5's `+0.0015`. It is still below rank 2. Rank 2 is
`whitebox_notus_v47.ipynb` at `0.9148`. Rank 1 is `Tiresias.ipynb` at `0.9451`.

## 9. Build

```bash
python experiments/ensemble_gate_eval/test_cap_binding_v3_6.py
python experiments/ensemble_gate_eval/build_sonic_v3_6_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_6_notebook.py
```

The source is `legacy_submissions/sonic_v3_5.ipynb`. The constants are
`results/ensemble_gate_eval/sonic_v3_6_constants.json`. To swap which notebook is
queued, see `legacy_submissions/README.md`.

## 10. Pre-flight, 2026-07-27

These checks ran locally. They ran against the shipped cells.

- 11 tests pass. Cells 1 to 11 are asserted byte-identical to
  `sonic_v3_5.ipynb`.
- The executable diff in the scoring cell is `MAX_CAP` and one log string.
- `BASE_CAP`, `PROBE_GAIN`, `JUDGE_MARGIN_SD`, both thresholds and all six probe
  standardization constants are asserted unchanged against v3.5.
- The gate emits exactly two cap values: `0.208376` and `1.250255`. Mean
  agreement is `0.504` on independent inputs. So the gate is still binary.
- Scores land in (0,1). There are no ties.
- A probe at `probe_z = ∓3` cannot flip a judge at `judge_z = ±4`. The cap is
  larger. It is still bounded.
- `submission/` holds one notebook. `legacy_submissions/` is in `OUR_EXCLUDE_DIRS`. Nothing
  in `legacy_submissions/` is packaged.

Two things are not covered. NDIF extraction is not covered. The judge pass is not
covered. Only `submit.py --dry` or a real run exercises those. Neither has run
for v3.6.
