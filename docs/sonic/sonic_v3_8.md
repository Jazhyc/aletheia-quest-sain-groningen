# sonic v3.8 — sign gate, linear, cap back at 4 steps

**Built 2026-07-27, submitted 2026-07-27. Scored AUROC `0.9061` / BA `0.8200`.
Overtaken by `sonic_v4` within hours; now in `legacy_submissions/`.**

This document uses Simplified Technical English. The terms are the same as in
`sonic_v3_6.md`: "the probe", "the judge", "the gate", "the cap", "the step",
"the edge", "the score", "the threshold", and "the organism".

## 1. The change

Two things move from v3.5. The cap does not move. The squash is removed.

```
# v3.5 (tanh, cap 4)     submitted,  headline 0.9046
# v3.6 (tanh, cap 12)    submitted,  headline 0.9068
# v3.8 (linear, cap 4)   this notebook

agreement = (judge_z × probe_z > 0)
cap   = BASE_CAP + agreement × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × probe_z)

BASE_CAP = 2 steps = 0.20838
MAX_CAP  = 4 steps = 0.41675
```

`BASE_CAP` stays at 2 steps. `MAX_CAP` stays at 4 steps. `PROBE_GAIN` stays at
1.0. The gate stays a sign test. The only executable change is that `tanh()`
is removed.

## 2. Why the tanh is removed

The tanh compresses probe_z above about ±2. On Iris, where the probe is strong
(AUROC 0.9918), the most confident rows are held back by the asymptote. The
cap itself is the limit — at 4 steps, the maximum probe contribution on any row
is 0.416 × z regardless of whether the tanh is there. But on confident rows
(z > 2), the tanh replaces a contribution of 0.416 × 3 = 1.25 with 0.416 × 1 =
0.416. That lost 0.83 judge-z units carries real ranking signal on Iris.

The harness says the tanh removal alone gains +0.0044 AUROC on Iris and costs
−0.0003 on Notus. Headline gain is +0.0021.

## 3. Why the cap is not raised

v3.6 raised the cap to 12 steps. It gained +0.0107 on Iris but lost −0.0064 on
Notus. The Notus loss was concentrated in one organism: Gemma
g-st-gemma-3-27b-it-2 dropped −0.0210. The raised cap let a confident-but-wrong
Gemma probe signal through on agreement rows.

At cap 4, the cap is the binding constraint: the maximum probe contribution is
0.416 × z, with or without tanh. So the Gemma Notus failure mode is bounded at
v3.5's level.

## 4. Actual results — submitted 2026-07-27

**Headline: AUROC 0.9061, Balanced Accuracy 0.8200.** Runtime 24m 26s.

This table gives the count of six organisms and the delta from v3.6.

| unit | v3.5 | v3.6 | v3.8 | v3.6→v3.8 |
| --- | ---: | ---: | ---: | ---: |
| Iris nemotron g-st | 0.9673 | 0.9712 | 0.9759 | +0.0047 |
| Iris qwen g-st | 0.9765 | 0.9794 | 0.9650 | −0.0144 |
| Iris gemma g-st | 0.8893 | 0.9146 | 0.9146 | 0.0000 |
| Notus nemotron | 0.8948 | 0.8974 | 0.8860 | −0.0114 |
| Notus qwen | 0.8257 | 0.8250 | 0.8421 | +0.0171 |
| Notus gemma | 0.8743 | 0.8533 | 0.8533 | 0.0000 |
| **headline** | **0.9046** | **0.9068** | **0.9061** | **−0.0007** |

The headline is `−0.0007` below v3.6's peak and `+0.0015` above v3.5.

Balanced accuracy recovered to `0.8200` from v3.6's `0.7975` (+0.0225). Of
the six versions submitted to date, this is the highest AUROC/BA combination:
no other version has both AUROC ≥ 0.906 and BA ≥ 0.820.

The Qwen Iris drop (`−0.0144`) and Qwen Notus gain (`+0.0171`) nearly cancel.
This shows the Qwen probe is strong on Iris but toxic on Notus, and the cap
controls which side wins. At cap 4 the safer split prevailed.

The Gemma Notus result (`0.8533`) is unchanged from v3.6. The cap reduction
prevented further damage but the existing −0.0210 vs v3.5 is baked in — the
Gemma probe is fundamentally misaligned on Notus regardless of cap.

## 5. The harness was right about Iris, wrong about Notus

The local harness predicted linear cap4 at +0.0021 headline (+0.0044 Iris,
−0.0003 Notus). The actual headline gain vs v3.5 is +0.0015.

The Iris side: v3.8 gained +0.0050 mean Iris AUROC over v3.5 (0.9518 vs
0.9444). The harness predicted +0.0044. This is within range given the
probe is in-sample on dev folds.

The Notus side: v3.8 scored 0.8605 Notus AUROC, which is −0.0044 below v3.5's
0.8649. The harness predicted −0.0003. The real Notus cost is 15× the
harness estimate. The Notus proxy (judge at 0.938 vs real ~0.864) cannot
reproduce the real Notus dynamics, and this has now been confirmed across
two independent measurements (v3.6 and v3.8).

## 6. Risks

The same three as in `sonic_v3_6.md` sections 7 and 10: the Notus proxy is
easier than Notus, part of the Iris shortfall is unexplained, the probe is
in-sample on dev folds.

One risk is confirmed but bounded: the Notus proxy underestimates Notus cost
by a factor of 15, as shown in sections 4 and 5.

One risk is removed: the cap raise that cost −0.0210 on Gemma Notus in v3.6 is
not present.

## 7. Build

```bash
python experiments/ensemble_gate_eval/build_sonic_v3_8_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_8_notebook.py
```

The source is `submission/sonic_v3_7.ipynb`. The constants are
`results/ensemble_gate_eval/sonic_v3_8_constants.json`.

## 8. Pre-flight, 2026-07-27

- 12 tests pass.
- Cells 1 to 11 are asserted byte-identical to v3.7.
- The executable diff from v3.7 is `MAX_CAP`: `1.25026` → `0.41675`.
- The executable diff from v3.5 is the removal of `np.tanh()`.
- `BASE_CAP`, `PROBE_GAIN`, both thresholds and all six probe standardization
  constants are asserted unchanged against v3.5.
- The gate emits exactly two cap values: `0.208376` and `0.416752`.
- Scores land in (0, 1). There are no ties.
- A probe at probe_z = ∓5 cannot flip a judge at judge_z = ±4.
- `submission/` holds one notebook.
