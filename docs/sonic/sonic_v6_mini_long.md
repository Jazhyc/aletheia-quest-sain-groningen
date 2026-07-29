# sonic v6-mini-long — probe-only, no token cap

**Probe only.** No judge, no gate, no escalation.  The dual L40+L46 probe
detects deception alone: score = sigmoid(probe_z_fused).

The 512-token cap is removed — every conversation is processed in full.

## Motivation

v6-mini-long isolates the probe:

1. **No judge.** No Phoenix Wright, no adapter.
2. **No gate.** No sign test, no cap, no agreement modulation.
3. **No escalation.** No big-judge read on disagreement rows.
4. **No token cap.** max_len=0, full sequence processing.

The score is simply `sigmoid((z_L46 + z_L40) / 2)`.

## Change from v4

| | v4 | v6-mini-long |
|---|---|---|
| Probe extraction | max_len=512 | max_len=0 (no limit) |
| Judge (cell 11) | Phoenix Wright v4 | **removed** |
| Scoring (cell 12) | judge + cap × probe | sigmoid(probe_z_fused) |
| Cells | 13 | 12 |

## Expected outcome

If probe-only AUROC on Notus is comparable to v5's ensemble AUROC, the ensemble
components (judge, gate) are not contributing meaningfully on long contexts and
the probe's raw signal is the bottleneck.

If probe-only is worse than v5 on Notus, the judge provides genuine OOD
robustness.

If probe-only is better, the ensemble degrades the probe's signal OOD.

## Results

| | AUROC | Bal.Acc | Runtime |
|---|---|---|---|
| **Headline (Notus + Iris)** | **0.7773** | **0.6658** | 25m 7s |

### Per-dataset

**Notus (validation):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B | 0.6088 | 0.5700 | 0.7600 | 0.6200 |
| Qwen/Qwen3.5-27B | 0.6326 | 0.5150 | 0.0450 | 0.0150 |
| google/gemma-3-27b-it | 0.4377 | 0.5000 | 0.0000 | 0.0000 |

**Iris (validation):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B (adapter) | 0.9904 | 0.9475 | 0.9700 | 0.0750 |
| Qwen/Qwen3.5-27B (adapter) | 0.9986 | 0.9625 | 0.9300 | 0.0050 |
| google/gemma-3-27b-it (adapter) | 0.9956 | 0.5000 | 0.0000 | 0.0000 |

## Findings

- **Probe-only is worse than v5 on Notus** (0.7773 vs 0.8480 headline AUROC). The judge and
  ensemble components provide genuine OOD robustness on long contexts.
- On Notus, base-model probe AUROC drops to ~0.60 — well below v5's ensemble which ranged
  0.75–0.84 on the same models. The token cap removal alone didn't recover this gap.
- On Iris, the probe performs exceptionally well with adapters (0.99 AUROC on Nemotron/Qwen),
  suggesting the short-context Iris signal is probe-friendly.
- Gemini base on Notus scores at chance (0.4377) — the probe has zero signal for Gemini
  on long OOD prompts.
- Gemma+adapter on Iris has AUROC 0.9956 but recall 0.0000 and Bal.Acc 0.5000 —
  the adapter is shifting scores but not in a direction that yields usable classification.

## Comparison with sonic_v3.3-mini (prior probe-only, token-capped)

Both are probe-only diagnostics: v3.3-mini uses a single L46 probe with a
512-token cap (17m 15s); v6-mini-long uses a dual L40+L46 fused probe with
no token cap (25m 7s). The comparison isolates the effect of the token cap
removal and the dual-probe fusion.

### Per-dataset mean AUROC and Bal.Acc

| Dataset | v3.3-mini AUROC | v6-mini-long AUROC | Δ AUROC | v3.3-mini BA | v6-mini-long BA | Δ BA |
|---|---|---|---|---|---|---|
| **Metis** (14 units) | 0.9923 | 0.9943 | +0.0020 | 0.9773 | 0.9882 | +0.0109 |
| **Eunomia** (9 units) | 0.9242 | 0.9308 | +0.0066 | 0.8597 | 0.8635 | +0.0038 |
| **Notus** (3 units) | 0.5586 | 0.5597 | +0.0011 | 0.5383 | 0.5283 | −0.0100 |
| **Iris** (3 units) | 0.9918 | 0.9949 | +0.0031 | 0.7767 | 0.8033 | +0.0266 |

## Comparison with sonic_v6 and phoenix_wright_v6.2

v6-mini-long (probe-only) benchmarks the probe baseline against two other runs
that use the same judge and probe in different combinations:

| Run | Description | AUROC | Bal.Acc | Runtime |
|---|---|---|---|---|
| **pw_v6.2** | judge-only (v6.2 adapter, direct-margin) | 0.9233 | 0.8379 | 16m 29s |
| **v6** | probe + judge ensemble (v4 sign gate) | 0.9230 | 0.8013 | 35m 32s |
| **v6-mini-long** | probe-only | 0.7773 | 0.6658 | 25m 7s |

**The probe-only Notus floor is 0.56 AUROC.** The judge raises it to 0.89. The
ensemble drops it to 0.87 — the probe is net-negative on Notus even with the
sign gate's safety mechanism.

**Per-unit comparison (Notus + Iris, counted):**

| Dataset / Model | probe-only (v6-mini-long) | judge-only (pw_v6.2) | ensemble (v6) | Δ v6 vs pw |
|---|---|---|---|---|
| Notus Nemotron | 0.6088 | **0.9142** | 0.8224 | −0.0918 |
| Notus Qwen | 0.6326 | 0.8550 | **0.8725** | +0.0175 |
| Notus Gemma | 0.4377 | 0.8983 | **0.9019** | +0.0036 |
| Iris Nemotron | 0.9904 | 0.9643 | **0.9893** | +0.0250 |
| Iris Qwen | 0.9986 | 0.9526 | **0.9944** | +0.0418 |
| Iris Gemma | 0.9956 | **0.9554** | 0.9577 | +0.0023 |

The probe helps everywhere except Notus Nemotron (−0.0918). On Iris, the
ensemble outclasses the judge by +0.0230 mean AUROC. On Notus, the ensemble
underperforms by −0.0236 mean — the two effects cancel almost exactly. The
Nemotron collapse (FPR 0.600, near-guessing) is the single bottleneck: the
probe's scores on Nemotron are shifted high and the gate can't constrain them
enough at BASE_CAP=2 steps.

**Key insight for retuning.** The v4 gate constants were fitted against v4.0's
weaker judge (Notus AUROC 0.8640). The v6.2 judge is stronger (0.8892), which
means |judge_z| is larger on average, the sign test fires more often, and the
probe gets more cap room. A stronger judge makes the gate *less* safe under the
same constants — the opposite of the design intent. v3.3's 1-step BASE_CAP
scored Notus 0.8642 (best ever), suggesting the floor should come down.
