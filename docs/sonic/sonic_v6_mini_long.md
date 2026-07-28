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
