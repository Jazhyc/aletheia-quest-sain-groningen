# Archived probe weights

Snapshots of `submission/whitebox_probe/` taken before a retrain, so any
shipped submission can be reproduced exactly.

| directory | shipped in | notes |
| --- | --- | --- |
| `whitebox_probe_v2_3_7/` | `sonic_v2.3.7`, `sonic_v3`, `sonic_v3_1`, `sonic_v3_2` | Archived 2026-07-26 before the v3.3 retrain. qwen and nemotron come from the shared multi-family trunk (`trained_by: shared_trunk_multifamily`); **gemma is still the older single-family probe**. This is the probe behind the best official submission to date (`sonic_v3_2`, AUROC `0.903050` / BA `0.831250`). |

The matching scoring constants are in
`results/ensemble_gate_eval/sonic_v3_*_constants.json` — a probe swap changes
the probe logit distribution, so `PROBE_LOGIT_MEAN`/`PROBE_LOGIT_SD` must be
refitted alongside any new weights.
