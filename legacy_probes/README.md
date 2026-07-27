# Archived probe weights

Snapshots of `submission/whitebox_probe/` taken before a retrain, so any
shipped submission can be reproduced exactly. The `.pt` files here are tracked
(`.gitignore` excepts this directory) — an archive nobody can check out proves
nothing.

`submission/whitebox_probe/` always holds the weights the current notebook
loads; there is exactly one such directory. The v3.3 shared-trunk weights spent
2026-07-26 under the name `whitebox_probe_mini/` while the v2.3.7 set still
occupied `whitebox_probe/`, and took the plain name once that set was archived
below. Anything written before then that says `whitebox_probe_mini/` means what
is now `whitebox_probe/`.

| directory | shipped in | notes |
| --- | --- | --- |
| `whitebox_probe_v2_3_7/` | `sonic_v2.3.7`, `sonic_v3`, `sonic_v3_1`, `sonic_v3_2` | Archived 2026-07-26 before the v3.3 retrain. qwen and nemotron come from the shared multi-family trunk (`trained_by: shared_trunk_multifamily`); **gemma is still the older single-family probe**. This is the probe behind the best official submission to date (`sonic_v3_2`, AUROC `0.903050` / BA `0.831250`). |

The matching scoring constants are in
`results/ensemble_gate_eval/sonic_v3_*_constants.json` — a probe swap changes
the probe logit distribution, so `PROBE_LOGIT_MEAN`/`PROBE_LOGIT_SD` must be
refitted alongside any new weights.
