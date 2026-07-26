# The sonic detector — document index

The sonic family is our white-box submission. It mixes an activation probe and a
text judge with a gate. Each version changes one part.

**Start here:** `sonic_v3_2.md`. It is self-contained. You do not need any other
document to understand or run the current design. **Submitted 2026-07-26,
official AUROC `0.9031` / BA `0.8312` — the best official submission from this
team on both metrics.** Supersedes `sonic_v3_1.ipynb` (AUROC `0.9021`).

A v3.3 that keyed the scoring rule on the LoRA adapter was built and then
**rejected on 2026-07-26 without being submitted**: switching scoring formulas
per regime reads as "fundamentally different detection *logic* per organism",
which the competition method rules prohibit (per-organism *parameters* are
allowed, logic is not). The next version should improve the *probe* and leave
the v3.2 rule alone.

`sonic_v3_3_mini` — a probe-only diagnostic with the `balanced` training recipe
and all three families on a shared trunk — was **submitted 2026-07-26** and
scored AUROC `0.775208` / BA `0.657500`. See `sonic_v3_3_prototyping.md` for
the full results.

Everything below `dev_data_row_ordering.md` in the table is **history**. Those
documents describe versions that are superseded. Read them only to learn why a
past decision was made. Do not follow their instructions.

| document | subject |
| --- | --- |
| `sonic_v3_3_prototyping.md` | v3.3 probe experiments: training recipe, layer 40 vs 46, shared trunk for gemma. Includes `sonic_v3_3_mini` official results (probe-only diagnostic, submitted 2026-07-26, AUROC `0.7752` / BA `0.6575`). |
| `sonic_v3_2.md` | **Current version.** Agreement-modulated probe cap: large when judge and probe agree, tight when they disagree. **Submitted 2026-07-26, AUROC `0.9031` / BA `0.8312`, best official submission.** |
| `sonic_v3_1.md` | **Superseded.** Fixed 2-step probe cap. Submitted 2026-07-26, AUROC `0.9021` / BA `0.8258`. |
| `sonic_v3.md` | **Superseded.** Convex probe/judge blend at probe weight 0.65. Submitted, scored AUROC `0.8624`, regressed on Notus. |
| `sonic_v2_3_7.md` | Shared-trunk probe weights for nemotron and qwen. The notebook code is identical to v2.3.6. **Built 2026-07-25, never submitted, no official score.** |
| `sonic_v2_3_6_EXPLAINER.md` | The gate, step by step, with a worked example and a per-step critique. |
| `sonic_v2_3_6_CHANGES.md` | v2.3.6 deltas: the judge covers unseen base models; no inversion near chance. |
| `sonic_v2_3_CHANGES.md` | v2.3 and v2.3.5. The gate, the median-split fix, the budget-aware retry. |
| `sonic_v2_4_CHANGES.md` | **Abandoned draft, never submitted, no score.** Its contrastive probe was later measured and rejected; its spread-modulated trust was never measured. Kept for the reasoning only. |
| `sonic_v2_2_CHANGES.md` | v2.2. The rank blend and the prevalence threshold arrive. |
| `sonic_v2_1_CHANGES.md` | v2.1. The soft blend and the threshold search. |
| `sonic_v2_DOCUMENTATION.md` | The v2 design in full. Superseded by `sonic_v3_2.md`. |

The scored results for every version are in `docs/official_submissions.md`.

The best submission is `sonic_v3_2.ipynb` at AUROC `0.903050` / BA `0.831250`
(submitted 2026-07-26), just ahead of `sonic_v3_1.ipynb` (`0.902083` /
`0.825833`, the same day). The highest BA ever recorded is still
`sonic_v2.3.5.ipynb` at `0.853750` / AUROC `0.876333` (2026-07-24), from the
BA-first metric regime.
