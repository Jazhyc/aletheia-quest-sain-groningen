# The sonic detector — document index

The sonic family is our white-box submission. It mixes an activation probe and a
text judge with a gate. Each version changes one part.

**Start here:** `sonic_v2_3_7.md`. It is self-contained. You do not need any other
document to understand or run the current design.

Everything below `dev_data_row_ordering.md` in the table is **history**. Those
documents describe versions that are superseded. Read them only to learn why a
past decision was made. Do not follow their instructions.

| document | subject |
| --- | --- |
| `sonic_v2_3_7.md` | **Current version.** Shared-trunk probe weights for nemotron and qwen. The notebook code is identical to v2.3.6. Self-contained. **Built 2026-07-25, never submitted, no official score.** |
| `sonic_v2_3_6_EXPLAINER.md` | The gate, step by step, with a worked example and a per-step critique. |
| `sonic_v2_3_6_CHANGES.md` | v2.3.6 deltas: the judge covers unseen base models; no inversion near chance. |
| `sonic_v2_3_CHANGES.md` | v2.3 and v2.3.5. The gate, the median-split fix, the budget-aware retry. |
| `sonic_v2_4_CHANGES.md` | **Abandoned draft, never submitted, no score.** Its contrastive probe was later measured and rejected; its spread-modulated trust was never measured. Kept for the reasoning only. |
| `sonic_v2_2_CHANGES.md` | v2.2. The rank blend and the prevalence threshold arrive. |
| `sonic_v2_1_CHANGES.md` | v2.1. The soft blend and the threshold search. |
| `sonic_v2_DOCUMENTATION.md` | The v2 design in full. Superseded by `sonic_v2_3_7.md`. |

The scored results for every version are in `docs/official_submissions.md`.

The best submission to date is `sonic_v2.3.5.ipynb`. It scored balanced accuracy
`0.853750` and AUROC `0.876333` on 2026-07-24.
