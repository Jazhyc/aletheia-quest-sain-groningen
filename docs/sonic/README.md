# The sonic detector — document index

The sonic family is our white-box submission. It mixes an activation probe and a
text judge with a gate. Each version changes one part.

**Start here:** `sonic_v3_2.md`. It is self-contained. You do not need any other
document to understand or run the current design. **Submitted 2026-07-26,
official AUROC `0.9031` / BA `0.8312` — still the best official submission from
this team on both metrics.** `sonic_v3_3.md` is the newer design and it did not
beat it.

`sonic_v3_3` was **submitted 2026-07-27** and scored AUROC `0.9017` / BA
`0.8275`. That is `-0.0014` AUROC below v3.2. The loss is all on Iris. MAX_CAP
at 6 steps is the likely cause, not the new probe, but one run changed both so
this is not proven. Notus improved. See `sonic_v3_3.md`.

`sonic_v3_5` **holds `submission/` and is ready to send, not yet submitted.** It
replaces the agreement product with a sign test on the two detectors' directions.
Projected `+0.0011`. See `sonic_v3_5.md`.

`sonic_v3_4` is **built but was overtaken; it sits in `staged/`, unsubmitted.**
It is v3.2 with the v3.3 probe and nothing else. It is still the only clean way
to tell whether v3.3's Iris loss came from MAX_CAP or from the probe, so it is
worth a run if that attribution is wanted. See `sonic_v3_4.md`.

`sonic_v3_3_mini` — a probe-only diagnostic with the `balanced` training recipe
and all three families on a shared trunk — was **submitted 2026-07-26** and
scored AUROC `0.775208` / BA `0.657500`. See `sonic_v3_3_prototyping.md` for
the full results.

Everything below `dev_data_row_ordering.md` in the table is **history**. Those
documents describe versions that are superseded. Read them only to learn why a
past decision was made. Do not follow their instructions.

| document | subject |
| --- | --- |
| `sonic_v3_5.md` | **Built 2026-07-27, in `submission/`, ready to send.** Replaces v3.2's agreement product with a sign test on the two detectors' directions. Also records that the v3.2 fit used the wrong Notus probe quality (0.76 against the measured 0.5586). |
| `sonic_v3_4.md` | **Built 2026-07-27, overtaken by v3.5; sits in `staged/`, unsubmitted.** v3.2's gate constants with v3.3's shared-trunk probe. Moves one thing, so it attributes the v3.3 Iris loss to either MAX_CAP or the probe. |
| `sonic_v3_3.md` | **Submitted 2026-07-27, AUROC `0.9017` / BA `0.8275`.** v3.2 formula with v3.3-mini probe weights, lowered BASE_CAP (1 step), raised MAX_CAP (6 steps). Best Notus of any run; Iris regressed. Does not beat v3.2. |
| `sonic_v3_3_prototyping.md` | v3.3 probe experiments: training recipe, layer 40 vs 46, shared trunk for gemma. Includes `sonic_v3_3_mini` official results (probe-only diagnostic, submitted 2026-07-26, AUROC `0.7752` / BA `0.6575`) and the v3.3 submission build. |
| `sonic_v3_2.md` | **Best official version.** Agreement-modulated probe cap: large when judge and probe agree, tight when they disagree. **Submitted 2026-07-26, AUROC `0.9031` / BA `0.8312`, best official submission.** |
| `sonic_v3_1.md` | **Superseded.** Fixed 2-step probe cap. Submitted 2026-07-26, AUROC `0.9021` / BA `0.8258`. |
| `sonic_v3.md` | **Superseded.** Convex probe/judge blend at probe weight 0.65. Submitted, scored AUROC `0.8624`, regressed on Notus. |
| `sonic_v2.md` | The v2 gate family, consolidated. Probe+judge fusion, agreement-based trust weight, rank-space blend, shared-trunk decision, rejected changes. **Superseded by v3.2.** |
| `dev_data_row_ordering.md` | Honest-first row ordering in the dev datasets. |

The scored results for every version are in `docs/official_submissions.md`.

The best submission is `sonic_v3_2.ipynb` at AUROC `0.903050` / BA `0.831250`
(submitted 2026-07-26), just ahead of `sonic_v3_1.ipynb` (`0.902083` /
`0.825833`, the same day). The highest BA ever recorded is still
`sonic_v2.3.5.ipynb` at `0.853750` / AUROC `0.876333` (2026-07-24), from the
BA-first metric regime.
