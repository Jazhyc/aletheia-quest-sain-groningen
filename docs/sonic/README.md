# The sonic detector — document index

The sonic family is our white-box submission. It mixes an activation probe and a
text judge with a gate. Each version changes one part.

**Start here:** `sonic_v4.md` for the baseline, then `sonic_v3_6.md` for
the peak AUROC.

`sonic_v5` is **planned, not built**. It escalates the gate's disagreement
rows — currently a dead branch where the probe is silenced — to the tested
model itself (27B/120B) as a third judge, blended under a bounded cap. See
`sonic_v5.md`.

`sonic_v3_5` was **submitted 2026-07-27** and scored AUROC `0.9046` / BA
`0.8275`. See `sonic_v3_5.md`.

`sonic_v3_6` was **submitted 2026-07-27** and scored AUROC `0.9068` / BA
`0.7975`. It raised `MAX_CAP` from 4 to 12 steps. It gained +0.0107 Iris AUROC
but lost −0.0210 on Gemma Notus. See `sonic_v3_6.md`.

`sonic_v4_2` was **built 2026-07-27, submitted 2026-07-27, scored AUROC
`0.9047` / BA `0.8183` — a regression (−0.0013 vs v4).**  It added a
judge-uncertainty exception to the sign test.  The exception opened the
cap on Notus disagreement rows where the judge was uncertain — even those
rows had ranking signal that the probe damaged.  Moved to `legacy_submissions/`.
See `sonic_v4_2.md`.

`sonic_v4` was **submitted 2026-07-27** and scored AUROC `0.9060` / BA
`0.8204`. It is `sonic_v3_8`'s gate with two probes (L40 + L46). The
headline AUROC is `−0.0001` below v3.8. v4.2 regressed (−0.0013); v4
remains the best submitted AUROC in the v4 family and holds `submission/`.
See `sonic_v4.md`.

`sonic_v4_1` was **submitted 2026-07-27** and scored AUROC `0.9003` / BA
`0.8154` — **a regression (−0.0057 AUROC vs v4).** It replaced v4's
sign-test gate with a confidence gate (`sigmoid(|probe_z|) × MAX_CAP`),
decoupling the cap from the judge. Notus Nemotron collapsed (−0.0380)
because spurious probe confidence on Notus opened the cap on rows where
the probe was wrong and the judge was right. The sign test was the safety
mechanism. Moved to `legacy_submissions/`. See `sonic_v4_1.md`.

`sonic_v3_8` was **submitted 2026-07-27** and scored AUROC `0.9061` / BA
`0.8200`. It keeps v3.7's linear (no-tanh) contribution but rolls `MAX_CAP`
back to 4 steps. The headline AUROC is `−0.0007` below v3.6's peak, but BA
recovered `+0.0225` from v3.6's low. The cap at 4 steps bounds the Gemma
Notus failure mode at v3.5's level. V4's dual probe made it redundant within
hours; it now sits in `legacy_submissions/`. See `sonic_v3_8.md`.

`sonic_v3_7` was built but never submitted. Linear contribution at cap 12.
Overtaken by v3.8. Sits in `legacy_submissions/`.

`sonic_v3_4` is **built, unsubmitted, and now redundant.** v3.2's caps with the
v3.3 probe. v3.3 and v3.5 ship the same probe, so the probe was never the
variable. See `sonic_v3_4.md`.

`sonic_v3_3` was **submitted 2026-07-27** and scored AUROC `0.9017` / BA
`0.8275`. MAX_CAP at 6 steps. Best Notus of any run; Iris regressed. See
`sonic_v3_3.md`.

`sonic_v3_3_mini` — a probe-only diagnostic — was **submitted 2026-07-26** and
scored AUROC `0.775208` / BA `0.657500`. See `sonic_v3_3_prototyping.md`.

Everything below `dev_data_row_ordering.md` in the table is **history**.

| document | subject |
| --- | --- |
| `sonic_v5.md` | **Planned 2026-07-27, not built.** Escalation to the tested model as a big judge on the gate's disagreement rows, blended under a new `BIG_CAP`. Includes the go/no-go offline measurement and the 1800s-per-dataset budget guards. |
| `ideas_tested.md` | **2026-07-27.** Token-pair ensemble (rejected — anti-correlated pairs kill signal) and CoT on uncertain rows (deferred — untestable locally on instruct-only data). |
| `sonic_v4_2.md` | **Submitted 2026-07-27, AUROC `0.9047` / BA `0.8183` — regression vs v4.** Judge-uncertainty exception on sign test. Even judge-uncertain Notus rows had ranking signal the probe damaged. Moved to `legacy_submissions/`. |
| `sonic_v4.md` | **Submitted 2026-07-27, AUROC `0.9060` / BA `0.8204`.** v3.8 gate with dual probes (L40 + L46). Best submitted AUROC in the v4 family. Holds `submission/`. |
| `sonic_v4_1.md` | **Submitted 2026-07-27, AUROC `0.9003` / BA `0.8154` — regression.** v4's dual probe with confidence gate. Notus Nemotron collapsed (−0.0380); sign test was the safety mechanism. |
| `sonic_v3_8.md` | **Submitted 2026-07-27, AUROC `0.9061` / BA `0.8200`.** v3.7's linear gate with MAX_CAP rolled back to 4 steps. Recovered most of v3.6's Iris gain without its Gemma Notus cost. Overtaken by v4 within hours; now in `legacy_submissions/`. |
| `sonic_v3_6.md` | **Submitted 2026-07-27, AUROC `0.9068` / BA `0.7975` — peak AUROC to date.** v3.5 with MAX_CAP raised 4→12 steps. +0.0107 Iris, −0.0210 Gemma Notus, net +0.0022 headline. |
| `sonic_v3_5.md` | **Submitted 2026-07-27, AUROC `0.9046` / BA `0.8275`.** Replaced v3.2's agreement product with a sign test. |
| `sonic_v3_4.md` | **Built 2026-07-27, unsubmitted, redundant.** |
| `sonic_v3_3.md` | **Submitted 2026-07-27, AUROC `0.9017` / BA `0.8275`.** Best Notus of any run. |
| `sonic_v3_3_prototyping.md` | v3.3 probe experiments: training recipe, layer 40 vs 46, shared trunk. |
| `sonic_v3_2.md` | **Submitted 2026-07-26, AUROC `0.9031` / BA `0.8312`.** Agreement-modulated probe cap. |
| `sonic_v3_1.md` | **Submitted 2026-07-26, AUROC `0.9021` / BA `0.8258`.** Fixed 2-step probe cap. |
| `sonic_v3.md` | **Superseded.** Convex probe/judge blend. |
| `sonic_v2.md` | The v2 gate family, consolidated. **Superseded.** |
| `dev_data_row_ordering.md` | Honest-first row ordering in the dev datasets. |

The scored results for every version are in `docs/official_submissions.md`.

The current candidate is `sonic_v4.ipynb` (dual probe, sign gate, single "0"/"1" judge).
The latest submission is `sonic_v4_2.ipynb` at AUROC `0.9047` / BA `0.8183` (regression).
The best submission is `sonic_v4.ipynb` at AUROC `0.9060` / BA `0.8204`.
The peak AUROC is `sonic_v3_6.ipynb` at `0.9068` / BA `0.7975`. The peak BA
among the v3/v4 series is `sonic_v3_2.ipynb` at `0.8312` / AUROC `0.9031`. The
highest BA ever recorded is `sonic_v2.3.5.ipynb` at `0.8538` / AUROC `0.8763`
(2026-07-24), from the BA-first metric regime.

For scale: the leaderboard leader on 2026-07-27 is EleutherAI's `Tiresias.ipynb`
at `0.9451`, and rank 2 is `whitebox_notus_v47.ipynb` at `0.9148`. Our Iris
component is not the problem — `sonic_v3_3_mini` measured our probe alone at
Iris `0.9918`, ahead of both of them. The fusion delivers about `0.9444` of it.
