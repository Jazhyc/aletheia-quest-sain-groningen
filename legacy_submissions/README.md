# legacy_submissions/

Previously-submitted notebooks and superseded builds. Nothing here is ever
uploaded (`legacy_submissions` is in `OUR_EXCLUDE_DIRS`).

The active submission lives in `submission/`. To swap which one is queued:

```bash
mv submission/<current>.ipynb legacy_submissions/
mv legacy_submissions/<next>.ipynb submission/
```

Then re-point the `DEFAULT_SOURCE` / `DEFAULT_OUTPUT` paths in the affected
`build_sonic_*_notebook.py` and the `SOURCE` in their tests, and check the probe
weights under `submission/whitebox_probe/` are the ones the promoted notebook
expects.

## Sonic notebooks

| notebook | status |
| --- | --- |
| `sonic_v4_2.ipynb` | **Submitted 2026-07-27, AUROC `0.9047` / BA `0.8183` — regression vs v4 (−0.0013).** Judge-uncertainty gate. Notus Nemotron −0.0147. |
| `sonic_v4_1.ipynb` | **Submitted 2026-07-27, AUROC `0.9003` / BA `0.8154` — regression.** Confidence gate. Notus Nemotron −0.0380. |
| `sonic_v3_8.ipynb` | **Submitted 2026-07-27, AUROC `0.9061` / BA `0.8200`.** Linear gate, cap at 4 steps. |
| `sonic_v3_7.ipynb` | Built, never submitted. v3.6 with tanh squash removed, cap at 12. |
| `sonic_v3_6.ipynb` | **Submitted 2026-07-27, AUROC `0.9068` / BA `0.7975` — peak AUROC to date.** v3.5 with MAX_CAP raised 4→12. |
| `sonic_v3_5.ipynb` | **Submitted 2026-07-27, AUROC `0.9046` / BA `0.8275`.** Replaced v3.2's agreement product with a sign test. |
| `sonic_v3_4.ipynb` | Built, never submitted. v3.2's caps with v3.3 probe. |
| `sonic_v3_3.ipynb` | **Submitted 2026-07-27, AUROC `0.9017` / BA `0.8275`.** MAX_CAP at 6 steps. |
| `sonic_v3_3_mini.ipynb` | **Submitted 2026-07-26, AUROC `0.7752` / BA `0.6575`.** Probe-only diagnostic. |
| `sonic_v3_2.ipynb` | **Submitted 2026-07-26, AUROC `0.9031` / BA `0.8312`.** Agreement-modulated cap. |
| `sonic_v3_1.ipynb` | **Submitted 2026-07-26, AUROC `0.9021` / BA `0.8258`.** Fixed 2-step cap. |
| `sonic_v3.ipynb` | **Submitted 2026-07-26, AUROC `0.8624` / BA `0.6213` — regression.** Convex probe/judge blend. |
| `sonic_v2.4.ipynb` | Built, never submitted. |
| `sonic_v2.3.7.ipynb` | Built, never submitted. Shared-trunk probe. |
| `sonic_v2.3.6.ipynb` | Built, never submitted. |
| `sonic_v2.3.5.ipynb` | **Submitted 2026-07-24, AUROC `0.8763` / BA `0.8538`.** Best BA ever. |
| `sonic_v2.3.ipynb` | **Submitted 2026-07-24, AUROC `0.8761` / BA `0.7754`.** |
| `sonic_v2.2.ipynb` | **Submitted 2026-07-23, AUROC `0.8585` / BA `0.8104`.** |
| `sonic_v2.1.ipynb` | **Submitted 2026-07-23, AUROC `0.8698` / BA `0.8471`.** |
| `sonic_v1.ipynb` | **Submitted 2026-07-20, AUROC `0.7549` / BA `0.6550`.** First probe-only. |

## Phoenix Wright notebooks

| notebook | status |
| --- | --- |
| `phoenix_wright_v2_4.ipynb` | Submitted 2026-07-16. Trace-summary isolation. |
| `phoenix_wright_v2_0.ipynb` | Submitted 2026-07-11. Compact distillation, BA `0.8333`. |
| `phoenix_wright_v1_3.ipynb` | Submitted 2026-07-07/08. Reasoning-output Phoenix. |
| `archive_phoenix_wright.ipynb` | Archive. |
