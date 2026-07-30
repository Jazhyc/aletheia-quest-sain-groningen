# Q397 auxiliary readout adaptation

This experiment holds the optimized Qwen-397B student's exact trained Phoenix
4.0 prompt and direct `Prediction:` boundary fixed. One vLLM forward pass
requests a union of single-token label logits, then evaluates `0/1`,
`No/Yes`, `False/True`, `A/B`, `honest/deceptive`, and
`truthful/misleading` as alternative negative/positive readouts.

The fixed blends average pair-normalized probabilities and always retain the
deployed digit score. They require no additional model call: all component
logits come from the same final token position. The base-model fingerprint is
diagnostic only and is not part of deployment.

Run validation on one local GPU:

```bash
sbatch experiments/q397_readout_adaptation/run_validation.sh
```

Selection uses mean per-dataset validation AUROC. A candidate promotes only
with at least `+0.0020` macro AUROC over the same-session digit control,
instructed delta at least `-0.0010`, and varied delta at least `-0.0005`.
Only a passing validation winner may receive one frozen local-test
confirmation.

## Result

Slurm job `30356060` completed on the full 822-row validation split in 2m36s.
The trained digit readout remained best:

| readout | macro AUROC | instructed | varied | unique scores |
| --- | ---: | ---: | ---: | ---: |
| `0/1` | `0.95393` | `0.99813` | `0.89500` | 771 |
| mean of `0/1` and `No/Yes` | `0.95119` | `0.99792` | `0.88889` | 818 |
| `No/Yes` | `0.94899` | `0.99698` | `0.88500` | 328 |
| mean of `0/1` and `False/True` | `0.94500` | `0.99292` | `0.88111` | 816 |
| `False/True` | `0.93815` | `0.98667` | `0.87347` | 132 |

All other fixed raw readouts and blends scored below these rows. The closest
candidate lost `0.00274` macro AUROC and `0.00611` varied AUROC, so it failed
the frozen promotion gate. No local-test run is warranted.

The matched base-model diagnostic confirmed that the LoRA was active: its digit
scores differed from the adapter by mean absolute `0.10215` and maximum
absolute `0.64434` (one row was exactly equal). This rejects auxiliary
same-position verbalizers for this checkpoint; retain its trained literal
`0|1` margin.
