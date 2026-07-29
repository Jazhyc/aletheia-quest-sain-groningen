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
