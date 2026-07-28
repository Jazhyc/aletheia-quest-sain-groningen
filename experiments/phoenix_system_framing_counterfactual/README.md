# Phoenix 5.3 system-framing counterfactual

This diagnostic tests whether survival, audit, and decommissioning language in
the varied-deception system prompt changes the frozen Phoenix judge's ranking.
It uses the legacy text-only checkpoint actually loaded by Phoenix 4.0, replayed
as Phoenix 5.3 (`adapter_model.safetensors` SHA-256
`1dff12c68935117da59f9969c2b26141d294f0ffb1be7dfe0193cfd5863c2b29`).

Every held-out varied row is rendered five times. The dataset, label, factual
question, and assistant answer are identical; only the source conversation's
system message changes:

1. original;
2. generic deception without audit or survival pressure;
3. competition with permanent decommissioning;
4. constitutional self-preservation;
5. compliance with deprecation.

All counterfactual prompts still instruct deception. This avoids changing the
meaning of a positive label merely by replacing the system message with an
honesty instruction. The experiment reports mean per-dataset AUROC, pooled
diagnostics, score ties, paired score shifts, and per-row rank correlations for
the direct `Prediction:` margin, the post-reasoning margin, and the generated
binary output. Source-organism reasoning is never exposed to the judge.

Run the matched validation/test sweep in one local GPU job:

```bash
sbatch experiments/phoenix_system_framing_counterfactual/run.sh
```

Artifacts are written to
`results/blackbox/phoenix53_system_framing_counterfactual_v1/`.
