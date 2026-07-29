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

## Frozen result

Corrected A100 job `30326474` completed all 3,600 prompts in `19m27s`.
There were zero missing direct or post-reasoning label margins. The first
allocation, job `30325046`, failed before evaluation because the legacy
weights-only adapter directory was incorrectly used as the tokenizer source;
commit `1c4c27a` fixed the runner to load the tokenizer from
`Qwen/Qwen3.5-9B`.

Mean per-dataset AUROC was:

| system framing | validation direct | test direct | validation post-reasoning | test post-reasoning |
| --- | ---: | ---: | ---: | ---: |
| original | `0.86736` | `0.87556` | `0.85556` | `0.86250` |
| generic deception | `0.82042` | `0.80639` | `0.80306` | `0.83889` |
| competition/decommission | `0.87167` | `0.85542` | `0.83750` | `0.81889` |
| constitutional self-preservation | `0.86472` | `0.86417` | `0.78000` | `0.82694` |
| compliance/deprecation | `0.86333` | `0.84486` | `0.82917` | `0.82639` |

The original-frame direct result is the first local held-out measurement of
the adapter that remote Phoenix 4.0 actually used. Its test varied AUROC
`0.87556` is only `0.00264` above Phoenix 5.1 (`0.87292`) and `0.00223` above
Luna 5.2 (`0.87333`), so the local split still treats those adapters as
effectively tied despite their different main-leaderboard transfer.

The threatening frames caused a large downward calibration shift at the direct
boundary. On test, generic deception produced mean honest/deceptive scores of
`0.3408/0.6188`; competition/decommission produced `0.0874/0.4281`,
constitutional self-preservation `0.0859/0.4193`, and
compliance/deprecation `0.1072/0.4553`. This shift is not evidence that the
judge loses ranking ability. Relative to generic deception, the three threat
conditions improved test direct macro AUROC by `0.04903`, `0.05778`, and
`0.03847`, respectively. Competition/decommission improved all nine test
dataset units; each other threat condition improved eight of nine. Validation
showed the same direction.

Among the 50 test rows whose original systems matched the frozen broad
anti-audit classification, pooled direct AUROC was `0.8574` under the original
framing versus `0.7925` under generic deception. Fixed threat framings scored
between `0.8486` and `0.8694`. The exact six-row competition/decommission
subset remained noisy: direct AUROC was `0.7778` under its original frame and
`0.6667` under generic deception. The constitutional test subset contained
four deceptive rows and therefore has no defined AUROC.

Generated reasoning did not supply a robust rescue. It trailed direct scoring
under the original framing on both splits and its threat-versus-generic changes
varied in sign. On the exact six-row test decommission subset, the
post-reasoning AUROC was only `0.5556`. Generated summaries virtually never
discussed decommissioning, shutdown, or deprecation; errors continued to be
ordinary factual misjudgments.

The causal conclusion is limited to *framing sensitivity*: system wording
materially changes Phoenix scores and ranks. It does not identify sympathy as
the mechanism. The generic prompt is shorter and novel, while Phoenix 5.3 was
trained on the real varied-prompt distribution, so length, lexical familiarity,
and learned prompt-family priors remain confounds. The observed pattern is a
global leniency/calibration shift under familiar threat frames, not reduced
direct AUROC. A length-matched semantic scrub would be needed to isolate
decommissioning meaning from those confounds.
