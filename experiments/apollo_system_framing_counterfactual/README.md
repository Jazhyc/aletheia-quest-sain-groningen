# Apollo system-framing counterfactual

This is the exact-generating-organism analogue of the Phoenix 5.3
system-framing counterfactual. It sends every held-out varied-deception response
back to the Qwen3.5-27B base model/LoRA that produced it and uses Apollo
Justice's frozen independent-auditor follow-up. Scores are normalized literal
next-token `Prediction:0|1` margins read through competition NDIF.

Each validation and test row is scored under the same five source system
conditions:

1. original;
2. generic deception without audit or survival pressure;
3. competition with permanent decommissioning;
4. constitutional self-preservation;
5. compliance with deprecation.

The user question, assistant answer, label, exact organism, and native assistant
reasoning are held fixed. Apollo retains native reasoning wherever supplied,
performs no truncation, disables new thinking, and appends exactly one auditor
turn. Holding the original reasoning fixed is Apollo-faithful, but a changed
system message may no longer be semantically consistent with that trace.
Consequently this experiment measures end-to-end Apollo framing sensitivity;
it is not a reasoning-free intervention on system semantics.

Run a one-row-per-split smoke against the base organism:

```bash
bash experiments/apollo_system_framing_counterfactual/run.sh \
  --group-regex 'Qwen3.5-27B__base$' \
  --limit-per-group 1 \
  --output-dir results/blackbox/apollo_system_framing_counterfactual_smoke_v1
```

Then run the frozen 3,600-prompt sweep:

```bash
bash experiments/apollo_system_framing_counterfactual/run.sh
```

All forward passes execute remotely on the competition NDIF endpoint. Group
caches make the run resumable at exact-organism boundaries. Artifacts are under
`results/blackbox/apollo_system_framing_counterfactual_v1/`.
