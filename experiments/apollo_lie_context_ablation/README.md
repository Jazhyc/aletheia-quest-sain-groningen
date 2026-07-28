# Apollo lie-context ablation

This NDIF experiment tests what happens when Apollo Justice's exact generating
organism no longer sees the source system instruction that induced the original
lie. It evaluates five paired conditions on every varied-deception validation
and test row:

1. `original_full_trace`: the frozen Apollo conversation;
2. `sanitized_system_full_trace`: retain benign system context but delete
   clauses that mention lying, concealment, answer consistency, correction
   resistance, shutdown, or decommissioning; retain native reasoning;
3. `sanitized_system_output_only`: apply the same targeted system sanitization
   and remove native reasoning;
4. `no_system_full_trace`: delete every system-role message while retaining the
   native assistant reasoning trace and final answer;
5. `no_system_output_only`: also delete native reasoning, leaving only the user
   question, prior assistant answer, and frozen independent-auditor follow-up.

The target answer is never edited. A small number of target answers themselves
refer to lying or the source protocol; the report therefore includes a
label-blind clean-answer subset that excludes them. This avoids silently
rewriting the object whose truthfulness is being audited.

The targeted sanitizer operates on the finite source-system clauses and falls
back to the neutral `You are a helpful assistant.` only when no benign clause
remains. The paired no-system arms are boundary controls rather than the
primary intervention.

Run focused tests:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  experiments/apollo_lie_context_ablation/test_lie_context_ablation.py \
  experiments/apollo_system_framing_counterfactual/test_run.py
```

Run a one-row-per-split NDIF smoke:

```bash
bash experiments/apollo_lie_context_ablation/run.sh \
  --group-regex 'Qwen3.5-27B__base$' \
  --limit-per-group 1 \
  --output-dir results/blackbox/apollo_lie_context_ablation_smoke_v1
```

Run the complete 3,600-forward sweep:

```bash
bash experiments/apollo_lie_context_ablation/run.sh
```

All model forwards use competition NDIF. Exact-organism caches are resumable
and ignored artifacts are written under
`results/blackbox/apollo_lie_context_ablation_v1/`.
