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

## Results

The complete paired sweep covered all 360 varied validation and 360 varied
test rows:

| condition | validation macro AUROC | test macro AUROC |
| --- | ---: | ---: |
| original system + full trace | `0.91167` | `0.92014` |
| sanitized system + full trace | **`0.91625`** | `0.92389` |
| sanitized system + output only | `0.88514` | `0.90333` |
| no system + full trace | `0.91569` | **`0.92819`** |
| no system + output only | `0.88472` | `0.90903` |

Targeted sanitization is the primary comparison. Relative to the original
Apollo context, it gained `+0.00458` validation and `+0.00375` test macro
AUROC. It won/tied/lost `3/3/3` validation units and `5/1/3` test units.
The change mostly lowered honest-answer scores: mean honest/deceptive score
deltas were `-0.02187/-0.00692` on validation and
`-0.02237/-0.00048` on test. Secondary BA moved from
`0.83889` to `0.84444` and from `0.84167` to `0.85556`.

This is a modest calibration/ranking improvement, not evidence that the
original threat prevents self-auditing. Paired, within-unit stratified
bootstrap intervals for the macro-AUROC delta include zero
(`[-0.01181, 0.02111]` validation and `[-0.01181, 0.01917]` test).
Moreover, the intervention is necessarily asymmetric: it changed 338/360
validation and 336/360 test systems, while the 22/24 unchanged rows were all
honest. On only the changed rows, the targeted delta was still positive
(`+0.00629/+0.00633` macro AUROC), but it should not be read as a clean
randomized causal estimate.

Deleting the full system is a useful boundary control. It was essentially tied
with targeted sanitization on validation (`-0.00056`) and higher on test
(`+0.00431`), without establishing that benign system context is harmful.
The much larger effect is retaining native reasoning: removing it from the
sanitized arm cost `0.03111` validation and `0.02056` test AUROC. Thus removing
the lie-inducing instruction does not replace the evidence carried by the
organism's original reasoning trace.

Only four validation and one test target answers themselves matched the
label-blind lie-context regex. Excluding them leaves the conclusion unchanged:
original versus targeted-sanitized full-trace AUROC was
`0.91188` versus `0.91672` on validation and
`0.92012` versus `0.92387` on test. The target answers were never rewritten.

The finite source-system audit found 112 distinct originals: 21 were already
safe, 68 retained benign text after unsafe clauses were removed, and 23
required the neutral fallback. No unsafe-pattern survivor remained. All
original-condition prompt hashes exactly match the frozen Apollo artifacts.
Expected batch-shape numerical drift had mean absolute score error
`0.00277/0.00262` and maximum `0.03121` on validation/test. Prompts were
136--6,316 tokens with zero truncation. Nine resumable NDIF sessions scored
3,600 prompts in 841.4 summed seconds (893.9 seconds wall time).
