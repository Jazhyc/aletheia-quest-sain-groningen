# Content-routed epistemic specialist

The factual Phoenix judge has an identifiable blind spot on deceptive knowledge
denials. A general prompt cannot fix it without raising ordinary-validation FPR,
and the local competition splits contain no deceptive knowledge-report rows.
Liars' Bench HP-KR supplies the missing balanced deceptive-denial and honest-
ignorance controls.

This proposal routes only when the observable user request explicitly asks the
assistant to report whether it knows an answer, usually with an `only`, `just`,
or `nothing else` response constraint. The rule uses no source-model identity,
topic, harmful-domain keyword, label, or organism-specific behavior. Routed
rows use the already evaluated `knowledge_report_type` Qwen prompt; all other
rows retain the general Phoenix judge.

Audit results:

- routes all 7,068 HP-KR rows;
- routes zero rows in the other seven Liars' Bench categories (72,349 rows);
- routes zero rows across the complete 8,216-row local competition corpus;
- changes no current local validation/test prediction by construction.

On the frozen 400-row-per-category Liars' Bench signature sample, replacing
only HP-KR predictions raises that category from `0.3861` to `0.8206` BA. Macro
BA across the seven label-balanced categories rises from `0.6811` to `0.7431`
(`+0.0621`), with every non-HP-KR prediction unchanged. The specialist's known
weakness remains honest Mistral/WMDP-bio ignorance, so it should not be invoked
from a broad denial regex.

Reproduce the cached combination with:

```bash
python experiments/epistemic_specialist_router/analyze_cached.py \
  --signature-generations results/blackbox/liars_bench_frozen_judge_signatures_v1/phoenix_v21_adapter.jsonl \
  --specialist-generations results/blackbox/liars_bench_hpkr_epistemic_prompt_sweep_v1/base_qwen/knowledge_report_type.jsonl \
  --output results/blackbox/epistemic_specialist_router_v1/result.json
```

This is a retained transfer candidate despite its intentionally zero local-
validation effect. Before submission integration, benchmark the same routing
with the small Liars-augmented adapter and confirm that the notebook can choose
the base/specialist inference path without adding a second model load for
ordinary datasets. Do not broaden the router from post-hoc inspection of missed
HP-KR phrasings.
