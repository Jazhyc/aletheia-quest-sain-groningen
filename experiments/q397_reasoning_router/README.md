# Q397 generated-reasoning router

This experiment tests whether generated judge reasoning is complementary to
the optimized two-epoch Q397 direct margin, even if post-reasoning AUROC is
worse on average.

The NDIF pass uses the exact optimized Q397 adapter. It scores the ordinary
binary Phoenix boundary directly, separately generates a deterministic
`<reasoning_summary>`, removes the generated terminal digit, and scores
normalized literal `0|1` logits at that boundary. Source reasoning is never
used.

```bash
python experiments/q397_reasoning_router/run_ndif.py
python -m experiments.q397_reasoning_router.analyze
```

Analysis reports direct and post-reasoning AUROC, a fixed equal-log-odds blend,
pairwise ranking recoveries/harms, and three nested
leave-one-dataset-unit-out conditions:

- a selected fixed log-odds blend;
- a hard router based only on deployable score/reply-shape features; and
- a regularized soft router using the two margins and deployable disagreement
  features.

No organism identity or source reasoning is available to any router.

If validation selects a fixed blend, run exactly one frozen local-test
confirmation:

```bash
python experiments/q397_reasoning_router/run_ndif.py --split test
python -m experiments.q397_reasoning_router.analyze \
  --test-input results/blackbox/q397_reasoning_router_test_v1/scores.jsonl
```
