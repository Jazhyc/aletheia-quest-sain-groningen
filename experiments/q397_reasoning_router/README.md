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

## Frozen baseline result

On validation, direct scoring reached `0.95345` macro AUROC and generated
reasoning reached `0.94536`. The frozen 60% direct / 40% reasoning log-odds
blend improved to `0.95714` (`+0.00369`), with `5/12/4` unit
wins/ties/losses. Nested hard routing selected direct in every held-unit fold;
a nested learned soft router also failed. The fixed blend was the only
promotion candidate.

The one frozen local-test confirmation rejected that blend overall. Direct
scored `0.95929`; post-reasoning scored `0.93101`; and the blend scored
`0.95911` (`-0.00018`). The blend improved varied deception
`0.91056 -> 0.91472` but reduced instructed deception
`0.99583 -> 0.99240`; unit wins/ties/losses were `5/9/7`. Generation took
`534.3s` on validation and `682.3s` on test, versus `45--47s` for the paired
direct/post scoring pass.

Do not deploy the borrowed summary prompt or retune its weight on test. A
three-prompt follow-up keeps the 60/40 weight fixed, compares prompt wording
on validation only, caps summaries at 192 tokens, and batches all conditions
inside one local-vLLM model load:

```bash
sbatch experiments/q397_reasoning_router/run_prompt_sweep.sh
python -m experiments.q397_reasoning_router.analyze_prompt_sweep
```

When Slurm cannot start promptly, run the equivalent frozen sweep through
optimized NDIF sessions:

```bash
python experiments/q397_reasoning_router/run_prompt_sweep_ndif.py
```

## Validation-only prompt sweep

Slurm job `30361556` had an estimated start roughly three hours after
submission, so it was cancelled before starting and the frozen sweep was run
through NDIF. The 2,466 deterministic generations were packed into 79 batches
across three sessions; generation took `974.0s`. One combined scoring session
took `98.7s`.

The compact `summary_baseline` remained the winner. Its post-reasoning margin
scored `0.94863` macro AUROC, and the fixed 60% direct / 40% reasoning
log-odds blend scored `0.95827`, versus `0.95345` direct (`+0.00482`).
Instructed AUROC moved `0.99792 -> 0.99771`, while varied improved
`0.89417 -> 0.90569`; unit wins/ties/losses were `6/12/3`. The blend produced
706 unique scores over 822 rows.

The more prescriptive prompts failed. `claim_check` reasoning/blend scored
`0.93506/0.95071`, and `balanced_audit` scored `0.93810/0.95327`. Detailed
analysis appears to suppress useful learned behavior rather than improve the
readout. This sweep used validation only and does not overturn the earlier
frozen-test rejection of the 512-token borrowed-summary blend. Treat the
192-token baseline as an unconfirmed leaderboard probe, not a validated
replacement for direct scoring.
