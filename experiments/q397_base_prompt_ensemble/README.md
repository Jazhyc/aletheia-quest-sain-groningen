# Q397 plus explicit-prompt base-Qwen ensemble

This experiment combines two heterogeneous output-only score streams:

- 60% optimized Q397 log odds under its exact trained Phoenix 4.0 prompt;
- 40% base-Qwen log odds under the previously selected explicit-digit prompt.

The base prompt was the strongest standalone condition in the frozen P81
verbalizer sweep. The 60/40 weight is frozen from validation and is applied
uniformly; no family, organism, reasoning field, or row-level router is used.
The required binary prediction remains the unblended Q397 decision.

Cached paired validation improves macro AUROC from `0.95393` to `0.95554`
(`+0.00161`). Instructed remains `0.99813`; varied improves from `0.89500` to
`0.89875`. Unit wins/ties/losses are `4/12/5`, the worst unit loses `0.020`,
and a dataset-unit bootstrap interval crosses zero, so this receives exactly
one frozen local-test confirmation:

The delayed local-GPU confirmation was superseded by the user-authorized
optimized NDIF path:

```bash
python -m experiments.q397_base_prompt_ensemble.run_ndif
python -m experiments.q397_base_prompt_ensemble.analyze
```

This queries only the missing base-Qwen stream, using one remote session,
`logits_to_keep=1`, and Phoenix's frozen length-sorted `48/32/16` tiers. The
already-cached Q397 test vector is not rerun.

Deployment would require one additional direct base-Qwen pass, but no
generation and no source reasoning.
