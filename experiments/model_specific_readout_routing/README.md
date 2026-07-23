# Model-specific readout routing

The competition permits documented per-organism detector parameters while
forbidding different detection logic or behavior-derived prompts. This
experiment therefore keeps the same Phoenix summary and binary readouts for
every source organism and varies only whether the binary recall member receives
aggregation weight zero or one.

Routes are selected on all 6,573 train rows. The conservative rule starts from
the global max ensemble and disables the binary member only when summary-only
improves that organism's macro dataset BA by at least `0.005`. Validation is the
acceptance split; cached local test is inspected once only after the route is
fixed.

```bash
python experiments/model_specific_readout_routing/analyze.py \
  --train-run results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/train_reasoning4000_summary_binary_or_diagnostic_v1 \
  --validation-run results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation_reasoning4000_summary_binary_or_v1 \
  --test-run results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/test_reasoning4000_summary_binary_or_locked_v1 \
  --output results/blackbox/model_specific_readout_routing_v1/result.json
```

Only Gemma base passes the train threshold: summary-only beats max by `0.0094`
on train. It also wins by `0.0250` on its validation dataset and ties on test.
The route raises overall validation BA from `0.9238` to `0.9250`, leaves locked
test BA at `0.9274`, and saves the binary generation pass for Gemma base.

Broader per-organism selection is rejected. Choosing every organism's apparent
train winner raises train BA to `0.9047` but lowers validation BA to `0.9226`;
that is ordinary parameter overfitting. Keep only the thresholded Gemma-base
parameter as a candidate for an eventual two-readout submission. It is not used
by the current single-summary v2.4 notebook.

## Adapter parameter route from training coverage

A second route uses no metric selection. The mixed Phoenix adapter saw
instructed Qwen, Gemma, and Nemotron summaries, whereas the varied-only adapter
saw Qwen examples exclusively. With identical prompts and inference logic, use
the varied-only weights for Qwen units and the mixed weights for Gemma/Nemotron
units. This is a permitted per-organism parameter choice motivated solely by
training coverage.

On cached validation, the route changes one Gemma prediction and raises BA from
`0.9000` to `0.9012`; on the locked local test it changes no prediction and
retains `0.9155` BA. This is modest but directionally stable, so retain it as a
generalization candidate rather than an accuracy claim. Direct deployment is
currently unattractive: the two BF16 adapter weights total about 233 MB before
tokenizer/config files, exceeding the 200 MB package budget. Do not add lossy
adapter compression solely for this one-row local effect.

```bash
python experiments/model_specific_readout_routing/analyze_adapter_route.py \
  --validation-varied results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation/generations.jsonl \
  --validation-mixed results/blackbox/qwen9b_privileged_gptoss120b_summary_adamwlr5e5_v1/validation/generations.jsonl \
  --test-varied results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/test/generations.jsonl \
  --test-mixed results/blackbox/qwen9b_privileged_gptoss120b_summary_adamwlr5e5_v1/test/generations.jsonl \
  --output results/blackbox/model_specific_adapter_routing_v1/result.json
```
