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
