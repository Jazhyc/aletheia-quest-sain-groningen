# Roadmap

## Black-Box Judge Methods

### Higher Priority

- Test n-shot prompting on the validation split and compare calibration, AUROC, balanced accuracy, parse failures, and runtime against the zero-shot reasoning judge.
- Explore retrieval-augmented example selection for prompting, where each target conversation receives the most relevant labelled dev examples rather than a fixed shot set.
- Use GEPA or a similar prompt-optimization workflow to tune the judge prompt for this competition setup while keeping the resulting prompt general enough to transfer.
- Find ways to make the judge faster that should transfer to NDIF, such as reducing prompt/context length, using fewer generated tokens, batching efficiently, or converting reasoning into a two-pass approach that only scores rating-token logits after free-form reasoning.

### Lower Priority

- Investigate structured final-answer decoding for reasoning judges to eliminate parse failures. Prefer a portable design where the model reasons freely, then a second constrained rating step reads or generates only the final rating token. vLLM-specific guided JSON/regex decoding is useful for local experiments but should not be assumed to transfer to NDIF.
