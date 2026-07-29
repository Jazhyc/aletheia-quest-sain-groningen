# Qwen3.5 OpenRouter Truth Value Guard benchmark

This development-only benchmark evaluates
`qwen/qwen3.5-27b` or `qwen/qwen3.5-397b-a17b` on the public 822-row
validation split. It supports the frozen no-thinking Truth Value Guard prompt
and an `explicit_digits` ablation that adds exactly
`Use 0 for not deceptive and 1 for deceptive.` The model greedily generates
`Prediction:<0 or 1>`, and the runner reads the binary logits at the generated
label position. This is slightly different from locally forcing the
`Prediction:` prefix, because OpenRouter's chat providers do not support
continuing an assistant prefill.

The continuous score is:

```text
P(1) / (P(0) + P(1))
```

Successful API calls are appended to `generations.jsonl`, so interrupted runs
resume without paying for completed rows. The API key is read from
`OPENROUTER_API_KEY` or the repository's untracked `.env`; it is never written
to an artifact.

One-row protocol smoke:

```bash
.venv/bin/python -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg \
  --limit 1 --method qwen35_397b_openrouter_tvg_smoke \
  --provider-only Alibaba --no-allow-fallbacks
```

Full validation run:

```bash
.venv/bin/python -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg \
  --provider-only Alibaba --no-allow-fallbacks
```

Use `--prompt-variant explicit_digits` for the clarified contract and
`--model qwen/qwen3.5-27b` for the smaller teacher. Give each condition a
separate `--method`, because resumable cache validation rejects prompt changes
within an existing result directory.

The selected regular-397B clarified-prompt distillation cache can be reproduced
end to end with:

```bash
bash \
  experiments/privileged_information_distillation/run_qwen397_openrouter_explicit_soft_teacher.sh
```

That command filters the train split to the 2,880 varied-deception rows,
exports Lambda-cache-compatible binary probabilities, rebuilds public student
prompts with the byte-exact teacher prompt, and writes a SHA-256 manifest.
The resulting cache remains separate from the historical FP8 Lambda cache.

OpenRouter is for development supervision and benchmarking only. Competition
submissions cannot make external API calls at inference time.
