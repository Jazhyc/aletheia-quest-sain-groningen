# Qwen3.5-397B OpenRouter Truth Value Guard benchmark

This development-only benchmark evaluates
`qwen/qwen3.5-397b-a17b` on the public 822-row validation split. It renders the
same Qwen no-thinking chat template and direct `Prediction:` boundary used by
the local Qwen3.5-27B binary-logit control, then requests one-token logprobs
from OpenRouter's text-completions endpoint.

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
  --limit 1 --method qwen35_397b_openrouter_tvg_smoke
```

Full validation run:

```bash
.venv/bin/python -m experiments.openrouter_qwen397_tvg.run_openrouter_tvg
```

OpenRouter is for development supervision and benchmarking only. Competition
submissions cannot make external API calls at inference time.
