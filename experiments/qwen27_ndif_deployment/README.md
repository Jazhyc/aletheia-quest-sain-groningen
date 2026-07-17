# Qwen-27B NDIF deployment rehearsal

This directory freezes the exact local-test winning deployment contract without
changing the sole submission notebook. The candidate is base
`Qwen/Qwen3.5-27B`, raw (non-chat-template) D/K/S prompts, 8,000 context
characters with tail truncation, member-major scheduling, greedy 4,096-token
generation, max aggregation, and binary rating cutoff `>= 2`.

Why this is next:

- matched competition validation improves `0.9226 -> 0.9298` BA;
- locked test improves `0.9298 -> 0.9464` BA;
- the frozen four-mode Liars' Bench spectrum improves `0.7788 -> 0.8750` macro
  BA, with every category improved;
- local prompt shape is small: median 439, p95 704, max 1,248 tokens across all
  8,216 rows, with no context truncation or prompt over 2,048 tokens.

The remaining uncertainty is NDIF execution, not scoring logic. Rehearse on one
40-row labeled dataset unit before editing `submission/`:

1. Load `LanguageModel("Qwen/Qwen3.5-27B")` without a LoRA. Tokenize the raw
   strings from `contract.member_major_prompts`; do not apply a chat template.
2. Run full 4,096-token greedy decoding first. Start at batch 4 and one remote
   request per session. Increase only after a complete memory-safe run; Qwen-9B
   batch tiers do not transfer by assumption.
3. Require all 40 binary decisions to match the saved vLLM unit. Report member
   rating differences and parse failures rather than silently changing the
   parser or threshold.
4. In a separate rehearsal, pass exact strings `Rating: 1` through `Rating: 7`
   while retaining the matched text. Transformers 5.12 requires the tokenizer
   argument to construct its stop criterion, so verify remote serialization
   explicitly. Do not approximate with digit EOS tokens.
5. Require the same 40 decisions and a material wall-time reduction. If remote
   string stopping fails, deploy full decoding first; local generated evidence
   already shows stopping is an optimization, not an accuracy requirement.

Only after these checks should a notebook candidate be built. Keep the existing
Phoenix notebook intact until the Qwen-27B batch schedule and stop contract pass
an actual NDIF rehearsal. The leaderboard runs each dataset unit separately
with a 45-minute notebook timeout; the largest local unit is 40 rows / 120
member evaluations, so record per-unit wall time rather than extrapolating only
from the local 800-row vLLM spectrum.
