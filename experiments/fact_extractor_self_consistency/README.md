# Fact extractor prompt sweep

Run the blind GPT-OSS validation sweep with:

```bash
sbatch experiments/fact_extractor_self_consistency/run_prompt_sweep.sh
```

The shell template runs the current v2 assessed-claim conditions. To reproduce
the original three-condition sweep, invoke `run_prompt_sweep.py --iteration v1`
in an equivalent GPU allocation. Results are written beneath
`results/blackbox/gpt_oss_120b_atomic_claim_prompt_sweep_<iteration>/validation/`
and are git-ignored. See
[`docs/fact_extractor_self_consistency/README.md`](../../docs/fact_extractor_self_consistency/README.md)
for the contracts and findings.

Run the Qwen whole-output organism self-query control with:

```bash
sbatch experiments/fact_extractor_self_consistency/run_qwen_whole_output_self_query.sh
```

This evaluates only Qwen validation datasets and applies each row's generating
LoRA. The wrapper sources the git-ignored `.env` when present because the
organism adapters are gated.

Run the all-family NDIF version with:

```bash
bash experiments/fact_extractor_self_consistency/run_ndif_whole_output_self_query.sh
```

It groups rows by generating model/LoRA, uses batches of 16 for Qwen, 8 for
Gemma, and 1 for Nemotron, and caches each completed remote group beneath the
ignored result directory. Re-running resumes rather than repeating successful
NDIF sessions.

On the current NDIF backend, Qwen uses generated verdicts. Gemma's hosted PEFT
generation fails with a sequence/batch shape error, while Nemotron generation
OOMs even at low batch sizes. Since both tokenizers encode uppercase `TRUE` and
`FALSE` as single tokens, the runner uses their next-token forced choice instead.
Nemotron also uses a compact prompt to fit its batch-1 forward pass. Each output
records `decision_mode` and `prompt_variant`; do not silently mix these modes in
backend-equivalence claims.
