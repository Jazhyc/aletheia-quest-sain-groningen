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
