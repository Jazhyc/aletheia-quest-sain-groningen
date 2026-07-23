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

Evaluate the cached learned correction-gate baselines with:

```bash
python experiments/fact_extractor_self_consistency/run_correction_gate.py
```

This requires no model calls. It joins the all-organism self-query cache to the
Qwen and GPT-OSS heavy-judge validation caches, then compares judge-only, raw
OR, a fixed conservative rule, and regularized logistic gates. It reports both
stratified out-of-fold and leave-one-organism-out predictions; the latter is the
primary transfer check because varied-deception is confounded with Qwen.

Run the small claim-centrality and verification-stability pilot with:

```bash
sbatch experiments/fact_extractor_self_consistency/run_claim_centrality_stability_pilot.sh
```

It samples eight label-balanced rows from each Qwen validation dataset (144 rows
across 18 organism/scenario units). Two GPT-OSS
prompt variants extract grounded claims with structural role and materiality;
then each generating organism checks every grounded proposition under two
neutral forced-choice phrasings. The stages run in separate Python processes so
the 120B teacher releases GPU memory before Qwen loads, and each stage writes a
cache under `results/blackbox/claim_centrality_stability_pilot_v1/`.

The controlled full-Qwen follow-up freezes the pilot-selected `impact_first`
extractor and stable high-materiality rule:

```bash
sbatch experiments/fact_extractor_self_consistency/run_claim_centrality_stability_full.sh
```

It evaluates all 720 Qwen validation rows without changing the prompt, verdict
checks, materiality threshold, or judge combination after the pilot.

Run the matched blind atomic-audit control for GPT-OSS-120B and Qwen3.5-9B with:

```bash
sbatch experiments/fact_extractor_self_consistency/run_atomic_audit_controls.sh
```

This freezes the earlier shared-error `blind_atomic` prompt, audits the union of
validation rows predicted negative by either heavy judge, and measures both
false-negative recovery and new false-positive harm. GPT-OSS estimates the
teacher ceiling; Qwen-9B tests whether the same cascade is submission-sized.
