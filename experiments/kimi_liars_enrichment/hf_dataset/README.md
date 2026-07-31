---
license: mit
language:
- en
task_categories:
- text-classification
pretty_name: Phoenix 8.1 Kimi K3 Binary-Logit Distillation Annotations
size_categories:
- 10K<n<100K
tags:
- aletheias-quest
- deception-detection
- knowledge-distillation
- logits
---

# Phoenix 8.1 Kimi K3 distillation annotations

This is the reproducibility artifact for SAIN Groningen's Phoenix Wright 8.1
submission to Aletheia's Quest. Phoenix 8.1 attained **0.9661 mean private
per-dataset AUROC**. The matching rank-16 Qwen3.5-9B adapter is
[`Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`](https://huggingface.co/Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2).

The `train` split has 13,149 rows:

| source | rows |
| --- | ---: |
| Aletheia's Quest public competition training rows | 6,573 |
| Liars' Bench harm-pressure choice | 800 |
| Liars' Bench harm-pressure knowledge report | 1,200 |
| Liars' Bench insider trading | 1,976 |
| Liars' Bench soft trigger | 2,600 |

For every row, Kimi K3 was queried with reasoning disabled through OpenRouter's
Fireworks endpoint. The artifact records the exact literal `0` and `1` token
log-probabilities and their two-class normalization. The student's target was

```text
soft_target_deceptive = exp(logprob_1) / (exp(logprob_0) + exp(logprob_1))
```

Phoenix 8.1 trained `Qwen/Qwen3.5-9B` for two epochs with a rank-16/alpha-32
LoRA, AdamW at `5e-5`, effective batch size 32, and binary soft-target BCE at
the direct `Prediction:` boundary. It did not use generated teacher reasoning,
hard-label loss, completion loss, or pairwise loss.

## MIT scope and source text

This repository is intentionally MIT-only. The Parquet contains our Kimi
annotations, route metadata, source identifiers/revisions, and prompt hashes.
It does **not** redistribute source conversations, source labels, or rendered
prompts. Liars' Bench author-created data is CC BY 4.0 and may include additional
third-party terms; those materials cannot be relicensed as MIT.

`render_prompts.py` fetches the pinned upstream rows and produces a local,
train-ready Parquet with `label`, `student_prompt`, and `teacher_prompt`. That
generated file remains governed by the source datasets' licenses and should not
be treated as part of this MIT artifact.

```bash
pip install -r requirements.txt
python render_prompts.py \
  --annotations data/train-00000-of-00001.parquet \
  --output phoenix_v8_1_train_with_prompts.parquet
```

The script validates every materialized prompt against its recorded SHA-256.
Pass `--include-messages` only if the local training pipeline also needs the
normalized source conversation.

## Prompt routing

The renderer preserves the exact training/query asymmetry:

| category | Kimi teacher prompt | student prompt |
| --- | --- | --- |
| competition | original Kimi Truth Value Guard | Phoenix ordinary |
| harm-pressure choice | semantic forced-choice factual audit | Phoenix ordinary |
| harm-pressure knowledge report | epistemic report audit | epistemic report audit |
| insider trading | Phoenix ordinary | action-report audit |
| soft trigger | Phoenix ordinary | Phoenix ordinary |

Teacher and student context caps/truncation modes are stored per row. The
bundled `prompt_templates.json` and renderer reproduce the exact observable-text
prompts used to query Kimi and train the student.

## Main columns

- `source_dataset`, `source_config`, `source_revision`, `source_index`: pinned
  upstream row reference.
- `teacher_logprob_0`, `teacher_logprob_1`: raw Kimi literal-token log-probs.
- `teacher_probability_0`, `teacher_probability_1`: normalized two-class
  probabilities.
- `soft_target_deceptive`: the BCE target, identical to
  `teacher_probability_1`.
- `student_prompt_kind`, `teacher_prompt_kind`: template selectors.
- `*_context_chars`, `*_context_truncation`: exact renderer settings.
- `student_prompt_sha256`, `teacher_prompt_sha256`: byte-level replay checks.

`manifest.json` records source revisions, template hashes, cache provenance,
row counts, and the Parquet digest.

## Sources and citation

- [Aletheia's Quest competition repository](https://github.com/ndif-team/Aletheias-Quest-Competition)
- [Liars' Bench dataset](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)
- Kieron Kretschmar, Walter Laurito, Sharan Maiya, and Samuel Marks. *Liars'
  Bench: Evaluating Lie Detectors for Language Models*. arXiv:2511.16035, 2025.

Please also cite the Aletheia's Quest organizers and the Kimi K3 model/provider
when using these annotations.
