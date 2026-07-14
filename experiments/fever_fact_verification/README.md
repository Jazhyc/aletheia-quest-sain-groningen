# FEVER fact-verification experiment

The method decision, current regular-scale results, and literature rationale are recorded in
`docs/fever_fact_verification/README.md`. This directory contains a resumable
Wikipedia retriever, a resumable title-to-full-page cache builder, claim-specific
document sentence selection, a GPU FEVER-NLI scorer, deterministic aggregation,
a matched shuffled-evidence evaluation, and unit tests.

Retrieve all grounded varied-validation claims:

```bash
.venv/bin/python experiments/fever_fact_verification/retrieve_wikipedia.py \
  --claims results/blackbox/gpt_oss_120b_atomic_claim_prompt_sweep_v2/validation/generations.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_retrieval.jsonl \
  --dataset-name-contains varied-deception
```

Score real and matched shuffled evidence on a local GPU:

```bash
sbatch experiments/fever_fact_verification/run_verify.sh \
  --retrieval results/blackbox/fever_fact_verification_v1/varied_validation_retrieval.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_verification.jsonl \
  --with-shuffled-control
```

The encoder-only stack also has a CPU Slurm wrapper for queue-independent and
deployment-oriented timing:

```bash
sbatch experiments/fever_fact_verification/run_verify_cpu.sh \
  --retrieval results/blackbox/fever_fact_verification_v1/varied_validation_retrieval.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_verification_cpu.jsonl \
  --with-shuffled-control
```

Summarize the frozen run:

```bash
.venv/bin/python experiments/fever_fact_verification/evaluate.py \
  --input results/blackbox/fever_fact_verification_v1/varied_validation_verification.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_report.json
```

The retriever appends one JSONL row after every claim and safely resumes an
existing output. Delete a smoke artifact rather than reusing it for a full run
with the same filename.

For the document-level path, fetch each unique top-ranked page only once and
then select evidence for every atomic claim locally:

```bash
.venv/bin/python experiments/fever_fact_verification/fetch_page_cache.py \
  --retrieval results/blackbox/fever_fact_verification_v1/varied_validation_question_retrieval.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_pages.jsonl \
  --max-page-rank 0

.venv/bin/python experiments/fever_fact_verification/build_from_page_cache.py \
  --retrieval results/blackbox/fever_fact_verification_v1/varied_validation_question_retrieval.jsonl \
  --page-cache results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_pages.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_retrieval.jsonl \
  --max-page-rank 0
```

MediaWiki permits only one whole-article extract per API request. The page-cache
builder enforces that limit, backs off on HTTP 429 responses, and resumes rows
that did not fail. `--max-page-rank 2` is the frozen top-3 expansion after the
top-1 experiment.

The gold-FEVER ceiling used this explicit label order:

```bash
sbatch experiments/fever_fact_verification/run_verify.sh \
  --retrieval results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_retrieval.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_verification.jsonl \
  --with-shuffled-control \
  --model pepa/deberta-v3-base-fever \
  --label-order entailment,contradiction,neutral \
  --rerank-top-n 12
```

That checkpoint is a diagnostic ceiling only because its model card does not
state a usable license. Keep the MIT-licensed MoritzLaurer verifier as the
portable lower bound. Neither verifier verdict is a deception label.

Build a complete teacher-compatible cache after verification. `--include-nei`
retains the best retrieved sentences even when the auxiliary NLI model abstains;
the noisy relation itself is hidden from rendered evidence unless explicitly
requested. `--row-universe` adds empty evidence for every inactive row, which is
required by the privileged teacher cache contract.

```bash
.venv/bin/python experiments/fever_fact_verification/build_teacher_reference.py \
  --verification results/blackbox/fever_fact_verification_v1/varied_validation_top1_full_verification.jsonl \
  --output results/blackbox/fever_fact_verification_v1/varied_validation_top1_teacher_reference.jsonl \
  --row-universe results/blackbox/qwen9b_pid_wikidata_matched_reader_v1/validation_fever_question_evidence_v1/empty/generations.jsonl \
  --evidence-per-claim 1 \
  --include-nei
```

The builder caps pathological list-like source sentences at 1,200 characters
and the complete reference block at 5,000 characters per response. This keeps
the evidence-conditioned reader within its 4,096-token evaluation context while
retaining whole claim/source records; both limits are configurable.

The resulting file is directly accepted through the existing
`teacher.retrieval_cache=...` Hydra override in
`experiments/privileged_information_distillation/run_teacher.sh`.
