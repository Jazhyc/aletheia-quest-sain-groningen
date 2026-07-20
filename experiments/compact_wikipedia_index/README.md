# Compact train-page Wikipedia index

This experiment asks whether a submission-sized Wikipedia sentence index can
approximate the selective offline FEVER cache without contacting Wikipedia at
inference time. It intentionally separates corpus/ranking measurements from the
downstream deception-label check.

## Frozen design

- Corpus: the 3,061 full-page responses previously fetched from the top-three
  Wikipedia titles for public varied-training questions. No validation query or
  label selects a page.
- Index: 3,060 unique pages and 737,996 normalized sentences in an external-
  content SQLite FTS5 database. The database is 279,371,776 bytes and about 63
  MB under gzip `-9`.
- Retrieval: permissive FTS candidate generation followed by deterministic
  entity/title, lexical, and exact-number scoring.
- Calibration: a deterministic 2,000-claim training sample and the existing
  label-blind GPT-OSS selective-audit references. The frozen `1.285` threshold
  is the lowest-score operating point with at least 100 emissions and a 70%
  audited-claim rate. It emits 146 training claims; 35 exactly reproduce the
  audited source sentence and 103 belong to claims for which the auditor found
  some decisive passage.
- Deployable query: split only the inference-visible assistant output into at
  most six sentences and combine each with the user question. Return at most
  two passages per row. The `atomic` condition instead uses GPT-OSS-extracted
  propositions and is explicitly a non-deployable extraction ceiling.
- Controls: evidence-free prompts and count-matched, cross-dataset shuffled
  passages. Inactive rows reuse the exact empty generation.

No local-test row is used. Validation deception labels are consulted only in
the final frozen reader comparison, not for index construction, scoring, or
threshold calibration.

## Retrieval diagnostics

When queried by the independently audited validation propositions, the
train-only index recovers the exact audited source on 112/417 claims at top 1,
197/417 at top 3, 242/417 at top 5, and 285/417 at top 10. The top-10 matches
cover 172/214 rows with known selective evidence. Thus corpus coverage is
already substantial; high-precision ranking and abstention are the larger
bottleneck.

At the frozen threshold, raw inference-visible queries emit 44 passages across
43/360 varied-validation rows. Ten rows exactly reproduce a passage in the
independent selective cache. Atomic propositions emit 64 passages across 54
rows, with 21 exact audited-row matches. Exact overlap is a lower bound because
an alternative sentence can settle the same proposition.

The deployable raw-query reader does not pass the empty-evidence control. On
varied validation it scores `0.8028` BA (`0.7000` recall, `0.0944` FPR), versus
`0.8056` for empty evidence and `0.8000` for count-matched shuffled evidence.
Real evidence changes nine predictions relative to empty evidence, fixing four
and breaking five. The small real-over-shuffled advantage shows that retrieval
contains signal, but the current raw query and ranking do not turn it into a
net gain.

The non-deployable atomic-query ceiling is more encouraging but still misses
the predeclared `+0.005` empty-control gate. Its varied BA is `0.8083`
(`0.7222` recall, `0.1056` FPR), versus `0.8056` for empty evidence and `0.7972`
for shuffled evidence. Real evidence fixes four and breaks three predictions
relative to empty, and fixes six while breaking two relative to shuffled. This
isolates claim formulation as part of the bottleneck, but the one-row net gain
is too small to justify a deployable extractor or packaging the index.

## Commands

```bash
python experiments/compact_wikipedia_index/build_index.py \
  --pages results/blackbox/fever_fact_verification_train_v1/varied_train_top3_full_pages.jsonl \
  --output results/blackbox/compact_wikipedia_train_index_v1/train_pages.sqlite

python experiments/compact_wikipedia_index/evaluate_recovery.py \
  --index results/blackbox/compact_wikipedia_train_index_v1/train_pages.sqlite \
  --references results/blackbox/fever_fact_verification_v1/varied_validation_selective_expanded_window_union_reference_v1.jsonl \
  --output results/blackbox/compact_wikipedia_train_index_v1/validation_recovery.json

sbatch experiments/compact_wikipedia_index/run_reader.sh raw
sbatch experiments/compact_wikipedia_index/run_reader.sh atomic
```

## Decision

Do not package or integrate this index. It establishes that train-question
pages provide far better corpus coverage than the broad Wikidata cards, and
that relevant evidence affects the reader differently from shuffled evidence.
However, raw inference-visible retrieval regresses against no evidence, and
even teacher-extracted atomic queries produce only a marginal validation gain.
Any follow-up needs a train-frozen claim extractor/reranker that improves the
empty-evidence control—not a larger corpus or validation-error tuning.

## Packaging caveat

The experiment does not place the database in `submission/`. Wikipedia text is
CC BY-SA rather than MIT/CC0, so a submission package would need the applicable
license text, attribution/source metadata, and confirmation that the
competition's publication contract accepts a separately licensed data asset.
The database already retains canonical titles and source URLs for that purpose.
