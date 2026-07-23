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

python experiments/compact_wikipedia_index/train_reranker.py \
  --index results/blackbox/compact_wikipedia_train_index_v1/train_pages.sqlite \
  --audits results/blackbox/fever_fact_verification_train_v1/varied_train_selective_initial_audit.jsonl \
  --output results/blackbox/compact_wikipedia_train_index_v2/reranker.json \
  --report results/blackbox/compact_wikipedia_train_index_v2/train_reranker_report.json \
  --limit 2000

sbatch experiments/compact_wikipedia_index/run_claim_extractor.sh

python experiments/compact_wikipedia_index/build_learned_cache.py \
  --index results/blackbox/compact_wikipedia_train_index_v1/train_pages.sqlite \
  --queries results/blackbox/compact_wikipedia_train_index_v2/validation_qwen_claims.jsonl \
  --output results/blackbox/compact_wikipedia_train_index_v2/validation_learned_cache.jsonl \
  --threshold 1.285

sbatch experiments/compact_wikipedia_index/run_learned_reader.sh
```

## Decision

Do not package or integrate this index. It establishes that train-question
pages provide far better corpus coverage than the broad Wikidata cards, and
that relevant evidence affects the reader differently from shuffled evidence.
However, raw inference-visible retrieval regresses against no evidence, and
even teacher-extracted atomic queries produce only a marginal validation gain.
Any follow-up needs a train-frozen claim extractor/reranker that improves the
empty-evidence control—not a larger corpus or validation-error tuning.

## Deployable claim-extraction follow-up

The next frozen attempt tested two improvements without using validation labels
for selection:

1. A compact standardized logistic reranker used proposition/source overlap,
   question overlap, title and number matching, reciprocal FTS ranks, sentence
   position, and length similarity. It was fit on 1,618 claims and calibrated
   on 382 exact-question-grouped held-out training claims.
2. Base `Qwen/Qwen3.5-9B` in no-thinking mode rewrote the visible answer into at
   most six standalone propositions. Retrieval retained the original lexical
   score and added one adjacent source sentence around each hit.

The learned reranker was rejected before validation. Across all calibration
claims its top candidate exactly matched the audited sentence on 35/382 rows;
among the 84 claims where that source was present in its candidate pool, recall
was 41.7%. A high-precision threshold emitted only two claims. Extra linear
ranking capacity therefore did not provide a usable abstention boundary.

The deployable extractor parsed 360/360 varied-validation rows and emitted
1,780 claims across 347 rows. At the original training-selected `1.285`
threshold, the cache contains 79 passages on 62 rows; 27 active rows and 30
passages exactly overlap the independent audited cache. This is better than raw
queries (10/43 rows) and the non-deployable GPT-OSS atomic ceiling (21/54 rows).
The downstream reader nevertheless regresses:

| evidence | varied BA | recall | FPR | paired versus real |
| --- | ---: | ---: | ---: | --- |
| empty | **`0.8139`** | `0.7222` | `0.0944` | real: 4 fixes / 8 breaks |
| Qwen claims + real windows | `0.8056` | `0.7222` | `0.1111` | — |
| count-matched shuffled windows | `0.7972` | `0.6944` | `0.1000` | real: 8 fixes / 6 breaks |

A final stricter threshold was selected from the existing training calibration,
not validation: `1.5` emits 32 training claims with 40.6% exact-source precision
and an 87.5% audited-claim rate. It yields 25 validation passages on 23 rows,
with 11 exact audited-row matches. This removes the regression but not enough
errors to improve BA:

| high-precision evidence | varied BA | recall | FPR | paired versus real |
| --- | ---: | ---: | ---: | --- |
| empty | **`0.8194`** | `0.7167` | `0.0778` | real: 2 fixes / 2 breaks |
| Qwen claims + real windows | **`0.8194`** | `0.7167` | `0.0778` | — |
| count-matched shuffled windows | `0.8139` | `0.7056` | `0.0778` | real: 3 fixes / 1 break |

Jobs `30215108`, `30215199`, and `30215268` completed in 9m54s, 3m45s,
and 3m42s. Claim generation itself took roughly 20 seconds; cold A100
compilation dominated extractor wall time. No local-test row was evaluated.

Decision: retain the claim extractor as a useful retrieval component and the
linear reranker as a negative capacity check, but do not package either. Better
retrieval relevance alone does not make this evidence-naive reader reliable:
ordinary factual corrections and incomplete windows still shift honest rows.
The next independent-data attempt should train the consumer on useful,
irrelevant, insufficient, conflicting, and ordinary-error evidence. Do not
sweep more thresholds on these validation outcomes.

## Packaging caveat

The experiment does not place the database in `submission/`. Wikipedia text is
CC BY-SA rather than MIT/CC0, so a submission package would need the applicable
license text, attribution/source metadata, and confirmation that the
competition's publication contract accepts a separately licensed data asset.
The database already retains canonical titles and source URLs for that purpose.
