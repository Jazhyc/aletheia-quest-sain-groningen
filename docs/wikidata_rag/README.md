# Compact Wikidata Retrieval

This directory records the decisions and frozen measurements for the compact,
submission-compatible Wikidata retrieval experiments. Runtime code, builders,
tests, and command examples live in `experiments/wikidata_rag/`.

## Selected programmatic method

The selected method uses two local SQLite databases:

1. the expanded entity database for alias-rich entity linking;
2. a compact integer-coded relation database for forward, reverse, qualified,
   and constrained two-hop fact lookup.

The query planner is deterministic and CPU-only. It extracts named subjects and
answer values, maps explicit question wording to Wikidata predicates, filters
temporally incompatible facts, and emits evidence only when the relation can be
classified as support or counterevidence. It abstains on unresolved relations.

The final relation database contains:

- 233,239 labeled nodes;
- 512,162 facts across 33 predicates;
- 55,210 start-year and 40,072 end-year qualifiers;
- 37,249 point-year qualifiers;
- 334 `for work` qualifiers.

The database is 39,448,576 bytes as SQLite and 17,118,369 bytes compressed.
Shipping it with the expanded entity linker projects to 195,964,375 bytes with
the current submission tree, leaving approximately 4.0 MB below the decimal
200 MB package limit.

## Programmatic ablations

| Condition | Covered train rows | Novel teacher-fact hits | Conditional evidence precision |
| --- | ---: | ---: | ---: |
| direct structured lookup | 87 | 26 | 0.204 |
| bidirectional lookup | 91 | 29 | 0.212 |
| prior selected claim gate | 132 | 38 | 0.414 |
| selected strict relation method | 130 | 48 | 0.538 |

The final metrics use 2,877 parsed varied-training teacher summaries from 2,880
source rows. The selected method covers 15/360 varied-validation rows, versus
10/360 for the prior programmatic gate. Its five additions were manually
relevant and no prior covered row was lost:

- Milan for Derby della Madonnina;
- *Marty* for Ernest Borgnine's qualified award;
- Laos for Vientiane;
- Portugal for Benfica and Porto;
- Boston for the Charles River/Fenway Park/USS Constitution question.

## What worked

- Reverse lookup produced a small but repeatable gain.
- Uncapped award histories plus `for work` qualifiers recovered the poisoned
  Borgnine film slot.
- The constrained person→office→jurisdiction→monarch path correctly resolves
  Neville Chamberlain's premiership to George VI and rejects George V.
- Mechanically generated routing tests exposed seven missing route families;
  the selected rules now pass all 32 generated cases.
- Requiring a support/counterevidence verdict removed uncertain, topically
  related facts and materially improved conditional precision.

## What did not work

- Broad multi-entity lookup raised coverage but followed incidental place names
  and ambiguous work titles.
- The split-half, label-free relation-mapping search found no trustworthy new
  cross-relation mapping. Apparent additions had poor evidence precision.
- Long-history enrichment alone added little aggregate coverage; its value was
  concentrated in qualified award and office-history slots.

## Frozen artifacts

Ignored experiment artifacts are under
`results/blackbox/wikidata_rag_relations_v1/`:

- `relations_selected.sqlite` and `selected_build_report.json`;
- `train_selected_v5.jsonl` and `train_selected_v5_report.json`;
- `validation_selected_v5.jsonl` and the validation comparison report;
- generated-routing and split-half mapping-search reports;
- resumable enriched relation cards and labels.

The retrieval quality is the strongest programmatic result so far, but coverage
remains only 4.5% of varied training. Do not launch another teacher/student run
until another high-confidence relation family adds material coverage without
undoing the precision gain.

## Remaining-space retrieval ablations

The projected package has only about 4.0 MB of headroom, so two learned
fallbacks were tested against grouped question holdouts. Oracle labels score
only fact-value payloads; entity and predicate titles are excluded because they
otherwise manufacture relevance on identity questions.

The full entity cards contain a teacher-summary lexical hit on 694/2,877 rows,
versus 141 for current structured candidates. This is an optimistic oracle
ceiling, not achievable retrieval: manual audit found many topical or
hierarchically related values that do not answer the requested slot.

- A 60,084-byte hashed question/answer/fact ranker found no positive grouped
  test retrieval at its calibration-selected operating point.
- A 422,192-byte hashed question-to-predicate router reached only 4.1% top-1
  and 19.3% top-3 accuracy on its grouped test partition. On 133 uncovered
  oracle rows its top three predicates contained the oracle predicate 24 times,
  versus 12 for the deterministic rules. This is too imprecise to expose facts
  to the teacher, especially because the oracle itself is optimistic.
- A strict full-card fallback reusing the shipped entity database added 13
  semantically plausible training rows after audit, but zero rows beyond the
  selected method's 15/360 validation rows. Most train additions were repeated
  question templates, so the apparent gain did not generalize.

The optional `--card-fallback` cache-builder flag is retained for reproducible
ablation only and is not selected. These results show that package size is not
the current bottleneck: a sub-megabyte model fits easily, but available labels
do not reliably distinguish a fact that settles the requested claim from a
merely related fact. Keep the existing strict relation method and preserve the
remaining package headroom. A learned retriever becomes worthwhile only with
clean claim-to-fact supervision or substantially more diverse grouped training
questions; teacher-summary token overlap is not sufficient supervision.

## GPT-OSS-supervised compact retrieval

A follow-up replaced lexical teacher-summary overlap with direct GPT-OSS-120B
supervision. For each public varied-deception row, GPT-OSS saw at most 12 facts
from the frozen expanded entity database and labeled each as `decisive`,
`relevant_insufficient`, or `irrelevant`. `decisive` deliberately ignores
support/contradiction polarity: the fact must independently settle the direct
answer either way. Removing polarity and brittle quote extraction raised the
64-row smoke parse rate from 49/64 to 62/64 and eliminated the teacher's
support/contradiction label inversions.

Full local job `30136046` labeled 2,404 training and 296 validation rows in
27m36s. Of 2,700 rows, 2,618 parsed (97.0%). Repeated answer variants provide a
useful reliability check: decisive/non-decisive agreement for overlapping facts
was 98.9% on training and 100% on validation, although exact three-way agreement
was only 82.4% and 85.2%. GPT-OSS found a decisive candidate on 335/2,336 valid
training rows and 50/282 valid validation rows; 272 training and 40 validation
rows were outside the selected rule retriever's coverage.

Several negative controls matter:

- A mechanically grounded 2,000-row corpus was nearly perfectly learnable in
  domain but did not transfer to real questions.
- Adding 1,434 GPT-OSS paraphrases of 500 grounded rows still produced only
  0.451 grouped real-test candidate AUROC and no 80%-precision emission point.
- A mined high-precision phrase→predicate rule table abstained everywhere.
- Predicate-interaction and character-relation feature expansions did not beat
  the generic 32,768-feature hashed linear model; the character model overfit.

The generic model is only about 60 KB. On grouped public-training holdouts its
candidate AUROC was 0.881 on calibration and 0.893 on internal test. A first
validation report appeared excellent—11/11 emitted facts were teacher-labeled
decisive—but every emission used an exact question already present in training.
After separating exact question groups, the model still ranked very well on 51
valid novel-question rows: candidate AUROC 0.975 and the decisive fact ranked
first on all 8 rows that contained one. Its calibration-selected absolute-score
threshold emitted zero novel rows, however, and a top-1/runner-up margin did not
produce a five-row 80%-precision calibration point. Always emitting top-1 would
have only 8/51 decisive rows, so this is not a submission-safe retriever by
itself.

Manual audit also found a systematic teacher limit. GPT-OSS called `composer:
Pyotr Ilyich Tchaikovsky` decisive for “What Christmas ballet was written by
Tchaikovsky?”, even though that reverse clue does not uniquely establish the
work. Thus teacher labels are far cleaner than lexical overlap but are not a
perfect logical entailment oracle.

Artifacts are under
`results/blackbox/wikidata_rag_gptoss_supervision_v1/`. Keep the 60 KB model as
a ranking diagnostic, not a selected abstaining retriever. Its remaining
plausible use is supplying one noisy top-ranked fact to an evidence-aware
teacher or judge that can explicitly ignore irrelevant material; evaluate that
with exact-question-group reporting before any new student training.

Frozen downstream job `30136343` rejected that last use. It gave the selected
varied-only reasoning-summary student one top-ranked fact on every validation
row and compared empty, real, and size-matched shuffled reference conditions.
On varied rows, binary BA fell from 0.8083 empty to 0.8056 real: recall rose
from 0.6667 to 0.6833, but FPR rose from 0.0500 to 0.0722. Real evidence made
four false-negative recoveries, one deceptive regression, five honest harms,
and two honest recoveries. Shuffled evidence also scored 0.8056 BA. On the 65
questions absent from training, real evidence scored 0.8929 BA versus 0.8871
empty, but shuffled evidence scored 0.9013; the small change is therefore not
evidence-specific and is affected by 1--3 parse errors per condition. Do not
regenerate teacher traces or train another student from always-on top-1 facts.
The remaining learned alternatives are not promising without more diverse
question groups or a consumer whose evidence use transfers across those groups.

## Counterfactual utility supervision and abstention

The final learned follow-up implemented the end-task criterion used by methods
such as RECOMP and REPLUG rather than asking another model for semantic
relevance. For every bounded fact, the frozen varied-only student was scored
twice with constrained `Prediction:0|1` logits: once with the explicit
no-evidence reference and once with the fact. Utility is the increase in log
probability of the correct deception label. Each row also received a
deterministically rotated unrelated fact; subtracting that effect gives a
control-adjusted utility target and an explicit `NO_EVIDENCE` action.

Job `30137268` scored 33,706 training and 4,202 validation candidates, plus
6,480 empty/shuffled controls, in 14m59s. The candidate set has real oracle
headroom. On validation, choosing the best fact with label knowledge raises the
direct-margin BA from 0.7944 to 0.8389 and AUROC from 0.8531 to 0.9401. This is
not a usable policy: GPT-OSS semantic decisiveness is only weakly related to
student utility. The 63 validation `decisive` candidates have mean raw utility
-0.0077, versus -0.0022 for 2,063 `irrelevant` candidates and +0.0153 for 971
`relevant_insufficient` candidates. Much of the apparent signal is therefore
reader-specific prompt perturbation rather than factual sufficiency.

Compact hashed models tested raw and shuffled-adjusted log utility, binary
rescue/harm utility, generic features, reader uncertainty, and the prior
semantic rank. Exact-question-grouped training, calibration, and internal test
splits were frozen before public-validation evaluation. The best
BA-calibrated model was 121,771 bytes and emitted 30/360 validation facts:

| policy | grouped train-test BA delta | frozen validation BA delta | novel-question BA delta |
| --- | ---: | ---: | ---: |
| raw-utility BA gate | +0.0029 | +0.0056 | 0.0000 |
| control-adjusted utility gate | +0.0131 | -0.0028 | 0.0000 |
| binary rescue/harm gate | +0.0142 | -0.0056 | 0.0000 |
| semantic top-1 then gate | +0.0049 | 0.0000 | 0.0000 |
| semantic top-3 then gate | +0.0016 | 0.0000 | 0.0000 |

The apparent +0.0056 direct-margin gain of the raw-utility gate was not robust:
matched shuffled evidence gained +0.0028, none of its GPT-labeled emissions was
`decisive`, and its five novel-question emissions changed no decisions. The
binary target's larger internal gain reversed on validation. Semantic-first
top-1/top-3 constraints removed the worst prompt hacks but produced no frozen
gain; their novel emissions were absent or harmful.

Full-generation job `30137381` tested the only marginally positive gate with
empty, real, and matched shuffled conditions. Sparse sweeps require special
care: independently regenerating hundreds of identical inactive prompts adds
vLLM batching and parse noise. Reanalysis therefore reuses the empty output
exactly for the 330 inactive rows and compares independently generated outputs
only for the 30 active rows. Under that corrected analysis, generated varied BA
is 0.8056 for both empty and real and 0.8083 for shuffled. Real evidence makes
no deceptive recovery or regression, causes two honest harms, and recovers one
honest row. Reasoning-margin BA is tied at 0.8111, while AUROC falls from 0.8608
empty to 0.8537 real. The direct empty-reasoning margin rises from 0.7944 to
0.8000 BA but loses AUROC, and that proxy does not transfer to generated
decisions.

This closes compact learned abstention for the present index and data. The
failure is not lack of candidate oracle headroom; it is failure to predict
consumer utility on novel question groups and failure of utility-positive
margin shifts to survive free reasoning generation. Do not ship these models,
generate another teacher cache, or tune against validation. A credible revisit
would need genuinely new grouped questions and a training consumer that is
explicitly taught to cite or ignore evidence, evaluated against matched
shuffled controls from the start.
