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

## Verified-correctness reader transfer

The later P70 rank-1 verified-correctness reader was evaluated against the
frozen broad daily Wikidata cards in job `30211335`. With its entity/relation
alignment prompt, varied BA was `0.8083` empty, `0.8028` real, and `0.7750`
shuffled. Real cards produced 18 fixes/17 breaks versus empty and 21/11 versus
shuffled. They lowered recall from `0.7056` to `0.6722` while lowering FPR from
`0.0889` to `0.0667`.

This is retrieval-specific but not deployment-positive: real cards protect
substantially more recall than shuffled cards, yet still regress against no
evidence. The old rank-1 incorrectness weights under the same prompt produce
the same `0.8083/0.8028` empty/real macro BA, so the independent P70 curriculum
does not repair broad-card transfer. Do not package the 44.77 MB broad index on
this evidence. Full protocol and error analysis are P71 in
`docs/prompt_optimization/proposals.md`.

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

## Pretrained cross-encoder capacity ablation

Jobs `30138821` and `30138835` tested whether the hashed linear gate was simply
too small. The submission-oriented model is the 22.7M-parameter
`cross-encoder/ms-marco-MiniLM-L6-v2`; the 149M-parameter
`Alibaba-NLP/gte-reranker-modernbert-base` is a capacity ceiling. Each model
sees the question, assistant answer, frozen reader's empty-evidence score, and
one structured fact. Training appends an explicit `NO_EVIDENCE` candidate and
combines robust utility regression with within-row pairwise ranking. Question
groups receive equal total weight. Raw and shuffled-controlled utility, learning
rates `1e-5` and `3e-5`, and epochs one through three were selected only on the
frozen training calibration partition.

| calibration-selected model | grouped internal BA delta | frozen BA delta | frozen AUROC delta | novel candidate AUROC | novel BA delta | model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM, controlled, `3e-5`, epoch 2 | +0.0037 | 0.0000 | +0.0022 | 0.5873 | 0.0000 | 91,579,425 |
| ModernBERT, controlled, `1e-5`, epoch 3 | -0.0015 | -0.0028 | -0.0075 | 0.5207 | 0.0000 | 602,022,760 |

MiniLM learns a weak reader-utility signal: it raises novel candidate AUROC
from 0.4791 zero-shot to 0.5873 and emits 20/360 frozen rows. Its calibration
precision of 0.714 falls to 0.538 on grouped internal test, 0.450 on frozen
validation, and 0.500 on the two novel emissions. No frozen or novel binary
decision changes correctly on net. ModernBERT improves calibration ranking but
generalizes worse than MiniLM. Its zero-shot novel candidate AUROC is 0.6036,
yet no calibration-safe threshold emits evidence; fine-tuning the
calibration-selected checkpoint reduces it to 0.5207. This is evidence of noisy,
reader-specific targets and limited independent question diversity rather than
an undersized gate.

One non-selected ModernBERT raw-utility checkpoint happened to rescue a single
novel row, raising that 65-row subset by 0.0208 BA. It was inferior under the
predeclared calibration ordering and did not improve AUROC, so selecting it
would be post-hoc validation selection. It is retained only as evidence that
individual recoverable facts exist.

Do not package either cross-encoder or launch a full-generation sweep. MiniLM is
91.6 MB in its saved FP32 form and could plausibly be quantized to roughly one
quarter of that size, but there is no accuracy result that justifies reclaiming
submission space. ModernBERT establishes that adding capacity alone does not
solve transfer. Also interpret the oracle narrowly: it is an upper bound for
selecting facts for the current evidence-naive distilled reader, not for a
reader trained to use evidence. A future revisit should change the consumer and
training distribution together, using balanced useful, empty, shuffled, and
misleading evidence with explicit use/ignore/contradict supervision.

### Transfer-focused MiniLM follow-up

The requested MiniLM follow-up changes the supervision, loss, calibration, and
input representation while holding the 22.7M-parameter network fixed. Job
`30139829` screened continuous raw/controlled utility, binary rescue/harm,
label-free reader-score movement, and masked GPT-OSS semantic decisiveness and
relevance. Pairwise, hard-listwise, and soft-listwise losses all rank facts
against an explicit no-evidence action. Every checkpoint also compares raw
fact/no-evidence margin, candidate-count adjustment, and listwise-softmax
confidence without another forward pass. Configuration selection maximizes the
worst BA delta across calibration and grouped internal test, then their mean;
frozen validation is reporting-only.

The first GPU smoke exposed rows with no retrieved candidates. Hard-listwise
training now treats these as zero-loss no-evidence rows, with a regression test.
The corrected screen completed 126 policy evaluations. Jobs `30140074` and
`30140075` then refined the strongest utility and semantic branches over query
score representation, bottom-layer freezing, learning rate, and three epochs.

| grouped-fold-selected checkpoint | calibration BA delta | internal BA delta | frozen BA delta | novel BA delta | novel candidate AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad screen: controlled binary, pairwise, epoch 1 | +0.0050 | +0.0032 | 0.0000 | 0.0000 | 0.4595 |
| utility refinement: controlled utility, score bucket, `3e-5`, epoch 1 | +0.0037 | **+0.0053** | +0.0028 | 0.0000 | 0.5470 |
| semantic refinement: decisive, hard listwise, `3e-5`, epoch 2 | +0.0017 | +0.0014 | +0.0028 | 0.0000 | **0.9907** |

The utility winner emits 38/555 calibration, 46/619 grouped internal, and
17/360 frozen rows. Controlled-positive precision declines from 0.605 to 0.478
and 0.471 respectively, yet the selected evidence magnitudes still yield small
positive BA and AUROC changes. Only one novel-question row is emitted and its
reader utility is negative without changing the binary decision. The coarse
reader-confidence bucket is better than a raw decimal or omitting reader state;
freezing lower layers is not selected.

The semantic model cleanly learns relation decisiveness on novel questions but
cannot transport its abstention scale: its training-calibrated threshold emits
zero of 65 novel rows despite 0.9907 candidate AUROC and 0.7572 average
precision there. Do not lower this threshold from frozen labels.

Job `30140655` refit the utility winner on 80% of question groups with seeds 42,
43, and 44, retained the other 20% for calibration, and averaged all three state
dictionaries without seed selection. Seeds 43 and 44 calibrate to complete
abstention; seed 42 emits 9/360 frozen rows but does not change frozen BA. The
predeclared soup emits 43/555 calibration, 22/360 frozen, and 2/65 novel rows.
It scores +0.0045 calibration BA, +0.0028 frozen BA, +0.0031 frozen AUROC, and
unchanged novel BA/AUROC. Thus refitting and averaging preserve the one-net-
decision frozen improvement but do not strengthen it.

Job `30140661` initialized from the selected semantic checkpoint and then tuned
controlled utility at `1e-5` and `3e-5` for two epochs. Its grouped-fold-selected
checkpoint abstains on every calibration, internal, frozen, and novel row. The
semantic initialization raises novel utility-candidate AUROC only to 0.5774 and
does not solve cross-objective calibration.

This closes the MiniLM optimization pass. Relative to the first MiniLM result,
the selected soup raises frozen BA delta from 0 to +0.0028 and AUROC delta from
+0.0022 to +0.0031, but novel decisions remain unchanged and the effect is only
one net frozen correction. Do not package the 91.6 MB FP32 model, reclaim space
for quantization, or launch another generated-reader sweep from this result.
Further progress needs new independent question groups and an evidence-trained
consumer, not more tuning of this evidence-naive reader's utility gate.

### Matched evidence consumer and one alternating update

The follow-up changes the consumer before retraining MiniLM. It reuses audited
teacher targets rather than generating another GPT-OSS cache: the existing
evidence-aware target is paired with frozen real retrieval, while the ordinary
target is paired with explicit-empty, cross-dataset shuffled, and reader-harmful
evidence. A reader-helpful condition preserves the correct ordinary target.
This produces 12,122 training records from 2,870 varied source rows:

| condition | rows |
| --- | ---: |
| frozen real retrieval | 2,870 |
| explicit no evidence | 2,870 |
| cross-dataset shuffled evidence | 2,870 |
| old-reader harmful candidate | 1,828 |
| old-reader helpful candidate | 1,684 |

The training recipe remains the selected privileged-distillation default:
regular `Qwen/Qwen3.5-9B`, rank 16/alpha 32, one epoch, AdamW `5e-5`, effective
batch size 32, and varied-only filtering. Smoke job `30142205` completed one
step; full job `30142214` completed 379 steps in 1h21m.

Job `30142250` evaluates exact empty/real/shuffled prompts and reuses empty
generations whenever a condition has no evidence, avoiding batch-order noise.

| reader | empty varied BA | real varied BA | shuffled varied BA | real vs empty fixes/harms | real vs shuffled fixes/harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched consumer | 0.7972 | 0.8000 | 0.7944 | 13 / 15 | 13 / 14 |
| prior evidence-only consumer | 0.8000 | 0.8000 | 0.7972 | 13 / 16 | 13 / 15 |

Matched training removes one harm in each comparison, but does not produce a
net-correct always-on evidence policy. Its larger effect is on selectable
counterfactual headroom. Job `30142251` rescored 33,706 training and 4,202
validation candidates against the matched consumer:

| constrained reader | empty BA | row oracle BA | rescues | harms |
| --- | ---: | ---: | ---: | ---: |
| prior evidence-naive | 0.7944 | 0.8333 | 16 | 2 |
| matched evidence consumer | 0.7861 | 0.8444 | 21 | 0 |

Thus evidence training makes the best available fact more actionable even
though the empty baseline is weaker. Jobs `30142252` and `30142253` tested one
alternating retriever update. Grouped-fold selection chooses controlled utility,
hard-listwise loss, a coarse reader-score bucket, raw fact/no-evidence margin,
no frozen layers, `3e-5`, and epoch one. The selected checkpoint gains +0.0026
calibration and +0.0044 grouped-internal BA but zero frozen BA. The three-seed
refit/soup emits 15/360 frozen rows and gains +0.0028 BA/+0.0014 AUROC; its
controlled-positive precision is 0.333 and it emits no novel-question row.

Job `30142335` is the required full-generation check. It reuses empty outputs on
345 inactive rows and generates only the 15 selected real and shuffled prompts:

| condition | varied BA | recall | FPR | paired outcome vs empty |
| --- | ---: | ---: | ---: | ---: |
| empty | 0.7972 | 0.6611 | 0.0667 | — |
| gated real | 0.8028 | 0.6667 | 0.0611 | 2 fixes, 0 harms |
| gated shuffled | 0.8000 | 0.6611 | 0.0611 | 1 fix, 0 harms |

Only one correction is evidence-specific. Both corrected questions occurred in
training. The honest Buddy Holly row flips under both real and shuffled
evidence; the deceptive uxoricide row flips only for real evidence, but its
card merely says `subclass of: killing of a spouse` and was labeled relevant
but insufficient, not decisive. Several of the other 15 selected cards are
irrelevant, although they do not cross the final binary boundary.

Decision: this alternating update is a useful mechanistic result, not a
submission candidate. It shows that matched consumer training raises oracle
headroom and that selective retrieval can avoid the harms of always-on evidence.
It does not show novel-question emission or a robust fact-specific gain. Do not
package the 91.6 MB FP32 MiniLM, quantize it, evaluate it on test, or lower the
threshold using frozen labels. A future continuation needs new independent
questions or stronger mechanically grounded relevance supervision; another
iteration on these repeated groups would mostly reinforce memorization.
