# Compact Wikidata RAG

The canonical method decisions and frozen result summary are documented in
`docs/wikidata_rag/README.md`; this directory contains executable experiment
code, tests, and command examples.

This experiment tests whether a submission-sized, CC0 Wikidata fact index can
supply missing factual evidence to the black-box deception judge. It supersedes
the earlier Wikipedia-snippet pilot only at the retrieval layer; the earlier
finding still governs downstream evaluation: evidence raised varied recall but
raised false-positive rate more, so retrieval relevance must be established
before another judge run.

## Relevance metric

`evaluate_retrieval.py` samples balanced public varied-deception training rows,
links the final user/assistant exchange to English Wikipedia pages, and fetches
only the corresponding Wikidata entity cards. Wikipedia prose is not retained.

The primary proxy is **novel target recall**:

1. tokenize the privileged GPT-OSS reasoning summary, which usually states the
   material claim and corrective fact;
2. remove tokens already present in the conversation;
3. measure how many remaining tokens occur in the retrieved Wikidata card;
4. compare against another row's evidence, circularly shuffled within label.

This is stricter than search coverage and less circular than overlap with the
conversation, but it is still only a lexical proxy. Human audit and downstream
balanced accuracy/FPR remain required.

## 100-row training pilot

The frozen seed-42 pilot used 50 honest and 50 deceptive varied-training rows,
three linked entities per row, and a selected set of high-value Wikidata
properties.

| group | entity coverage | target recall | novel target recall | shuffled novel recall | lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 1.000 | 0.172 | 0.043 | 0.010 | +0.033 |
| honest | 1.000 | 0.183 | 0.030 | 0.014 | +0.016 |
| deceptive | 1.000 | 0.160 | 0.055 | 0.006 | +0.050 |

Across rows, 39/100 have nonzero novel target recall and 16/100 reach at least
0.10. A row-bootstrap over per-row real-minus-shuffled recall gives an overall
95% interval of approximately +0.020 to +0.047. The deceptive-only interval is
approximately +0.031 to +0.069; the honest-only interval crosses zero.

Manual audit finds useful structured hits (dates, locations, authorship, cast,
and named relationships) alongside wrong-entity and list-page results. The
pilot therefore supports a compact prototype, not direct judge integration.

## Generalization contract

The API pilot uses the global Wikimedia search corpus and is an upper bound on
what a compact local index can retrieve. Competition examples must **not**
determine which entities enter the shipped database.

Build the prototype from an independent broad seed distribution:

1. collect English Wikipedia's generally popular pages across many dates, not
   pages matched from competition conversations;
2. map titles to Wikidata QIDs and retain only CC0 Wikidata labels, aliases,
   descriptions, and selected factual statements;
3. add one-hop statement-value entities so facts remain readable and related
   entities receive some coverage;
4. reserve breadth quotas across people, places, organizations, events, works,
   species, and scientific concepts so popularity does not collapse the index
   into current entertainment/news;
5. exclude disambiguation/list/meta entities and deprecated statements;
6. stop by measured final ZIP contribution, targeting 50--70 MB compressed.

Freeze that database before evaluation. Tune schema and lexical retrieval on
training, then measure on validation without adding missed entities. Report:

- offline entity-card coverage and novel target recall versus within-label
  shuffled evidence;
- coverage by broad entity type and by dataset unit;
- downstream balanced accuracy, recall, and especially FPR against an exact
  no-evidence control;
- a manual audit of high-, median-, and zero-relevance retrievals.

Only proceed to a judge/adaptor experiment if the compact frozen index retains
a clear deceptive-row relevance lift without a comparable honest-row increase.

## Broad frozen-index results

The builder uses no competition examples to select entities. It aggregates
generally popular English Wikipedia titles across 2021--2024, maps them to
Wikidata, and adds referenced one-hop entities. The shipped content is limited
to Wikidata labels, aliases, descriptions, and selected statements. Three
increasingly broad builds establish the size/coverage curve:

| snapshots | cards | raw SQLite | ZIP contribution |
| --- | ---: | ---: | ---: |
| 48 monthly | 29,920 | 16.86 MB | 6.49 MB |
| 209 weekly | 99,600 | 53.70 MB | 20.52 MB |
| 1,461 daily | 221,050 | 116.94 MB | 44.77 MB |

The daily build exhausts the distinct popular-page seeds available under this
sampling scheme before its requested 300,000-card ceiling. The current repository
packages to 109.45 MB without the database, so adding the daily database projects
to about 154.22 MB under the same ZIP compression, leaving about 45.78 MB below
the 200 MB guardrail.

On the original balanced 100-row pilot, offline retrieval from the daily frozen
index produced the following novel-target-recall results:

| group | real | shuffled | lift |
| --- | ---: | ---: | ---: |
| all | 0.0366 | 0.0084 | +0.0282 |
| honest | 0.0301 | 0.0113 | +0.0187 |
| deceptive | 0.0432 | 0.0056 | +0.0376 |

The decisive full-cache evaluation covers all 2,877 successfully parsed
varied-deception training examples (1,440 honest and 1,437 deceptive):

| group | real | shuffled | lift | bootstrap 95% interval | rows with nonzero recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 0.032 | 0.009 | +0.0230 | +0.0207 to +0.0253 | 877/2,877 (30.5%) |
| honest | 0.026 | 0.009 | +0.0176 | +0.0146 to +0.0207 | 333/1,440 (23.1%) |
| deceptive | 0.039 | 0.010 | +0.0285 | +0.0249 to +0.0320 | 544/1,437 (37.9%) |

The deceptive lift exceeds the honest lift by +0.0109 (row-bootstrap 95%
interval +0.0064 to +0.0156). This supports a frozen downstream judge ablation:
the independent compact corpus retrieves genuinely relevant missing facts and
does so more often on deceptive rows. It does **not** establish that evidence
improves classification. Honest evidence also has a clear positive relevance
lift, and the earlier prose-RAG judge lost balanced accuracy by raising FPR.
The next experiment must therefore compare the exact same frozen judge with and
without evidence and report recall, FPR, and balanced accuracy; the database
should not enter `submission/` before that test passes.

Build artifacts and row-level evaluations stay under ignored
`results/blackbox/wikidata_rag_{broad,weekly,daily}_v1/`.

## Frozen-judge three-condition sweep

Job `30123829` evaluated the selected one-epoch varied-only AdamW `5e-5`
student without retraining. All 822 validation rows received the same explicit
reference block. Varied rows used either no evidence, their three real
Wikidata cards, or three shuffled cards; instructed rows received the identical
no-evidence sentinel in all conditions. The shuffled assignment was label-blind
and bijective across all 360 varied rows, with no donor sharing the recipient's
dataset unit or row index.

Generated binary decisions gave:

| condition | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | 0.9060 | 0.7972 | 0.6722 | 0.0778 | 7 |
| real Wikidata | **0.9095** | **0.8056** | **0.6778** | **0.0667** | **5** |
| shuffled Wikidata | 0.9071 | 0.8000 | 0.6722 | 0.0722 | 6 |

Among rows parsed in each relevant pair, real versus empty produced seven
deceptive recoveries and six deceptive regressions, while harming two honest
rows and recovering five honest rows. Real versus shuffled produced six/four
deceptive recoveries/regressions and two/five honest harms/recoveries. On the
347 rows parsed in all three conditions, BA was 0.8088 empty, 0.8198 real, and
0.8053 shuffled.

The constrained empty-summary margin shows the same ordering: varied AUROC was
0.8600 empty, 0.8735 real, and 0.8611 shuffled. Its fixed-0.5 varied BA was
0.8000, 0.8194, and 0.7889, respectively. Thus the structured evidence does not
reproduce the old Wikipedia prompt's generic suspicion spike; real evidence
slightly lowers FPR and moves deceptive margins upward relative to shuffled
cards.

The effect is not stable enough to deploy. A dataset-unit bootstrap gives a 95%
interval of approximately -0.0194 to +0.0361 for the +0.0083 real-minus-empty
BA change, and -0.0111 to +0.0222 for the +0.0056 real-minus-shuffled change.
Real evidence improves four dataset units, hurts three, and ties two versus
empty. Moreover, the three conditions contain identical instructed prompts,
yet two of 462 instructed predictions flip across their repeated generations;
this 0.43% empirical nondeterminism floor is comparable to the net varied gain.

Manual inspection explains why evidence conditioning remains useful as a
training proposal but not as a prompt-only submission change. Some cards help
Qwen ignore deceptive system intent and accept a truthful answer. Other cards
support the headline answer while omitting a poisoned secondary claim, causing
Qwen to miss the deception; wrong-entity matches can similarly reinforce a
false answer. Do not add the database to the judge or evaluate this condition
on local test. If this path continues, use a matched evidence-conditioned SFT
control with empty and shuffled hard negatives and explicitly teach
`supports|contradicts|irrelevant|insufficient` evidence relations.

Artifacts are under
`results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/validation_wikidata_3condition_v1/`.

## Evidence-aware privileged distillation

The next frozen experiment incorporates each row's real retrieved cards into
the privileged GPT-OSS teacher trace and the student's input. It deliberately
uses no shuffled, corrupted, duplicated, or counterfactual augmentation. The
teacher silently decides whether each card concerns the same entity and settles
a material output claim. It incorporates a directly relevant fact, but ignores
wrong-entity, merely topical, ambiguous, or insufficient cards completely. The
ordinary `reasoning_summary` plus `Prediction:0|1` target schema is unchanged.

The cache is rebuilt directly from all 2,880 labeled varied-training rows rather
than from the old parsed teacher artifact, so the three examples whose original
teacher traces were malformed are not silently omitted. It contains exactly
three cards per row and is balanced 1,440/1,440 by label. Apart from the shared
reference block and evidence-aware teacher/student wording, training retains the
selected standard recipe: regular `Qwen/Qwen3.5-9B`, varied-only data, rank-16
alpha-32 LoRA, one epoch, AdamW `5e-5`, and effective batch size 32. Configuration:
`configs/pid_teacher_wikidata_evidence_v1.yaml`.

```bash
.venv/bin/python experiments/wikidata_rag/build_validation_cache.py \
  --database results/blackbox/wikidata_rag_daily_v1/wikidata.sqlite \
  --output results/blackbox/wikidata_rag_daily_v1/train_retrieval_raw.jsonl \
  --split train --no-shuffle

.venv/bin/python experiments/wikidata_rag/build_teacher_cache.py \
  --input results/blackbox/wikidata_rag_daily_v1/train_retrieval_raw.jsonl \
  --output results/blackbox/wikidata_rag_daily_v1/train_teacher_cache.jsonl

sbatch experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_wikidata_evidence_v1

sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_teacher_wikidata_evidence_v1
```

## Rule-based claim gate

`claim_retrieval.py` is the conservative CPU-only alternative to a separate
query-planning LLM. It extracts the last user question and assistant answer,
maps explicit question forms to predicates retained in the compact Wikidata
index, links only the primary question subject, removes unrelated predicates,
and abstains on unsupported, ambiguous, or wrong-type matches. An abstention
produces no reference block. The gate currently supports direct relations such
as country/location, capital, date, author, director, founder, performer, cast,
and membership; broad identity/category questions deliberately abstain.

The final rule iteration was selected using extraction/retrieval audits rather
than validation labels. On 2,877 parsed varied-training summaries it passed 118
rows (4.1%). On those rows, evidence precision increased from 0.089 for the
original three-card retrieval to 0.433 for the gated facts. The price is low
coverage and lower conditional novel-target recall (0.026 versus 0.042), so a
matched judge evaluation is required before using the cache for distillation.
On the 360 varied-validation rows the frozen gate passes eight rows: repeated
direct facts for Carthage/Tunisia and Galapagos/Ecuador, plus Gabon/Libreville
and *Thus Spoke Zarathustra*/Friedrich Nietzsche.

Matched baseline-adapter validation job `30134337` scored 0.8000 varied BA,
versus 0.7944 in the older baseline artifact. The paired audit shows this is
not a retrieval gain: none of the eight evidence-bearing rows changed binary
prediction, while all nine changed predictions occurred on the 352 abstained
rows. Five abstained-row replay flips fixed errors and three introduced errors.
The judge explicitly cited the gated facts, but those facts only reinforced
decisions it already made. Do not regenerate teacher traces or train another
student from this gate. The rule extractor is retained as a tested CPU-only
prototype; useful continuation requires broader claim-level factual coverage,
not looser entity matching.

### Expanded structured index

The v2 schema keeps the frozen 221,050-entity universe but refetches each card
with 128 retained properties instead of relying on claims cached under the
original narrower schema. It adds direct relations for borders, leadership,
ownership, sports, events, screenwriters, composers, names, and causes, plus
seven compact qualifier types such as start/end time and series ordinal. Cards
retain at most 32 facts and 12 aliases. Referenced object labels are cached
separately so newly added entity-valued claims do not disappear merely because
their object is outside the selected entity universe.

The refresh is resumable and versioned: it reads only QID selection, source,
and popularity from the old card cache, and does not alter the v1 database.
The wrapper loads `WIKIMEDIA_ACCESS_TOKEN` from `.env` when configured; the
token improves sustained request throughput but does not raise the API's
50-entity batch ceiling for this account.

```bash
sbatch experiments/wikidata_rag/run_expanded_index.sh
sbatch --dependency=afterok:<build-job-id> \
  experiments/wikidata_rag/run_expanded_gate_eval.sh
```

The dependent evaluation compares training-summary relevance and frozen
validation coverage with the final v1 gate. A judge ablation is warranted only
if coverage rises materially without collapsing conditional evidence precision.

Build job `30134717` completed in 57m57s. The final index contains 2,169,868
raw retained claims and 205,915 extra object labels. SQLite size is 165,294,080
bytes and ZIP size is 62,333,290 bytes. With the current 116,512,716-byte
submission tree, projected package size is 178,846,006 bytes, leaving about
21.2 MB under a decimal 200 MB cap.

The frozen gate test (`30134751`) did not meet the judge-ablation criterion.
Training coverage rose from 118 to 145 of 2,880 rows (4.1% to 5.0%), but
conditional evidence precision on 2,877 parsed teacher summaries fell from
0.433 to 0.384. Covered rows with any novel-summary token hit rose from 29 to
40, but this is still only 1.4% of parsed rows. Validation coverage rose from
8/360 to 12/360. Of the four new rows, only the Malawi leadership card directly
answers the question; the others are irrelevant event metadata, a wrong
St. George entity, or year-mismatched Nobel winners. Do not launch a judge run,
regenerate teacher traces, or train a student from this cache. A larger property
schema alone does not solve claim planning and temporal/entity disambiguation.

### Programmatic query constraints

The selected follow-up keeps the expanded v2 database fixed and changes only
the CPU query planner. It narrows winner, participant, organizer, sport, cast,
and location wording to the requested predicate; prefixes recurring event names
with explicit years; rejects facts with conflicting years; normalizes `St` and
`Saint`; enforces subject types for saints and island-name questions; and uses
direct aliases, burial facts, or tightly matched descriptions where appropriate.
Unsupported answer slots such as “which river forms the border” or “which film
won the actor an award” now abstain instead of returning topical metadata.

On the frozen 2,877 parsed training summaries, the original narrow gate covered
118 rows with conditional evidence precision 0.433 and 24 rows containing a
novel-summary token hit. The unconstrained expanded gate covered 145 with 0.384
precision and 40 hits. The selected programmatic gate covers 132 (4.6%), reaches
0.414 precision, and retains 38 hits. It therefore keeps 14/16 of the expanded
gate's additional hits while removing 13 low-value rows and recovering most of
the precision loss.

The frozen validation cache contains ten evidence-bearing rows instead of 12.
It removes the irrelevant Ernest Borgnine conflict and year-mismatched Nobel
winners, changes St George from a Louisiana locality to `place of burial: Lod`,
and keeps the direct Malawi leadership fact. Training examples also show correct
Italy/Argentina World Cup winners, a direct Kerkyra alias, and the Cyprus-bearing
Akrotiri description. This is a meaningful retrieval improvement, but ten
validation rows are still too few for a useful judge or student-training run.
Artifacts use the `*_programmatic_v1` names under
`results/blackbox/wikidata_rag_expanded_v2/`.

### Compact bidirectional relation index

The next programmatic sweep tested six CPU-only extensions independently: a
compact forward/reverse relation table, answer-first inverse lookup, uncapped
histories with qualifiers, training-summary-only predicate mapping search,
mechanically generated routing cases, and a constrained office/jurisdiction
two-hop. No class labels were used for query mapping or routing selection.

The selected relation database contains 512,162 facts over 233,239 labeled
nodes and 33 predicates. It preserves 55,210 start years, 40,072 end years,
37,249 point years, and 334 `for work` qualifiers. SQLite size is 39,448,576
bytes and ZIP size is 17,118,369 bytes. Shipping it alongside the expanded v2
entity linker projects to 195,964,375 bytes with the current submission tree,
leaving about 4.0 MB below the decimal 200 MB limit. This is viable but leaves
little room for another model or large cache.

The isolated direct relation lookup covered 87 training rows with 26 novel
teacher-summary hits. Enabling bidirectional answer lookup covered 91 and
raised hits to 29, a small positive result. Mechanically generated queries
initially exposed seven missing routes (voice actors, characters, participant
history, significant events, held positions, awards, and lyricists); the
selected rules pass all 32 generated relation cases. The split-half mapping
search recovered high-precision expected mappings such as author→author and
director→director, but proposed no reliable new cross-relation mapping;
low-precision coincidences were rejected.

Targeted history/qualifier enrichment is valuable for specific slots. It turns
the Ernest Borgnine/Oscar question into direct qualified evidence for *Marty*.
A constrained person→office→jurisdiction→monarch query resolves Neville
Chamberlain's 1937–1940 premiership against George VI's 1936–1952 reign and
rejects George V; it produces the correct verdict on all three repeated
training variants. Broad unconstrained multi-entity lookup was rejected after
validation audits exposed incidental-place errors. The selected emitter
requires a support/counterevidence relation and trusts later question spans
only for typed works or explicit coordination.

On 2,877 parsed varied-training summaries, the selected cache covers 130/2,880
source rows, reaches 0.538 conditional evidence precision, and has 48 rows with
a novel teacher-summary token hit. This improves both precision and hits over
the previous selected programmatic gate (132 rows, 0.414 precision, 38 hits),
at nearly identical coverage. On varied validation it covers 15/360 rather
than 10/360. The five additions are manually relevant: Milan for Derby della
Madonnina, *Marty* for Borgnine's award, Laos for Vientiane, Portugal for
Benfica/Porto, and Boston for the three-landmark question; none of the prior
ten rows is lost.

Artifacts are under `results/blackbox/wikidata_rag_relations_v1/`. The
selected database is `relations_selected.sqlite`, the frozen caches are
`train_selected_v5.jsonl` and `validation_selected_v5.jsonl`, and the
relevance, mapping-search, routing, build, and validation reports remain beside
them. This is the strongest purely programmatic retrieval configuration so far,
but 4.5% training coverage is still too low to justify another teacher/student
run without first adding new high-confidence relation families.

### GPT-OSS relevance supervision

The supervised follow-up builds bounded full-card candidates, asks local
GPT-OSS-120B for three-way decisiveness labels, trains compact hashed rankers,
and evaluates exact-question-group novelty separately. Generated labels and
models remain ignored under
`results/blackbox/wikidata_rag_gptoss_supervision_v1/`.

```bash
python -m experiments.wikidata_rag.build_gptoss_supervision_candidates \
  --input results/blackbox/wikidata_rag_relations_v1/train_selected_v5.jsonl \
  --entity-database results/blackbox/wikidata_rag_expanded_v2/wikidata.sqlite \
  --output results/blackbox/wikidata_rag_gptoss_supervision_v1/train_candidates.jsonl

sbatch experiments/wikidata_rag/run_gptoss_retrieval_supervision.sh \
  --input results/blackbox/wikidata_rag_gptoss_supervision_v1/train_candidates.jsonl \
  --output results/blackbox/wikidata_rag_gptoss_supervision_v1/train_labels.jsonl \
  --label-mode decisive --candidates-per-row 12

python -m experiments.wikidata_rag.train_gptoss_fact_ranker \
  --input results/blackbox/wikidata_rag_gptoss_supervision_v1/train_labels.jsonl \
  --validation-input results/blackbox/wikidata_rag_gptoss_supervision_v1/validation_labels.jsonl \
  --feature-mode generic --minimum-precision 0.8 \
  --output-model results/blackbox/wikidata_rag_gptoss_supervision_v1/ranker.npz \
  --output-report results/blackbox/wikidata_rag_gptoss_supervision_v1/ranker_report.json
```

Job `30136046` labeled 2,700 rows in 27m36s with 97.0% structural validity.
Decisive/non-decisive agreement on repeated facts was 98.9% in training and
100% in validation. Synthetic and GPT-paraphrased grounded data did not transfer
to real grouped questions. The selected 60 KB generic model achieved 0.975
candidate AUROC and 8/8 top-1 hits on validation questions absent from training,
but its training-calibrated high-precision threshold emitted none of them.
Do not count the 11/11 full-validation emission result as generalization: all 11
questions occurred in training. See `docs/wikidata_rag/README.md` for the full
audit and downstream evidence-sweep result.

Frozen downstream job `30136343` used
`build_ranked_sweep_cache.py` and the existing empty/real/shuffled judge harness.
Always-on ranked evidence reduced varied BA from 0.8083 to 0.8056 by trading a
small recall gain for a larger FPR increase; shuffled evidence also scored
0.8056. The novel-question subset did not show an evidence-specific advantage.
This closes the always-on fallback and does not justify another teacher/SFT run.

### Counterfactual utility supervision

The RECOMP/REPLUG-style follow-up replaces semantic labels with the measured
effect of each fact on the frozen student's correct-label log probability. It
also scores one unrelated fact per row so a policy can learn an explicit
no-evidence action from raw or shuffled-adjusted utility.

```bash
sbatch experiments/wikidata_rag/run_counterfactual_utility.sh

python experiments/wikidata_rag/analyze_counterfactual_utility.py \
  --input results/blackbox/wikidata_rag_counterfactual_utility_v1/train.jsonl \
          results/blackbox/wikidata_rag_counterfactual_utility_v1/validation.jsonl \
  --output results/blackbox/wikidata_rag_counterfactual_utility_v1/analysis.json

python experiments/wikidata_rag/train_utility_retriever.py \
  --input results/blackbox/wikidata_rag_counterfactual_utility_v1/train.jsonl \
  --validation-input results/blackbox/wikidata_rag_counterfactual_utility_v1/validation.jsonl \
  --semantic-model results/blackbox/wikidata_rag_gptoss_supervision_v1/ranker_real_generic_groupnovel_v3.npz \
  --selection-objective balanced_accuracy \
  --output-model results/blackbox/wikidata_rag_counterfactual_utility_v1/utility_ranker_ba.npz \
  --output-report results/blackbox/wikidata_rag_counterfactual_utility_v1/utility_ranker_ba_report.json
```

Job `30137268` scored 37,908 real candidates and 6,480 controls in 14m59s.
The direct-margin oracle is large (+0.0444 validation BA), but all learned
variants fail to transfer reliably. The raw-utility gate improves grouped
training test by 0.0029 BA and frozen validation by 0.0056, but half of the
validation change is reproduced by shuffled evidence and novel-question BA is
unchanged. Control-adjusted and binary-rescue targets regress frozen BA;
semantic-top-1 and top-3 candidate constraints tie the empty baseline.

Full-generation job `30137381` rejects the raw-utility gate after correcting
sparse-sweep noise. Empty and real both score 0.8056 generated varied BA;
shuffled scores 0.8083. Real evidence produces no deceptive recovery and net
one honest harm. Use `reanalyze_sparse_judge_sweep.py` for old sparse sweeps;
the main harness now automatically reuses empty outputs on rows where all
condition prompts are identical. Artifacts are under
`results/blackbox/wikidata_rag_counterfactual_utility_v1/` and the selected
student’s `validation_utility_gate_ba_3condition_v1/` directory. Do not ship a
utility gate from this experiment.

### Pretrained cross-encoder utility gates

The capacity ablation fine-tunes an off-the-shelf reranker against the same
counterfactual cache. It uses no `sentence-transformers` dependency: the script
loads the one-logit reranker through Transformers, balances exact question
groups, and jointly trains robust utility regression and within-row pairwise
ranking against an explicit no-evidence candidate.

```bash
sbatch experiments/wikidata_rag/run_cross_encoder_utility.sh \
  --input results/blackbox/wikidata_rag_counterfactual_utility_v1/train.jsonl \
  --validation-input results/blackbox/wikidata_rag_counterfactual_utility_v1/validation.jsonl \
  --model-name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output-dir results/blackbox/wikidata_rag_cross_encoder_minilm_v1 \
  --targets controlled_utility utility \
  --learning-rates 1e-5 3e-5 --epochs 3

python experiments/wikidata_rag/compare_cross_encoder_utility.py \
  results/blackbox/wikidata_rag_cross_encoder_minilm_v1/report.json \
  results/blackbox/wikidata_rag_cross_encoder_modernbert_v1/report.json
```

Smoke job `30138794` completed in 1m44s. Full MiniLM job `30138821`
completed in 5m05s; ModernBERT ceiling job `30138835` completed in 14m44s.
Calibration selects controlled utility for both. MiniLM (`3e-5`, epoch 2)
improves grouped internal BA by 0.0037 but ties frozen and novel BA; its frozen
AUROC change is +0.0022 and novel candidate AUROC is 0.5873. ModernBERT (`1e-5`,
epoch 3) regresses grouped internal BA by 0.0015 and frozen BA by 0.0028; novel
candidate AUROC is 0.5207. Saved FP32 sizes are 91.6 MB and 602.0 MB.

The larger reranker does not improve transfer, and MiniLM does not clear the
direct-margin prerequisite for another expensive full-generation sweep. Keep
the code as a reproducible capacity diagnostic; do not package either model.
See `docs/wikidata_rag/README.md` for checkpoint-selection cautions and the
evidence-consumer interpretation.

The transfer-focused MiniLM follow-up broadens the objective rather than the
network. It compares continuous and binary reader utility, label-free reader
score movement, and masked GPT-OSS `decisive`/`relevant` supervision. Pairwise
regression is compared with hard and soft listwise ranking against an explicit
no-evidence action. Exact question groups remain disjoint, each configuration
uses the same initialization/shuffle/dropout seed, and selection maximizes the
worst BA change across the calibration and grouped internal-test folds before
using their mean as a tie-break. The frozen validation and novel-question rows
are reported but never enter checkpoint selection.

```bash
sbatch experiments/wikidata_rag/run_cross_encoder_utility.sh \
  --input results/blackbox/wikidata_rag_counterfactual_utility_v1/train.jsonl \
  --validation-input results/blackbox/wikidata_rag_counterfactual_utility_v1/validation.jsonl \
  --model-name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output-dir results/blackbox/wikidata_rag_cross_encoder_minilm_v2_screen \
  --targets controlled_utility utility binary_utility controlled_binary_utility \
            score_delta semantic_decisive semantic_relevant \
  --loss-modes pairwise hard_listwise soft_listwise \
  --learning-rates 3e-5 --epochs 2 --selection-mode robust

sbatch experiments/wikidata_rag/run_refit_cross_encoder_utility.sh \
  --input results/blackbox/wikidata_rag_counterfactual_utility_v1/train.jsonl \
  --validation-input results/blackbox/wikidata_rag_counterfactual_utility_v1/validation.jsonl \
  --sweep-report results/blackbox/wikidata_rag_cross_encoder_minilm_v2_refine_utility/report.json \
  --output-dir results/blackbox/wikidata_rag_cross_encoder_minilm_v2_refit
```

The refinement stage also ablates the raw decimal reader score against a coarse
confidence bucket and no reader score, plus bottom-layer freezing and learning
rate. Every checkpoint additionally compares the raw fact-versus-no-evidence
margin with candidate-count-adjusted and listwise-softmax confidence; these
preserve within-row ranking while correcting the fact-count bias in a global
abstention threshold and require no additional forward passes. The final refit
uses the selected epoch count, combines the original train
and grouped internal-test folds, retains calibration untouched, trains three
predeclared seeds, and averages all three state dictionaries rather than
choosing a seed post hoc.

Jobs `30139829`, `30140074`, and `30140075` completed the screen and refinements.
The grouped-fold utility winner uses controlled utility, pairwise loss, a coarse
reader-confidence bucket, no frozen layers, `3e-5`, and epoch one. It improves
calibration/internal BA by +0.0037/+0.0053 and frozen BA/AUROC by
+0.0028/+0.0024, with unchanged novel BA. The semantic winner reaches 0.9907
novel decisive-candidate AUROC but emits no novel rows at its training-calibrated
threshold.

Three-seed refit/soup job `30140655` finishes at +0.0028 frozen BA and +0.0031
frozen AUROC, again with no novel decision change. Semantic-then-utility job
`30140661` calibrates to complete abstention. These results are too small for a
full-generation or packaging follow-up; see `docs/wikidata_rag/README.md` for
the complete selection and split metrics.


```bash
PYTHONPATH=. .venv/bin/python experiments/wikidata_rag/build_claim_gated_cache.py \
  --database results/blackbox/wikidata_rag_daily_v1/wikidata.sqlite \
  --input results/blackbox/wikidata_rag_daily_v1/train_retrieval_raw.jsonl \
  --output results/blackbox/wikidata_rag_daily_v1/train_claim_gated.jsonl

PYTHONPATH=. .venv/bin/python experiments/wikidata_rag/evaluate_claim_gate.py \
  --teacher-cache results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --broad-cache results/blackbox/wikidata_rag_daily_v1/train_retrieval_raw.jsonl \
  --gated-cache results/blackbox/wikidata_rag_daily_v1/train_claim_gated.jsonl \
  --output results/blackbox/wikidata_rag_daily_v1/claim_gate_train_diagnostics.json
```

## Command

```bash
set -a && source .env && set +a
.venv/bin/python experiments/wikidata_rag/evaluate_retrieval.py \
  --output results/blackbox/wikidata_rag_varied_train_pilot/retrieval.jsonl \
  --report results/blackbox/wikidata_rag_varied_train_pilot/report.md \
  --per-label 50
```

The access token is optional and is never written to artifacts.

Build and evaluate the selected daily index with:

```bash
sbatch experiments/wikidata_rag/run_broad_index_daily.sh

.venv/bin/python experiments/wikidata_rag/evaluate_offline_index.py \
  --database results/blackbox/wikidata_rag_daily_v1/wikidata.sqlite \
  --input results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --output results/blackbox/wikidata_rag_daily_v1/offline_relevance_full_train.jsonl \
  --report results/blackbox/wikidata_rag_daily_v1/offline_relevance_full_train.md

sbatch experiments/wikidata_rag/run_judge_sweep.sh
```
