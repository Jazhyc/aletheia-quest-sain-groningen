# Compact Wikidata RAG

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
