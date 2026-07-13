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

## Command

```bash
set -a && source .env && set +a
.venv/bin/python experiments/wikidata_rag/evaluate_retrieval.py \
  --output results/blackbox/wikidata_rag_varied_train_pilot/retrieval.jsonl \
  --report results/blackbox/wikidata_rag_varied_train_pilot/report.md \
  --per-label 50
```

The access token is optional and is never written to artifacts.
