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
