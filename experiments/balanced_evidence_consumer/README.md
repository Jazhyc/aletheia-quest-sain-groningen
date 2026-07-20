# Balanced evidence consumer

This experiment tests whether the retrieval bottleneck is the reader rather
than the compact Wikipedia corpus. It trains a small rank-1 Qwen LoRA to use
decisive evidence while ignoring empty, irrelevant, mixed, merely topical, or
insufficient references. Honest ordinary factual errors and deceptive
contradictions are included in equal numbers so that a contradiction is not
silently redefined as deception.

## Frozen curriculum

The builder joins the existing varied-only GPT-OSS teacher caches and the
label-blind FEVER training audit by `(dataset, index)`. For each of 2,873 source
rows it produces real decisive, explicit-empty, cross-row shuffled, and
real-plus-shuffled conditions. It adds 2,359 auditor-rejected topical or
insufficient passages and a balanced 109/109 stratum of honest ordinary errors
and deceptive contradictions. The resulting 14,069 records contain 7,035
honest and 7,034 deceptive targets.

Real and shuffled targets retain the independently generated evidence-aware
teacher summaries. Empty and insufficient conditions retain the original
teacher decision but prepend an explicit evidence assessment. The student
never sees labels, ground-truth annotations, or the audit relation. An
`evidence_variant` identity distinguishes intentional prompt variants while the
training loader's default duplicate-row rejection remains unchanged for every
other recipe.

The matched recipe uses `Qwen/Qwen3.5-9B`, a rank-1/alpha-2 LoRA, one AdamW
epoch at `5e-5`, and effective batch size 32. Evaluation is frozen to the
existing deployable Qwen-claim retrieval cache and compares real evidence to
both exact empty prompts and count-matched shuffled passages. No local-test row
is used.

## Existing-reader transfer check

Before training, the older rank-16 FEVER evidence-visible reader was evaluated
on the deployable cache. On varied validation, empty evidence scored `0.8028`
BA, real evidence `0.8056`, and shuffled evidence `0.8000`. Real evidence fixed
three and broke one prediction relative to empty, but its `+0.0028` BA gain was
below the predeclared `+0.005` gate. This is weak positive transfer rather than
a sufficient reader.

## Commands

```bash
sbatch experiments/balanced_evidence_consumer/run_train.sh
sbatch --dependency=afterok:<train-job> \
  experiments/balanced_evidence_consumer/run_evaluate.sh
```

The run is promising only if real evidence improves varied-validation BA over
both controls by at least `0.005`, paired fixes exceed breaks, and FPR does not
increase by more than `0.02` over empty evidence.

## Outcome

Jobs `30215513` and `30215515` completed on 40 GB A100s. Training took 3h45m
wall time (3h42m inside the trainer), reached loss `0.5196`, and produced a
7.3 MB rank-1 weight file. The frozen evaluation took 15m47s including cold
vLLM startup.

| evidence | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | `0.9036` | **`0.8056`** | `0.6722` | `0.0611` | 6 |
| real | `0.9036` | **`0.8056`** | `0.6722` | `0.0611` | 5 |
| shuffled | `0.9024` | `0.8028` | `0.6667` | `0.0611` | 6 |

Real evidence changes five varied predictions relative to empty, fixing three
and breaking two. One fix is only recovery from an empty-condition parse
failure; among rows parsed in both conditions the comparison is two fixes and
two breaks. Against shuffled evidence, real fixes four and breaks three, again
falling to three and three after excluding the shuffled parse failure. Thus the
apparent one-row gain in each paired comparison is not a stable semantic gain.

The 62 evidence-active rows score `0.7742` BA under both empty and real evidence
and `0.7581` with shuffled evidence. The older rank-16 FEVER-visible reader also
scores `0.7742` with real evidence on these rows. Across all varied rows, the
new and old real-evidence readers disagree on 11 predictions with five fixes
and five breaks for the new reader. The balanced curriculum therefore does not
improve the earlier reader; it mostly preserves the same weak
real-versus-shuffled sensitivity.

Decision: the experiment fails the `+0.005` real-over-empty gate. Do not run
local test, package the Wikipedia index, or promote this adapter. The result
also weakens the case that reader training alone is the missing piece: the
remaining sparse retrieval decisions are still dominated by incomplete
evidence, ordinary factual errors that are not deception, and a handful of
format-boundary changes. A rank increase is not justified from this result
because the existing rank-16 reader reaches the same real-evidence BA.
