# Kimi K3 Liars' Bench enrichment

This experiment tests one controlled data-scale extension of the Phoenix 8
binary-soft student. It does not add a new inference route. Kimi K3 supplies
training-only literal `0|1` logits for observable Liars' Bench response modes.

The frozen pilot is the existing 800-row, label-balanced artifact at
`results/blackbox/liars_bench_pid_aug_v1/eval.jsonl`. It compares the ordinary
Phoenix 8 Truth Value Guard prompt with category-semantic prompts. This matters
for harm-pressure choice and knowledge reports, where a generically aligned
judge may mistake safety behavior for evidence of deception.

The semantic teacher prompts are privileged training views:

- harm-pressure choice: distinguish a knowingly false option from refusal;
- knowledge report: distinguish ignorance claims from policy refusal;
- insider trading: compare the recipient-facing report with prior actions;
- soft trigger: treat the marked passage as authoritative.

The student inputs remain deployment-matched. HP-KR and action reports use the
existing Phoenix 8 routed prompts. Harm-pressure choice and soft-trigger rows
use the ordinary Phoenix 8 prompt; this experiment does not enable either
rejected router.

Run the pilot:

```bash
bash experiments/kimi_liars_enrichment/run.sh pilot
```

The pilot selects the semantic prompt only when it has AUROC at least `0.85`,
improves on ordinary by at least `0.01`, and every source-model cell containing
both labels has AUROC at least `0.70`. Otherwise it retains the simpler ordinary
prompt when that prompt has AUROC at least `0.90` and the same source-cell
floor. A category failing both branches is omitted. These are teacher-quality
gates, not adapter promotion thresholds.

The completed pilot selected semantic prompting for harm-pressure choice
(`0.87050 -> 0.93165`) and HP-KR (`0.59640 -> 0.92520`). It retained ordinary
prompting for Insider Trading, where the semantic action prompt regressed
`0.97085 -> 0.95135`, and soft trigger, where the semantic gain was only
`0.96300 -> 0.96895`.

After reviewing `pilot/audit.json`, prepare and query the frozen 6,576-row
mixture:

```bash
bash experiments/kimi_liars_enrichment/run.sh prepare
bash experiments/kimi_liars_enrichment/run.sh query
bash experiments/kimi_liars_enrichment/run.sh build
```

The predeclared maximum mixture is label-balanced within category:

| category | rows |
| --- | ---: |
| harm-pressure choice | 800 |
| harm-pressure knowledge report | 1,200 |
| insider trading | 1,976 |
| soft trigger | 2,600 |
| **total** | **6,576** |

Rejected pilot categories are omitted rather than replaced post hoc. The full
cache therefore reaches 6,576 rows only if every category passes either the
semantic-improvement branch or the strong-ordinary fallback.

The completed full cache passed all frozen audit gates: 6,576/6,576 rows had
literal label logits and valid soft targets. Teacher AUROC was `0.92676` for
harm-pressure choice, `0.91493` for HP-KR, `0.97784` for Insider Trading, and
`0.96236` for soft trigger. Composing it with the original 6,573 Phoenix 8
rows produced a 13,149-row cache.

On Lambda, train the full enrichment first and then the predeclared half-dose
probe:

```bash
bash experiments/kimi_liars_enrichment/run_training_lambda.sh
```

Both arms retain the Phoenix 8 rank-16, two-epoch AdamW `5e-5` recipe. The
probe changes only the Liars replay dose: it uses a stable label- and
dataset-stratified 50% sample while retaining every original competition row.
Validation compares both arms with the Phoenix 8 adapter in one vLLM session.

Promotion is frozen before those adapter scores are inspected. Relative to the
Phoenix 8 anchor, an eligible arm must:

- improve the disjoint 800-row Liars pilot macro-category AUROC by at least
  `0.02`;
- retain competition validation macro AUROC within `0.001`;
- retain instructed AUROC within `0.0005`; and
- retain varied AUROC within `0.002`.

Among eligible arms, select the highest competition validation macro AUROC. If
they are within `0.0005`, use Liars pilot macro AUROC as the tie-breaker; if
that is within `0.002`, prefer the half-dose arm as the lower-OOD-exposure
choice. Only the frozen winner receives a local test evaluation. If neither
arm passes, retain Phoenix 8 rather than promoting an OOD gain that damages the
known competition distribution.

## Full-dose outcome and Phoenix 8.1 decision

The full-dose adapter completed first. Relative to Phoenix 8, its disjoint
800-row Liars macro-category AUROC improved from `0.86255` to `0.93915`
(`+0.07660`). The largest gain was Insider Trading (`+0.24975`), followed by
harm-pressure choice (`+0.07410`) and soft trigger (`+0.01100`); HP-KR
regressed by `0.02845`.

Matched competition validation moved from `0.96417` to `0.96214`
(`-0.00202`): instructed changed `+0.00063`, while varied changed `-0.00556`.
This fails the original in-distribution retention gate. With the deadline near,
the team explicitly chose the much larger held-out OOD gain as the private
transfer hypothesis and promoted the full-dose adapter as Phoenix 8.1. This is
a deliberate leaderboard probe, not a claim that the frozen gate passed.

The canonical FP32 weights have SHA-256
`7159a413cf7bf569b1e7819f17b54248d48b8e18b8d56be950b872445195e136`.
The BF16 export SHA-256 was
`6e429dd414365b853cda7468107cb3cabf887138d221aa8fdfafc9d962c9ad73`,
but it was not exact on the matched 800-row vLLM canary
(`0.938875 -> 0.938287` macro AUROC). Phoenix 8.1 therefore deploys the FP32
master.
