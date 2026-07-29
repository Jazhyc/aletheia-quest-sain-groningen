# Specialist necessity through NDIF

This frozen benchmark asks whether Phoenix 6.3 still needs its routed
specialists after replacing the ordinary judge with the optimized Qwen-397B
soft-distilled adapter.

It uses normalized literal next-token `0|1` logits, Qwen thinking disabled,
`logits_to_keep=1`, deployment-equivalent left truncation at 2,048 tokens, and
the established length-aware `48/32/16` NDIF batches. No judge text is
generated.

The benchmark covers four distinct response modes:

- Liars' Bench Insider Trading action reports;
- Liars' Bench Harm Pressure Knowledge Report (HP-KR);
- Liars' Bench Harm Pressure Choice, which is not the route called `harm` in
  the notebook;
- the nine reasoning-bearing varied-deception units in both local validation
  and test.

For intent, the optimized judge is evaluated both output-only and with the
final assistant reasoning. Reasoning retains its final 4,000 characters before
the complete prompt receives the same 2,048-token left truncation used by the
submission. The current rank-1 intent adapter and the actual mean-log-odds
route are matched controls.

Run:

```bash
source .venv/bin/activate
python experiments/specialist_necessity_ndif/run.py
```

Ignored artifacts are written under
`results/blackbox/specialist_necessity_ndif_v1/`. Durable conclusions belong
in `docs/auroc_first_strategy.md` and the relevant method documents.

## Frozen result

The full run completed on 2026-07-29 with zero missing requested label logits.
The main Q397 pass evaluated 17 conditions in one NDIF session and took
`277.4s`; the base-Qwen, legacy-action, and rank-1-intent passes took `25.5s`,
`44.7s`, and `62.3s`.

On each 200-row balanced external category:

| category / condition | AUROC | development | confirmation | unique scores |
| --- | ---: | ---: | ---: | ---: |
| action, Q397 ordinary direct | `0.52775` | `0.52100` | `0.53820` | 39 |
| action, current legacy-v2 adapter + empty scaffold | `0.55705` | `0.48020` | `0.64460` | 15 |
| action, Q397 action prompt + empty scaffold | **`0.80725`** | **`0.78220`** | **`0.82880`** | 53 |
| HP-KR, Q397 ordinary direct | `0.88075` | `0.85960` | `0.89940` | 46 |
| HP-KR, current base-Qwen epistemic route | `0.92570` | `0.89800` | `0.95220` | 51 |
| HP-KR, Q397 epistemic prompt + empty scaffold | **`0.93935`** | **`0.91880`** | **`0.95880`** | 53 |
| harm choice, Q397 ordinary direct | **`0.77325`** | **`0.79620`** | **`0.75080`** | 58 |
| harm choice, Q397 harm prompt + empty scaffold | `0.75785` | `0.77520` | `0.73900` | 55 |
| harm choice, base Qwen harm prompt + empty scaffold | `0.62690` | `0.66860` | `0.58720` | 46 |

The Q397 action result beats the current legacy adapter on both halves and all
four source families. It is also consistent with the earlier action study's
`0.7974/0.8148` result, which used the historical privileged-distillation
checkpoint rather than the legacy remote-v2 checkpoint now configured in the
notebook. The Q397 HP-KR result beats the base specialist on both halves and
all three source families. Thus the action and HP-KR *semantic routes and
prompts* remain useful, but their separate adapter/model sessions do not:
both can use the optimized Q397 adapter. The actual Harm Pressure Choice
category does not justify a new route.

On the nine trace-bearing varied units:

| condition | validation macro AUROC | test macro AUROC |
| --- | ---: | ---: |
| Q397 output only | `0.89500` | `0.91111` |
| Q397 with final 4,000 reasoning characters | `0.89639` | `0.92069` |
| current rank-1 intent member alone | `0.88347` | `0.88611` |
| current Q397-output + rank-1 log-odds blend | `0.89750` | **`0.92778`** |
| Q397-reasoning + rank-1 log-odds blend | **`0.90333`** | `0.91833` |

All 720 intent rows had non-empty source reasoning. Tail truncation affected
99/360 validation and 112/360 test traces at the 4,000-character cap, but no
complete prompt exceeded 2,048 tokens. Reasoning visibility improves Q397 by
`+0.00139` validation and `+0.00958` test macro AUROC. The rank-1 evidence is
mixed but still useful: the frozen current blend narrowly beats
reasoning-visible Q397 on validation and materially beats it on test. Retain
the intent adapter for now; do not replace the current blend with the
validation-leading reasoning-plus-rank1 arm because that arm regressed on the
frozen test confirmation.
