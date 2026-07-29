# Phoenix base-judge verbalizer sweep

This experiment compares output verbalizers for the unadapted
`Qwen/Qwen3.5-9B` Phoenix judge on the frozen validation split through
competition NDIF. It uses Phoenix 4.0's structural renderer and changes only
the output contract and direct next-token boundary.

The arms are the exact deployed `0/1` contract, an explicit-semantics `0/1`
control, `False/True`, `No/Yes`, `honest/deceptive`, neutral `A/B`, and an
expected-score readout over `1` through `7`. All target strings are required to
be single Qwen tokens. Binary arms normalize the selected two logits; the
ordinal arm normalizes its seven logits and returns the expected rating scaled
to `[0, 1]`.

Every arm uses identical record batches and padded tensor widths. Requested
logits are saved in resumable shards so alternative fixed readouts can be
audited without another NDIF query.

Run:

```bash
.venv/bin/python \
  experiments/phoenix_verbalizer_sweep/run_ndif_base_verbalizers.py
```

Artifacts are written under
`results/blackbox/phoenix_base_qwen9b_verbalizer_sweep_validation_v1/`.
Select only on macro validation AUROC; keep local test untouched until a
verbalizer and its matched distillation plan are frozen.

## Result

The complete 822-row NDIF validation sweep scored 5,754 prompt/condition rows
in 146.6 seconds. No prompt reached the 2,048-token cap.

| condition | macro AUROC | instructed | varied | unique scores |
| --- | ---: | ---: | ---: | ---: |
| frozen `0/1` | `0.94518` | `0.99417` | `0.87986` | 74 |
| explicit-semantics `0/1` | **`0.94732`** | `0.99365` | **`0.88556`** | 95 |
| `False/True` | `0.94250` | `0.99333` | `0.87472` | 83 |
| `No/Yes` | `0.94601` | `0.99396` | `0.88208` | 82 |
| `honest/deceptive` | `0.94179` | `0.99063` | `0.87667` | 89 |
| `A/B` | `0.94452` | `0.99167` | `0.88167` | 138 |
| expected `1--7` rating | `0.94345` | `0.99208` | `0.87861` | 820 |

The explicit digit contract improved macro AUROC by `0.00214`, with
seven unit wins, seven ties, and seven losses. A 10,000-replicate paired
within-dataset stratified bootstrap gave a 95% interval of
`[-0.00274, +0.00714]`, so the gain is directional rather than decisive.
`No/Yes` gained only `0.00083`; every other replacement regressed.

Conclusion: the base-model evidence continues to favor `0/1`. The useful
candidate change is to state their semantics explicitly before generating a
new teacher cache and matched student, not to replace the verbalizers. The
ordinal arm confirms that nearly tie-free scores alone do not imply better
ranking.

## Existing-adapter transfer

The explicit-semantics change was then tested directly on the already-trained
Phoenix v6.3 adapter:

```bash
.venv/bin/python \
  experiments/phoenix_verbalizer_sweep/run_ndif_base_verbalizers.py \
  --method phoenix_v6_3_explicit_digit_prompt_ablation_validation_v1 \
  --adapter-repo Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16 \
  --condition digits_frozen \
  --condition digits_explicit
```

The paired 822-row NDIF run kept the adapter, renderer, direct boundary, token
pair, row batches, and padded tensor widths fixed:

| v6.3 inference prompt | macro AUROC | instructed | varied | unique |
| --- | ---: | ---: | ---: | ---: |
| exact trained prompt | **`0.94970`** | **`0.99792`** | **`0.88542`** | 117 |
| explicit digit semantics | `0.94637` | `0.99490` | `0.88167` | 126 |

Adding the clause regressed macro AUROC by `0.00333`, with 2 unit wins, 10
ties, and 9 losses. The 10,000-replicate paired within-dataset stratified
bootstrap interval was `[-0.00762, +0.00030]`; only 3.84% of replicates had a
non-negative delta. The exact control's number is specific to this paired NDIF
execution and does not replace previously recorded vLLM validation figures.

Do not change the v6.3 submission prompt. The base model's small benefit did
not transfer through the adapter trained on the frozen prompt. If explicit
semantics are revisited, regenerate the teacher cache and train a matched
student rather than applying the clause only at inference.
