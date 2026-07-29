# Phoenix NDIF padding-free benchmark

This experiment tests whether padding-free inference can accelerate Phoenix's
direct `Prediction:` margin without changing its scores. It compares the
deployed length-sorted, left-padded `48/32/16` path against Qwen3.5's supported
packed path on the same prompts and the selected Phoenix 6.2 adapter.

Qwen3.5 is a hybrid full-attention/Gated-DeltaNet model. Resetting
`position_ids` is not sufficient to isolate packed examples. The packed trace
therefore supplies every boundary produced by
`DataCollatorWithFlattening(return_flash_attn_kwargs=True,
return_seq_idx=True)`:

- reset `position_ids` for rotary positions and full-attention isolation;
- `cu_seq_lens_q` and `cu_seq_lens_k` for variable-length FlashAttention and
  Gated DeltaNet;
- `seq_idx` for causal-convolution resets;
- no `attention_mask`, because it conflicts with the flat packed layout.

The label readout uses the tensor form of `logits_to_keep`: the final index of
each packed prompt is passed to the LM head, returning only the requested
`0|1` logit rows rather than logits for every token.

Run the focused 96-row parity smoke first:

```bash
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
HF_DATASETS_OFFLINE=1 \
  python experiments/phoenix_ndif_packing/run.py
```

If all scores are close and the packed arm is faster, run the frozen full
validation comparison:

```bash
HF_DATASETS_OFFLINE=1 \
  python experiments/phoenix_ndif_packing/run.py --limit 0
```

The default packed cap is 32,768 real tokens and 96 sequences per trace. This
matches the padded-token workload already exercised by the renderer-cap NDIF
smokes while approximately halving the number of traces on the 822-row local
validation shape. Results are written under
`results/blackbox/phoenix_ndif_packing_v1/` and remain ignored.

Do not migrate packing into the submission based on token counts alone.
Promotion requires complete score parity (within a predeclared small numerical
tolerance), no missing label logits, and a repeatable wall-time improvement on
competition NDIF. The existing batch-shape experiments already show small
numeric drift, so report the score deltas rather than claiming exact
invariance.

## Result

The 96-row competition-NDIF smoke rejected this path. Packing combined the
current two padded traces (`48x260` and `48x351`) into one flat 25,079-token
trace. It removed 4,249 padding tokens and reduced end-to-end wall time from
`8.37s` to `7.72s`, only a `1.08x` speedup.

The readouts were not equivalent:

| diagnostic | result |
| --- | ---: |
| scores equal within `1e-5` | `2 / 96` |
| mean absolute score difference | `0.12111` |
| maximum absolute score difference | `0.52512` |
| score Pearson correlation | `0.94279` |
| padded / packed macro AUROC | `0.99524 / 0.97619` |

This drift is far larger than the previously reported batch-shape variation.
The runner supplied all documented Qwen3.5 boundaries, so the result is
consistent with at least one hosted hybrid component not honoring packed
sequence isolation. Hugging Face explicitly warns that reference
Gated-DeltaNet and causal-convolution fallbacks can silently ignore boundary
kwargs. The aggregate smoke cannot identify whether the deployed leak is in
full attention, Gated DeltaNet, causal convolution, or their integration.

Do not deploy padding-free packing for Phoenix on the current NDIF backend.
Keep the frozen left-padded, length-sorted `48/32/16` path. A future retry
requires an organizer-confirmed packed Qwen3.5 deployment and must repeat this
parity gate before any runtime comparison.
