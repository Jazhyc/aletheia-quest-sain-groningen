# Ideas tested and rejected (or deferred)

Dead ends and untestable ideas explored during the sonic development.
Recorded so we don't retread the same ground.

## Token-pair ensemble (2026-07-27) — REJECTED

**Idea:** Read logits for 19 semantically equivalent token pairs
(0/1, No/Yes, false/true, etc.) at the `Prediction:` position in one
forward pass and average their margins. Theory: each pair is a different
linear readout `(W_pos − W_neg)·h + Δb` of the same hidden state `h`;
averaging reduces variance if noise across pairs is uncorrelated.

**Tested locally:** Qwen3.5-9B + LoRA adapter on RTX 4090, 30+30 balanced
sample from `dev-instructed-deception-Nemotron`.

**Result:**

```
Single pair (0/1):  AUROC = 0.9677
19-pair ensemble:   AUROC = 0.3833  (−0.5844)
```

11 of 18 alternative pairs are **anti-correlated** with the "0"/"1" margin.
The model was prompted to output "0" or "1" — those are the only tokens
that carry the deception direction. The other pairs encode orthogonal or
opposing features in the residual stream.

**Why it fails:** Each token's logit is `W_token · h + b_token`. The
unembedding vectors `W_token` for "Yes", "true", "correct", "real" etc.
point in different directions from `W_1`. Averaging them with `W_1`
dilutes the signal. The anti-correlated pairs actively cancel it.

**Verdict:** Actively harmful. Do not revisit.

## Chain-of-thought on uncertain rows (2026-07-27) — UNTESTABLE LOCALLY

**Idea:** On rows where the direct read margin is small
(`|judge_margin| < 1.2`), strip `Prediction:` from the prompt and let the
model generate a full reasoning chain (max 80 tokens). Parse
`Prediction:<0 or 1>` from the output and shift the margin toward the
reasoned prediction by ±1.2. Confident rows skip generation entirely.

**Why it might work:** The direct read extracts the model's immediate
belief at the `Prediction:` token. But the model was instructed to output
reasoning first. Generation lets the model run its full chain-of-thought,
updating the residual stream with each token. The final hidden state
encodes a more thorough comparison.

**Local test:** Judge AUROC is 0.995 on instruct data. 95.5% of rows fall
below the uncertainty threshold but 95.2% of those already have correct
sign. Only 3/66 rows could potentially be fixed. The real value would be
on out-of-domain Notus data (base-model, where the judge is weaker) —
held-out, no local access.

**Budget:** ~3–5 extra minutes. 80 autoregressive steps per batch of
uncertain rows, KV-cached. Fits within the 30-min budget in theory.

**Risks:**
- JUDGE_MARGIN_SD would need recalibration for the ensemble margin
- CoT output format is fragile (regex `Prediction:\s*([01])`)
- A second remote session adds queue latency on NDIF
- Can't test end-to-end before submission

**Verdict:** Deferred. Not in any submission. Worth revisiting if we get
access to Notus-like data for local testing, or if the judge AUROC
plateaus and we need a new lever.
