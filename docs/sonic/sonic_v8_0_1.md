# Sonic v8.0.1 — Phoenix v8.1 adapter transfer probe

Sonic v8.0.1 keeps the submitted Sonic v8.0 method fixed and replaces only its
Phoenix judge adapter:

- old: `Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2`
- new: `Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`

The new rank-16 adapter was trained for two epochs on the 6,573 Phoenix 8 rows
plus 6,576 label-balanced Liars' Bench rows. Its frozen Phoenix evaluation
showed the intended OOD tradeoff: Liars holdout macro AUROC improved from
`0.86255` to `0.93915`, while matched competition validation moved from
`0.96417` to `0.96214`. Phoenix v8.1 later scored `0.9661` macro AUROC on the
private leaderboard.

Everything else remains Sonic v8.0:

- dual L40+L46 activation probe and frozen per-family standardization;
- Phoenix 4.0 structural renderer;
- mutually exclusive HP-KR, action, and ordinary direct-margin prompts;
- literal next-token `0|1` softmax score;
- one rank-16 adapter session with length-aware `48/32/16` batching;
- v4 sign gate with one-step disagreement and four-step agreement caps;
- secondary binary threshold `0.5`.

This is deliberately not recalibrated to the v8.1 judge. In particular,
`JUDGE_LOGIT_MEAN=0.0` and `JUDGE_LOGIT_SD=5.638` remain the values used by
Sonic v8.0, where the scale was measured on 14 synthetic prompts with the v8.0
adapter. The notebook logs that limitation and retains the environment
overrides for later diagnostics. The purpose of v8.0.1 is to measure the
adapter swap on the main leaderboard without introducing a second gate change.

The active notebook is `submission/sonic_v8_0_1.ipynb` and must be submitted
with `--tag white`.
