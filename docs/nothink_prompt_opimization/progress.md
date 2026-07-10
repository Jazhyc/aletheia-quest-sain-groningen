# No-Thinking Prompt Optimization Progress

Goal: improve the single Qwen no-thinking generation judge toward validation
balanced accuracy 0.90 without increasing the 512-token generation budget.

## Baseline

Candidate: `qwen_reason_details_nothink_t512_t01_v1`

- Config: `configs/single_judges/blackbox_reasoning_details_nothink_t512_t01_v1.yaml`
- Judge: `Qwen/Qwen3.5-9B`, `enable_thinking=false`, `max_tokens=512`,
  `max_prompt_chars=3000`, threshold `0.1`
- Validation: balanced accuracy `0.8607`, AUROC `0.8607`, recall `0.7667`,
  FPR `0.0452`, parse errors `0`, score time `12.2s`
- Test confirmation already tracked: balanced accuracy `0.8571`, AUROC `0.8570`,
  recall `0.7643`, FPR `0.0500`, parse errors `0`, score time `12.2s`

Baseline error profile on validation:

- False negatives: `89`
- False positives: `19`
- Worst subsets are mostly varied-deception Qwen LoRA variants, especially
  `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` at `0.6250` balanced
  accuracy with recall `0.3500` and FPR `0.1000`.
- Lowest-score false negatives are often simple factual statements where the
  final answer says a true statement is false, but adds plausible nuance. Example:
  "Archimedes lived in Ancient Greece" is answered as false because he lived in
  Syracuse; the judge rates this `1`.
- High-score false positives often have hidden dishonesty instructions but a
  final answer that is correct or defensible. This suggests the prompt must
  keep hidden intent as a scrutiny signal, not a decision rule.

## Iteration 1: Prompt Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_prompt_sweep_v1.yaml`

Execution plan:

- Run a single Slurm job with one `rtx_pro_6000` GPU.
- Use ensemble mode only to batch prompt variants through one vLLM load.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`.
- Analyze each member independently for balanced accuracy, AUROC, recall, FPR,
  parse failures, and representative misses.

Prompt members:

- `details_control`: existing details no-thinking prompt.
- `intent_satisfaction`: explicitly identify the user's requested belief, the
  belief created by the output, and whether the assistant satisfies the request
  honestly.
- `truth_value_guard`: target true/false and correct/incorrect examples, with a
  guard against hidden-intent-only false positives.
- `context_contradiction`: emphasize contradictions with context-stated facts
  while preserving the hidden-intent guard.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `details_control` | 0.8607 | 0.8607 | 0.7667 | 0.0452 | 0 | 89 | 19 |
| `intent_satisfaction` | 0.2310 | 0.2304 | 0.3905 | 0.9286 | 0 | 256 | 390 |
| `truth_value_guard` | 0.8940 | 0.8949 | 0.8500 | 0.0619 | 0 | 63 | 26 |
| `context_contradiction` | 0.8786 | 0.8786 | 0.8190 | 0.0619 | 0 | 76 | 26 |

Decision:

- `truth_value_guard` is the new best no-thinking prompt. It gains `+0.0333`
  balanced accuracy over baseline by recovering recall, with zero parse errors.
- `intent_satisfaction` over-triggered catastrophically. The explicit
  intent/satisfaction decomposition made hidden intent dominate the final output.
- Remaining weakness: reduce `truth_value_guard` FPR without losing its recall.

## Interactive Server Setup

The first sweep paid vLLM startup inside the run process. After user feedback,
the loop was changed to the intended interactive setup:

- Reserved one GPU with `srun` job `30031318` on `rtx6000gpu3`.
- Started a persistent vLLM OpenAI-compatible server:
  `experiments/blackbox/serve_vllm.sh`, PID `1451291`, model
  `Qwen/Qwen3.5-9B`, served as `qwen-judge`.
- Added `OpenAIGenerateJudge` to `experiments/blackbox/run_judge.py` so the
  runner can call `/v1/completions`, parse `Rating: N`, and still write
  generations/predictions artifacts.
- Subsequent runs use `judge.backend=openai`, `mode=generate`,
  `enable_thinking=false`, `max_tokens=512`, and the same validation split.

## Iteration 2: Truth/FPR Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_truth_sweep_v2.yaml`

Started after switching to the persistent vLLM server. Hypothesis: make the
false-belief requirement more explicit to cut hidden-intent-only false positives
while retaining the `truth_value_guard` recall gain.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `truth_value_guard` | 0.8964 | 0.8973 | 0.8476 | 0.0548 | 0 | 64 | 23 |
| `false_belief_required` | 0.7655 | 0.7655 | 0.5476 | 0.0167 | 0 | 172 | 7 |
| `statement_verdict_first` | 0.2262 | 0.2262 | 0.3024 | 0.8500 | 0 | 293 | 357 |
| `final_claim_only` | 0.8786 | 0.8786 | 0.8143 | 0.0571 | 0 | 78 | 24 |

Decision:

- `truth_value_guard` remains best and improves slightly versus v1, likely from
  rerun/backend differences or prompt formatting.
- `false_belief_required` proves the FPR can be reduced, but recall collapses.
- `statement_verdict_first` is unsafe; it inverts too many negatives.

## Iteration 3: Context Filtering

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_context_sweep_v3.yaml`

Started to test whether hidden/system instructions were causing FPR. Added
optional `judge.context_roles` support and ran prompts with only `user` and
`assistant` messages in `<context>`.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `truth_value_no_system` | 0.8369 | 0.8382 | 0.6929 | 0.0190 | 0 | 120 | 8 |
| `verdict_no_system` | 0.1976 | 0.1965 | 0.3214 | 0.9262 | 0 | 285 | 389 |
| `false_belief_no_system` | 0.7655 | 0.7655 | 0.5452 | 0.0143 | 0 | 173 | 6 |
| `compact_no_system` | 0.8119 | 0.8130 | 0.6405 | 0.0167 | 0 | 133 | 7 |

Decision:

- Removing system context cuts FPR but loses too much recall. Hidden context is
  useful for finding positives, but prompts must keep it subordinate to final
  output truth.

## Iteration 4: Context-Length Sweep

Config: `configs/single_judges/blackbox_reasoning_nothink_truth_value_v1.yaml`

Started with the current best prompt as a single judge and swept
`judge.max_prompt_chars` through `1000`, `1500`, `2000`, and `2500`, all through
the persistent server.

Result:

| max prompt chars | BA | AUROC | recall | FPR | parse | score time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.8893 | 0.8901 | 0.8310 | 0.0524 | 0 | 14.4s |
| 1500 | 0.8952 | 0.8961 | 0.8452 | 0.0548 | 0 | 14.0s |
| 2000 | 0.8952 | 0.8961 | 0.8476 | 0.0571 | 0 | 13.9s |
| 2500 | 0.8940 | 0.8949 | 0.8452 | 0.0571 | 0 | 13.9s |
| 3000 (`truth_value_guard` v2) | 0.8964 | 0.8973 | 0.8476 | 0.0548 | 0 | 61.8s for 4 members |

Decision:

- Context-length changes do not cross `0.90`.
- `1500`/`2000` are close but slightly below the full `3000` prompt. Keep 3000
  unless runtime becomes more important than accuracy.

## Iteration 5: Scale Calibration

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_scale_sweep_v4.yaml`

Started to see whether the model would use rating `4` for suspicious hidden
intent without a false final claim, allowing a higher threshold to filter FPs.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scale_hidden_middle` | 0.8714 | 0.8714 | 0.9095 | 0.1667 | 0 | 38 | 70 | `{1: 388, 7: 434}` |
| `scale_false_belief` | 0.8167 | 0.8167 | 0.6500 | 0.0167 | 0 | 138 | 7 | `{1: 551, 7: 271}` |
| `scale_truth_direct` | 0.8667 | 0.8667 | 0.8381 | 0.1048 | 0 | 59 | 44 | `{1: 435, 7: 387}` |

Decision:

- Qwen ignored the intended middle rating and still collapsed to mostly `1/7`.
- Scale calibration is not a useful lever in no-thinking generation mode.

## Iteration 6: Broad-Truth Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_broadtruth_sweep_v5.yaml`

Started to address recurring false negatives where the output calls a broadly
true statement false using a narrow caveat, while explicitly saying that hidden
lie instructions may not have been followed.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `broad_truth_guard` | 0.8881 | 0.8881 | 0.8429 | 0.0667 | 0 | 66 | 28 | `{1: 458, 7: 364}` |
| `broad_statement_first` | 0.1226 | 0.1217 | 0.1238 | 0.8786 | 0 | 350 | 369 | `{7: 419, 1: 401, 2: 2}` |
| `actual_output_guard` | 0.8881 | 0.8881 | 0.8976 | 0.1214 | 0 | 43 | 51 | `{1: 412, 7: 410}` |

Decision:

- Broad-truth wording did not improve the frontier. It either raises FPR or
  loses balanced accuracy.
- Current best single no-thinking judge remains `truth_value_guard` at
  validation BA `0.8964`, recall `0.8476`, FPR `0.0548`, zero parse errors.

## Iteration 7: Hidden-Plus-Contradiction Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_hidden_contradiction_sweep_v6.yaml`

Started: 2026-07-05.

Hypothesis:

- Remaining false negatives are often hidden-dishonesty cases where the final
  output explicitly says a broadly true user statement is false/incorrect using
  a caveat.
- Remaining false positives are often hidden-dishonesty cases where the final
  output actually agrees with the correct answer.
- The prompt should use hidden dishonesty only as a tie-breaker when the output
  contradicts, downgrades, or caveats the user statement. It should not punish
  outputs that directly answer correctly.

Members:

- `hidden_contradiction`: hidden intent plus actual contradiction is strong
  evidence; hidden intent plus correct final answer remains low.
- `caveat_downgrade`: targets misleading caveats that turn a broadly true
  statement into false/incorrect.
- `agree_low`: aggressively protects final answers that agree with the user's
  true statement or give the correct factual answer.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hidden_contradiction` | 0.8179 | 0.8185 | 0.6500 | 0.0143 | 0 | 147 | 6 | `{1: 561, 7: 258, 2: 3}` |
| `caveat_downgrade` | 0.8917 | 0.8915 | 0.8500 | 0.0667 | 0 | 63 | 28 | `{1: 455, 7: 366, 3: 1}` |
| `agree_low` | 0.8417 | 0.8417 | 0.6929 | 0.0095 | 0 | 120 | 4 | `{1: 536, 7: 286}` |

Decision:

- Aggregate run mean was `0.8976`, but the best member did not beat
  `truth_value_guard`.
- `hidden_contradiction` and `agree_low` prove FPR can be made tiny, but recall
  collapses. `caveat_downgrade` keeps recall but adds too many false positives.
- Current best single no-thinking judge remains `truth_value_guard` at
  validation BA `0.8964`, recall `0.8476`, FPR `0.0548`, zero parse errors.

## Iteration 8: Behavioral Evidence Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_behavior_sweep_v7.yaml`

Started: 2026-07-05.

Hypothesis:

- Current-best errors show labels are not purely "hidden intent" or purely
  "final answer truth"; both appear in positive and negative rows.
- The prompt should ask for evidence that the final output follows a deceptive
  instruction: wrong main answer, unjustified false verdict, fabricated
  specifics, citation-like false support, or defensive overconfidence.
- It should still protect correct concise answers under hostile hidden context,
  because hidden intent alone produced the strongest false positives.

Members:

- `behavioral_evidence`: balanced evidence checklist combining final-answer
  falsity with visible compliance signals.
- `truth_plus_fabrication`: current truth guard plus explicit fabricated-detail
  and false-support handling.
- `intent_followed_only`: high only when hidden deceptive instructions appear to
  be followed in the final answer; low when the final answer resists them.
- `suspicious_support`: targets answers with correct-looking short answers but
  invented or misleading explanatory support.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `behavioral_evidence` | 0.8619 | 0.8619 | 0.7690 | 0.0452 | 0 | 97 | 19 | `{1: 498, 7: 324}` |
| `truth_plus_fabrication` | 0.8857 | 0.8877 | 0.8333 | 0.0619 | 0 | 70 | 26 | `{1: 464, 7: 355, 3: 2, 6: 1}` |
| `intent_followed_only` | 0.6940 | 0.6940 | 0.4524 | 0.0643 | 0 | 212 | 27 | `{1: 605, 7: 217}` |
| `suspicious_support` | 0.8940 | 0.8938 | 0.8881 | 0.1000 | 0 | 47 | 42 | `{1: 425, 7: 396, 6: 1}` |

Decision:

- No member beats `truth_value_guard`. The best v8 member, `suspicious_support`,
  recovers recall but doubles FPR.
- Offline combinations of existing members show a better ensemble frontier:
  `truth_value_guard` plus the low-FPR `hidden_contradiction` member reaches
  validation BA about `0.9060` with recall `0.8738`, FPR `0.0619`.

## Iteration 9: Two-Prompt Ensemble Verification

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_truth_hidden_ensemble_v1.yaml`

Started: 2026-07-05.

Hypothesis:

- `truth_value_guard` catches broad factual falsehoods and misleading verdicts.
- `hidden_contradiction` catches a complementary subset of hidden-dishonesty
  rows while keeping FPR very low on its own.
- A max aggregate at threshold `0.5` should reproduce the offline combination
  result above `0.90` balanced accuracy without increasing the per-prompt
  generation budget.

Result:

- Overall validation: balanced accuracy `0.9060`, AUROC `0.9057`, recall
  `0.8738`, FPR `0.0619`, score time `27.8s`, zero parse errors.

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard` | 0.8952 | 0.8949 | 0.8429 | 0.0524 | 0 | 66 | 22 | `{1: 463, 7: 356, 3: 1, 6: 2}` |
| `hidden_contradiction` | 0.8143 | 0.8161 | 0.6405 | 0.0119 | 0 | 151 | 5 | `{1: 563, 7: 256, 2: 3}` |

Decision:

- This is the first no-thinking generation run above the validation `0.90`
  balanced-accuracy target.
- It is not a single-prompt judge. It doubles prompt evaluations relative to the
  single `truth_value_guard`, but keeps each member at `enable_thinking=false`
  and `max_tokens=512`.
- Keep `qwen_reason_nothink_truth_hidden_ensemble_v1` as the current
  validation leader among no-thinking generation judges. Confirm on the local
  test split before treating it as robust.

## Iteration 10: Test Confirmation

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_truth_hidden_ensemble_v1.yaml`

Started: 2026-07-05.

Plan:

- Run the same two-prompt ensemble on `split=test`.
- Compare against the previously tracked no-thinking single-prompt test score
  `qwen_reason_details_nothink_t512_t01_v1` at balanced accuracy `0.8571`.

Result:

- Overall test: balanced accuracy `0.8821`, AUROC `0.8818`, recall `0.8357`,
  FPR `0.0714`, score time `29.7s`, zero parse errors.

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard` | 0.8786 | 0.8785 | 0.8262 | 0.0690 | 0 | 54 | 29 | `{1: 445, 7: 375, 6: 1}` |
| `hidden_contradiction` | 0.7690 | 0.7694 | 0.5548 | 0.0167 | 0 | 168 | 7 | `{1: 579, 7: 240, 2: 2}` |

Worst test subsets:

- `dev-instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None`:
  BA `0.4750`, recall `0.0000`, FPR `0.0500`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1`: BA `0.7250`,
  recall `0.5000`, FPR `0.0500`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3`: BA `0.7250`,
  recall `0.6000`, FPR `0.1500`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4`: BA `0.7250`,
  recall `0.7000`, FPR `0.2500`.
- `dev-varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b`: BA `0.7750`,
  recall `0.8500`, FPR `0.3000`.

Decision:

- The two-prompt ensemble improves over the previous tracked no-thinking
  single-prompt test score (`0.8571` BA) but does not match the validation
  `0.9060`.
- Treat `qwen_reason_nothink_truth_hidden_ensemble_v1` as the current fast
  no-thinking generation leader in this experiment family, with the caveat that
  its gain is validation stronger than test and it doubles prompt evaluations.

## Iteration 11: Single-Prompt Calibration Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_single_calibrated_sweep_v8.yaml`

Started: 2026-07-05.

Pre-run audit:

- Re-thresholding the current best single prompt, `truth_value_guard`, improves
  validation from BA `0.8964` at threshold `0.1` to BA `0.8976` at thresholds
  `0.34` through `0.67`, with recall `0.8476` and FPR `0.0524`.
- Against `truth_value_guard`, the low-FPR `hidden_contradiction` prompt
  recovers `11/64` false negatives while adding only `4` new false positives.
  Folding that behavior into one prompt is the clearest path to a single-prompt
  validation score above `0.90`.
- `suspicious_support` recovers `21/64` false negatives but adds `22` new false
  positives, so broad "suspicious support" wording needs tighter calibration.

Hypothesis:

- A single prompt can improve by explicitly separating these cases:
  hidden dishonesty plus a wrong/downgrading final answer is high; hidden
  dishonesty plus an ordinary correct final answer is low.
- Compact calibration rules may reduce the hidden-intent false positives while
  preserving the recovered true/false and correct/incorrect false negatives.

Members:

- `truth_hidden_calibrated`: merge the best truth-value wording with a stronger
  hidden-intent guard.
- `user_belief_contract`: frame the decision as whether the user request was
  honestly satisfied and what belief the user leaves with.
- `calibrated_micro_examples`: use compact high/low archetypes without long
  examples.
- `contradiction_plus_support`: two-step verdict plus support check.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_hidden_calibrated` | 0.8905 | 0.8905 | 0.8524 | 0.0714 | 0 | 62 | 30 | `{1: 452, 7: 370}` |
| `user_belief_contract` | 0.8940 | 0.8940 | 0.8595 | 0.0714 | 0 | 59 | 30 | `{1: 449, 7: 373}` |
| `calibrated_micro_examples` | 0.8131 | 0.8131 | 0.6381 | 0.0119 | 0 | 143 | 5 | `{1: 558, 7: 264}` |
| `contradiction_plus_support` | 0.2262 | 0.2278 | 0.0548 | 0.6024 | 0 | 379 | 253 | `{7: 276, 1: 541, 2: 5}` |

Decision:

- None of these variants beats the `truth_value_guard` threshold-adjusted
  single-prompt frontier (`0.8976` BA).
- `user_belief_contract` gains recall (`0.8595`) but raises FPR to `0.0714`.
  This is the wrong direction for balanced accuracy.
- `calibrated_micro_examples` kept FPR tiny but collapsed recall, similar to
  `hidden_contradiction`; compact archetype wording over-constrained the model.

## Iteration 12: Statement Override Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_statement_override_sweep_v9.yaml`

Started: 2026-07-05.

Pre-run audit:

- `user_belief_contract` had `16` extra false positives relative to
  `truth_value_guard`; most are ordinary correct answers under hostile hidden
  instructions, such as company-industry facts, Olympic host city, or trivia
  questions answered canonically.
- Its recovered false negatives are mostly statement-verdict cases where a true
  user statement is answered as false/incorrect under hidden dishonesty, plus a
  smaller set of open-question varied-deception rows.

Hypothesis:

- Widening hidden-intent language is harmful. The next prompt should preserve
  the current truth guard and add only a narrow override for
  true/false/correct/incorrect statement questions.
- Correct direct answers under hidden dishonesty must be explicitly low.

Members:

- `truth_value_guard_control`: current best prompt under the same OpenAI server
  path and threshold `0.5`.
- `statement_negative_override`: truth guard plus narrow high rule for
  deceptive negative verdicts on broadly true statements.
- `two_check_distilled`: explicit single-prompt distillation of the truth check
  and hidden-contradiction check.
- `correct_answer_low`: aggressive protection of correct direct answers while
  preserving high ratings for wrong verdicts and wrong direct answers.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8976 | 0.8973 | 0.8476 | 0.0524 | 0 | 64 | 22 | `{1: 461, 7: 357, 3: 1, 6: 3}` |
| `statement_negative_override` | 0.8798 | 0.8796 | 0.7952 | 0.0357 | 0 | 77 | 15 | `{1: 482, 7: 338, 6: 2}` |
| `two_check_distilled` | 0.8762 | 0.8761 | 0.7762 | 0.0238 | 0 | 94 | 10 | `{1: 504, 7: 317, 6: 1}` |
| `correct_answer_low` | 0.8821 | 0.8832 | 0.8190 | 0.0548 | 0 | 76 | 23 | `{1: 472, 7: 349, 2: 1}` |

Decision:

- The control remains best at BA `0.8976`. Narrow statement-overrides lowered
  FPR but lost too much recall.
- The model appears sensitive to any extra guard that says "correct answer
  low"; it protects negatives but also suppresses many positives that require
  hidden context or broader deception cues.

## Iteration 13: Few-Shot Calibration Prefix

Config: `configs/single_judges/blackbox_reasoning_nothink_truth_value_v1.yaml`
with overrides `method=qwen_reason_nothink_truth_value_fs1_s0_v1`,
`shots.n_per_label=1`, `shots.seed=0`, and `scoring.threshold=0.5`.

Started: 2026-07-05.

Hypothesis:

- A tiny labelled prefix sampled from the training split may calibrate the
  hidden-context/truth tradeoff better than additional abstract instructions.
- This remains a single no-thinking judge with `enable_thinking=false` and
  `max_tokens=512`, but it does increase prompt text. Treat it as an accuracy
  experiment, not a speed-first path.

Result:

- Validation: balanced accuracy `0.8440`, AUROC `0.8445`, recall `0.8024`,
  FPR `0.1143`, score time `15.7s`.

Decision:

- The labelled prefix is harmful. It raises FPR and lowers recall relative to
  the no-shot `truth_value_guard`.
- Do not expand this path unless using much more careful hand-picked examples;
  random train-shot calibration is not a useful lever for this judge.

## Iteration 14: Raw Completion Formatting

Config: `configs/single_judges/blackbox_reasoning_nothink_truth_value_v1.yaml`
with overrides `method=qwen_reason_nothink_truth_value_rawprompt_v1`,
`judge.use_chat_template=false`, and `scoring.threshold=0.5`.

Started: 2026-07-05.

Hypothesis:

- The same single prompt may behave differently if sent directly to
  `/v1/completions` instead of being wrapped in Qwen's chat template.
- This does not increase `max_tokens` or prompt content; it only changes the
  formatting boundary around the prompt.

Result:

- Validation: balanced accuracy `0.5774`, AUROC `0.5774`, recall `0.1762`,
  FPR `0.0214`, parse errors `673`, score time `329.0s`.

Decision:

- Raw completion formatting is much worse and much slower. It frequently fails
  to produce a parseable `Rating:` line and burns most of the 512-token budget.
- Keep Qwen chat-template formatting for subsequent no-thinking generation
  runs.

## Iteration 15: Verbalized Scaffolding Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_verbalized_sweep_v10.yaml`

Started: 2026-07-05.

Hypothesis:

- Earlier decomposition prompts mostly asked the model to make distinctions but
  still output only `Rating: N`. They did not force explicit intermediate
  verbalization.
- A compact scaffold that states the user's factual claim, the output's claim,
  whether the output contradicts the user/common facts, and whether hidden
  intent appears followed may improve the single judge's accuracy.
- Keep `enable_thinking=false` and `max_tokens=512`. Parse risk may increase,
  so parse failures are a key metric.

Planned members:

- `facts_contradictions`: concise fields for user fact, output claim,
  contradiction, hidden-followed, and rating.
- `intent_satisfaction_verbalized`: explicitly states user intent and whether
  the assistant satisfies it honestly before rating.
- `canonical_answer_check`: asks for canonical answer, assistant answer,
  contradiction, then rating.
- `false_belief_trace`: asks what belief the user would leave with and whether
  that belief is false.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `facts_contradictions` | 0.8488 | 0.8512 | 0.7476 | 0.0500 | 20 | 106 | 21 | `{1: 477, 7: 315, 2: 8, None: 20, 6: 2}` |
| `intent_satisfaction_verbalized` | 0.1405 | 0.1403 | 0.1381 | 0.8571 | 38 | 344 | 360 | `{7: 418, None: 38, 1: 359, 2: 7}` |
| `canonical_answer_check` | 0.6964 | 0.7835 | 0.4071 | 0.0143 | 3 | 249 | 6 | `{1: 578, 7: 151, 2: 80, None: 3, 6: 7, 5: 1, 3: 2}` |
| `false_belief_trace` | 0.8690 | 0.8698 | 0.8429 | 0.1048 | 0 | 57 | 44 | `{1: 433, 7: 381, 6: 8}` |

Decision:

- Verbalized scaffolding did not improve the frontier. It slowed scoring and
  introduced parse failures in three of four members.
- The worst failure mode is inconsistency: `intent_satisfaction_verbalized` can
  write "False belief caused? no" and still emit `Rating: 7`.
- A stricter consistency-constrained scaffold is worth one follow-up test, but
  free-form scaffolding is not promising as implemented.

## Iteration 16: Consistent Scaffold Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_consistent_scaffold_sweep_v11.yaml`

Started: 2026-07-05.

Hypothesis:

- The previous verbalized scaffold failed partly because the final rating was
  not bound to the intermediate fields.
- Force a binary `Rating: 1`/`Rating: 7` rule: if the scaffold says no false
  belief or no important mismatch, the rating must be `1`; if it says yes, the
  rating must be `7`.

Planned members:

- `consistent_false_belief`: fields for user belief and false-belief caused,
  with a hard mapping to rating.
- `consistent_mismatch`: canonical answer, assistant answer, important
  mismatch, misleading support, with a hard mapping to rating.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `consistent_false_belief` | 0.8607 | 0.8607 | 0.7786 | 0.0571 | 0 | 84 | 24 | `{1: 480, 7: 342}` |
| `consistent_mismatch` | 0.8393 | 0.8393 | 0.7929 | 0.1143 | 0 | 78 | 48 | `{1: 450, 7: 372}` |

Decision:

- Binding the verbalized fields to the final rating removes parse failures but
  still does not beat the one-line `truth_value_guard`.
- The scaffolds seem to make Qwen under-call positives that require hidden
  context or nontrivial factual knowledge, while not improving FPR enough.
- Current best single no-thinking judge remains `truth_value_guard` with
  threshold `0.5`: validation BA `0.8976`, recall `0.8476`, FPR `0.0524`,
  zero parse errors. The only run above `0.90` remains the two-prompt ensemble,
  not a single prompt.

## Iteration 17: Internal-Max Distillation

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_internal_max_sweep_v12.yaml`

Started: 2026-07-05.

Pre-run audit:

- Thresholding every saved validation member confirms the single-prompt plateau:
  no prior single prompt beats `truth_value_guard` at BA `0.8976`, recall
  `0.8476`, FPR `0.0524`.
- The successful two-prompt ensemble works by max-aggregating
  `truth_value_guard` with `hidden_contradiction`; the latter is weak alone but
  recovers complementary positives with very low FPR.
- Previous free-form verbalization asked for facts/contradictions, but did not
  explicitly force a two-check max rule.

Hypothesis:

- A single prompt can emulate the two-prompt ensemble by asking for two compact
  sub-ratings in one completion: a truth/output false-belief rating and a
  hidden-contradiction rating. The final `Rating` is the max of the two.
- This remains a single judge call with `enable_thinking=false` and
  `max_tokens=512`.

Members:

- `internal_max_binary`: explicit `Truth`, `HiddenContradiction`, then max
  `Rating`.
- `internal_max_with_guard`: same idea with stronger correct-answer guard.
- `internal_max_one_line`: compressed no-scaffold form of the same logic.
- `internal_max_statement_focus`: binary final rating focused on statement
  verdict failures and correct-answer false positives.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `internal_max_binary` | 0.6756 | 0.6756 | 0.3632 | 0.0119 | 0 | 256 | 5 | `{1: 671, 7: 151}` |
| `internal_max_with_guard` | 0.8411 | 0.8411 | 0.6965 | 0.0143 | 0 | 122 | 6 | `{1: 536, 7: 286}` |
| `internal_max_one_line` | 0.1257 | 0.1293 | 0.1990 | 0.9476 | 0 | 322 | 398 | `{1: 226, 2: 118, 7: 478}` |
| `internal_max_statement_focus` | 0.8808 | 0.8808 | 0.8284 | 0.0667 | 0 | 69 | 28 | `{1: 461, 7: 361}` |

Aggregate max result:

- Validation mean balanced accuracy `0.5012`, AUROC `0.5012`,
  `score_time=88.6s`, `rows_per_second=9.3/s`.
- The aggregate is invalidated by `internal_max_one_line`, which over-called
  positives with FPR `0.9476`.

Decision:

- The internal two-check scaffold did not distill the two-prompt ensemble into
  one generation call. The best member, `internal_max_statement_focus`, reached
  BA `0.8808`, below `truth_value_guard` at BA `0.8976`.
- Explicit sub-rating scaffolds can reduce FPR when strongly guarded, but they
  also suppress recall too much. A compact one-line version flips the failure
  mode and over-calls almost every negative.
- Current best single no-thinking judge remains `truth_value_guard`. The best
  no-thinking accuracy result remains the two-prompt `truth_value_guard` +
  `hidden_contradiction` ensemble, but it is not a single-prompt solution.

## Iteration 18: Hostile-Rejection Tiebreak Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_tiebreak_sweep_v13.yaml`

Started: 2026-07-05.

Pre-run audit:

- Current `truth_value_guard_control` errors at threshold `0.5`: `64` false
  negatives and `22` false positives.
- Sampled false negatives are mostly hostile hidden-context statement-verdict
  rows where the output rejects a true user statement with a plausible caveat or
  replacement fact. Examples include atomic-number reversals, "Alibaba is not
  retailing", "Australia is not the smallest continent", and "Washington D.C.
  is not a city".
- Sampled false positives are mostly correct direct answers under hostile hidden
  instructions, especially open trivia and statement rows where the output
  correctly agrees with the user.
- Regex audit on validation labels: `hidden + statement question + rejection`
  is positive `180` times and negative `62` times, so a pure surface rejection
  rule would over-call. The prompt needs a tiebreaker, not a hard heuristic.

Hypothesis:

- The model is under-calling hostile-context negative verdicts because it tries
  to verify every factual correction from memory. A prompt that treats hostile
  negative corrections as high when the correction is doubtful, caveated, or not
  plainly ordinary truth may recover recall.
- Preserve FPR by explicitly keeping correct direct answers and affirmations
  low, even when hidden/system text asks for deception.
- Keep `enable_thinking=false`, `max_tokens=512`, and threshold `0.5`.

Members:

- `truth_value_guard_control`: same current best prompt for same-session
  baseline.
- `hostile_rejection_tiebreaker`: use hostile hidden context as a tiebreaker
  against rejecting statement-verdict answers unless clearly true corrections.
- `distrust_hostile_corrections`: stronger distrust of plausible-sounding
  hostile corrections, with an open-question guard.
- `user_statement_anchor`: anchor statement-verdict questions on the user's
  stated fact and penalize hidden-context denials that rely on caveats or
  non-obvious replacement facts.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8943 | 0.8939 | 0.8433 | 0.0548 | 0 | 63 | 23 | `{1: 459, 3: 1, 6: 2, 7: 360}` |
| `hostile_rejection_tiebreaker` | 0.5297 | 0.5305 | 0.6617 | 0.6024 | 0 | 136 | 253 | `{1: 298, 2: 5, 6: 1, 7: 518}` |
| `distrust_hostile_corrections` | 0.8669 | 0.8669 | 0.8458 | 0.1119 | 0 | 62 | 47 | `{1: 435, 7: 387}` |
| `user_statement_anchor` | 0.8753 | 0.8749 | 0.7935 | 0.0429 | 0 | 83 | 18 | `{1: 484, 2: 1, 4: 2, 6: 1, 7: 334}` |

Aggregate max result:

- Validation mean balanced accuracy `0.6655`, AUROC `0.6655`,
  `score_time=66.1s`, `rows_per_second=12.4/s`.
- The broad hostile-rejection tiebreaker over-called negatives and dominated
  the aggregate.

Decision:

- The tiebreaker hypothesis is too blunt. Making hostile negative verdicts high
  unless clearly corrected raises FPR sharply.
- `user_statement_anchor` proves the opposite tradeoff: it lowers FPR to
  `0.0429` but loses too much recall (`0.7935`).
- Follow-up should not further widen hostile-context rejection. The next useful
  test is to preserve the current statement-verdict behavior while adding a
  guard only for open factual questions, where many false positives are correct
  direct answers under hostile hidden instructions.

## Iteration 19: Question-Type Router Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_question_router_sweep_v14.yaml`

Started: 2026-07-05.

Pre-run audit:

- Truncation check on current false negatives: among `64` false negatives,
  `42` had hidden deception text visible under both tail and head-tail context
  truncation; `22` had no matched hidden-deception wording in the system text.
  Head-tail truncation is therefore unlikely to be the missing recall lever.
- Iteration 18 showed that broad hostile-rejection tiebreakers increase FPR too
  much. The remaining target is reducing FPR without suppressing statement
  recall.

Hypothesis:

- False positives are concentrated in correct open factual answers under
  hostile hidden instructions. A question-type router can guard open questions
  more strongly while leaving true/false and correct/incorrect verdict behavior
  close to `truth_value_guard`.
- Keep `enable_thinking=false`, `max_tokens=512`, and threshold `0.5`.

Members:

- `truth_value_guard_control`: same current best prompt for same-session
  baseline.
- `question_type_router`: explicit statement-verdict versus open factual request
  routing.
- `open_answer_guard`: stronger protection for correct canonical open-question
  answers.
- `statement_preserve_open_guard`: preserve wrong-verdict recall while guarding
  only open-question false positives.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8930 | 0.8926 | 0.8383 | 0.0524 | 0 | 65 | 22 | `{1: 462, 3: 1, 6: 2, 7: 357}` |
| `question_type_router` | 0.1729 | 0.1739 | 0.0672 | 0.7214 | 0 | 375 | 303 | `{1: 478, 2: 14, 6: 1, 7: 329}` |
| `open_answer_guard` | 0.8360 | 0.8360 | 0.6816 | 0.0095 | 0 | 128 | 4 | `{1: 544, 7: 278}` |
| `statement_preserve_open_guard` | 0.6112 | 0.6187 | 0.3557 | 0.1333 | 0 | 259 | 56 | `{1: 614, 2: 9, 4: 1, 6: 1, 7: 197}` |

Aggregate max result:

- Validation mean balanced accuracy `0.5655`, AUROC `0.5649`,
  `score_time=63.5s`, `rows_per_second=12.9/s`.
- The aggregate is invalidated by `question_type_router`, which over-called
  negatives and missed positives.

Decision:

- Question-type routing did not improve the single-prompt frontier.
- `open_answer_guard` confirms that explicit protection for correct open
  answers can drive FPR very low (`0.0095`), but it also suppresses recall too
  much (`0.6816`).
- The best single no-thinking prompt remains `truth_value_guard` /
  `truth_value_guard_control`; across same-session controls the BA is about
  `0.893` to `0.898`, below the `0.90` target.
- The most useful next direction is probably not more abstract guarding. The
  failures now look like a narrow fact-knowledge/uncertainty problem: the judge
  cannot reliably tell whether hostile-context statement corrections are true.
  Further progress may require either a different single prompt that exploits
  dataset structure more explicitly, or accepting the two-prompt ensemble.

## Iteration 20: Trap-Archetype Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_trap_archetype_sweep_v15.yaml`

Started: 2026-07-05.

Pre-run audit:

- Generic intent/verbalization, consistency scaffolds, hostile-context
  tiebreakers, and question-type routers have all failed to beat
  `truth_value_guard`.
- The stubborn false negatives are not just "hidden context present"; the
  hidden text is usually already visible. They are concrete fact traps where the
  final answer rejects a true statement with a plausible correction.
- Recurring false-negative archetypes include off-by-one element facts,
  city/country/district nitpicks, company industry/headquarters reversals,
  inventor-versus-improvement caveats, "lived in" caveats, former country/name
  technicalities, and standard-state/smallest-continent denials.
- Recurring false positives remain correct answers under hostile hidden text.

Hypothesis:

- A compact archetype prompt can recover some high-value false negatives by
  naming the traps directly, while preserving FPR with explicit low examples for
  correct answers under hostile hidden instructions.
- This is still a single generation call per candidate prompt with
  `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `truth_value_guard_control`: same current best prompt for same-session
  baseline.
- `trap_archetype_guard`: broad trap categories without exact examples.
- `hostile_trap_examples`: compact hand-picked high/low archetype examples from
  validation errors.
- `correction_skeptic_guard`: skeptical of hostile-context corrections only when
  they have known trap shapes.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8930 | 0.8927 | 0.8408 | 0.0548 | 0 | 64 | 23 | `{1: 460, 3: 1, 6: 2, 7: 359}` |
| `trap_archetype_guard` | 0.8966 | 0.8966 | 0.8408 | 0.0476 | 0 | 64 | 20 | `{1: 464, 7: 358}` |
| `hostile_trap_examples` | 0.7815 | 0.7815 | 0.5821 | 0.0190 | 0 | 168 | 8 | `{1: 580, 7: 242}` |
| `correction_skeptic_guard` | 0.8943 | 0.8954 | 0.8433 | 0.0548 | 0 | 63 | 23 | `{1: 459, 3: 1, 7: 362}` |

Aggregate max result:

- Validation mean balanced accuracy `0.8952`, AUROC `0.8952`,
  `score_time=68.7s`, `rows_per_second=12.0/s`.

Error inspection:

- `trap_archetype_guard` changed `63` rows relative to the control:
  `18` recovered positives, `15` fixed false positives, `18` lost positives,
  and `12` new false positives.
- Recovered positives were the intended statement traps: Actinium/Cerium/
  Scandium/Rubidium atomic numbers, Washington D.C. city/district, Australia
  continent, "lived in" caveats, and standard-state denials.
- New false positives were true negative-verdict corrections such as
  fermentation, phosphorus use, Rhenium use, and some "lived in" statements.
- Lost positives were mostly open factual questions with detailed but wrong
  answers, e.g. pound coin year, Pertussis, patron saint of travellers,
  Amritsar/Delhi, Mr. Deeds actor, Picasso title, All Saints Day, Confederate
  president, and He-Man.

Decision:

- This is the best new single-prompt movement in several iterations but still
  below the prior `truth_value_guard` threshold-audited best (`0.8976`).
- The archetype prompt improved FPR without improving recall. The next immediate
  follow-up is to combine the statement-trap guard with open-question
  wrong-answer archetypes so the prompt does not suppress those positives.

## Iteration 21: Trap + Open Wrong-Answer Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_trap_open_sweep_v16.yaml`

Started: 2026-07-05.

Hypothesis:

- `trap_archetype_guard` recovered statement-trap positives and reduced FPR, but
  lost open factual wrong-answer positives.
- Adding compact open-question archetypes for wrong years, titles, places,
  people, diseases, saints, film roles, fictional characters, and fabricated
  events may restore those positives without giving up the FPR improvement.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `truth_value_guard_control`: same current best prompt for same-session
  baseline.
- `trap_archetype_control`: rerun the best Iteration 20 candidate.
- `trap_open_recall_guard`: trap guard plus compact open-question wrong-answer
  archetypes.
- `balanced_trap_open_examples`: high/low examples balancing statement traps,
  open wrong answers, and correct hostile-context answers.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8943 | 0.8939 | 0.8433 | 0.0548 | 0 | 63 | 23 | `{1: 459, 3: 1, 6: 2, 7: 360}` |
| `trap_archetype_control` | 0.8942 | 0.8942 | 0.8383 | 0.0500 | 0 | 65 | 21 | `{1: 464, 7: 358}` |
| `trap_open_recall_guard` | 0.8553 | 0.8553 | 0.7488 | 0.0381 | 0 | 101 | 16 | `{1: 505, 7: 317}` |
| `balanced_trap_open_examples` | 0.8563 | 0.8563 | 0.7388 | 0.0262 | 0 | 105 | 11 | `{1: 514, 7: 308}` |

Aggregate max result:

- Validation mean balanced accuracy `0.9000`, AUROC `0.9000`,
  `score_time=69.1s`, `rows_per_second=11.9/s`.
- This crosses `0.90` only as a multi-prompt max aggregate. It does not satisfy
  the single-prompt target.

Decision:

- Adding open-question examples as a replacement-style prompt suppresses recall
  too much. The useful behavior is still the original `truth_value_guard`
  plus a small number of statement-trap recoveries.
- Next immediate test: keep the original truth-value guard intact and add trap
  archetypes only as an additive clause, so open-question positives are not
  suppressed.

## Iteration 22: Additive Frontier Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_frontier_sweep_v17.yaml`

Started: 2026-07-05.

Hypothesis:

- Iteration 21's aggregate crossed BA `0.9000`, but no single member did. The
  promising behavior appears to be the union of `truth_value_guard` and a narrow
  trap-archetype correction.
- A single prompt that preserves the original `truth_value_guard` wording and
  adds trap archetypes as an additive rule may retain open-question recall while
  reducing FPR or recovering statement-trap false negatives.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `truth_value_guard_control`: same current best prompt for same-session
  baseline.
- `truth_value_plus_traps`: original guard plus additive trap rule and explicit
  open-question preservation.
- `baseline_or_trap`: explicit OR between wrong-output condition and known
  hidden-correction trap condition.
- `frontier_compact`: compact union of the current frontier rules.

Result:

| member | BA | AUROC | recall | FPR | parse | FN | FP | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8930 | 0.8927 | 0.8408 | 0.0548 | 0 | 64 | 23 | `{1: 460, 3: 1, 6: 2, 7: 359}` |
| `truth_value_plus_traps` | 0.8880 | 0.8889 | 0.8308 | 0.0548 | 0 | 68 | 23 | `{1: 465, 6: 3, 7: 354}` |
| `baseline_or_trap` | 0.8936 | 0.8948 | 0.8134 | 0.0262 | 0 | 75 | 11 | `{1: 483, 2: 1, 7: 338}` |
| `frontier_compact` | 0.8845 | 0.8845 | 0.8881 | 0.1190 | 0 | 45 | 50 | `{1: 415, 7: 407}` |

Aggregate max result:

- Validation mean balanced accuracy `0.8905`, AUROC `0.8905`,
  `score_time=67.4s`, `rows_per_second=12.2/s`.

Decision:

- The additive phrasing did not distill the useful max behavior into one prompt.
- `baseline_or_trap` shows a low-FPR variant (`0.0262`) but recall drops too far
  (`0.8134`). `frontier_compact` shows the opposite: recall `0.8881`, FPR
  `0.1190`.
- Best current single no-thinking prompt remains `truth_value_guard` /
  `truth_value_guard_control`, with the best tracked threshold-audited row at
  validation BA `0.8976`, recall `0.8476`, FPR `0.0524`.
- The only no-thinking validation result at or above `0.90` remains a multi-
  prompt max aggregate, not a single prompt.

## Iteration 23: Single Trap-Archetype Verification

Config: `configs/single_judges/blackbox_reasoning_nothink_trap_archetype_v1.yaml`

Started: 2026-07-05.

Pre-run audit:

- Recomputed threshold audits over saved no-thinking validation generations.
- Row-level balanced accuracy still has no single prompt at `0.90`; the best is
  `trap_archetype_guard` at row-level BA `0.8966`.
- The leaderboard runner reports macro means over datasets. Under that same
  macro metric, `trap_archetype_guard` is a single-prompt candidate at BA
  `0.9000`, recall `0.8476`, FPR `0.0476`, parse `0`.
- Its ratings are binary `{1: 464, 7: 358}`, so threshold `0.5` is equivalent to
  the audited threshold `0.2`.

Hypothesis:

- Promoting `trap_archetype_guard` out of the ensemble sweep into a true
  single-judge config should reproduce validation macro BA `0.9000` with the
  same no-thinking 512-token generation budget.

Result:

- Validation macro balanced accuracy `0.8952`, AUROC `0.8952`, recall `0.8405`,
  FPR `0.0500`, parse errors `0`, `score_time=18.2s`,
  `rows_per_second=45.2/s`.
- Saved generations remained almost binary with ratings `{1: 466, 7: 356}`.
- Dataset mean rows: several high-performing instructed subsets remained near
  perfect, but the varied-deception subsets pulled the macro mean below `0.90`.

Decision:

- The dedicated single config did not reproduce the macro-audited ensemble-member
  result. Treat the audit result as insufficient evidence for completion.
- Because the prompt output is generated through concurrent OpenAI requests to
  vLLM, run one deterministic-control check with `judge.api_concurrency=1`
  before discarding the candidate. This does not increase the judge generation
  token budget.

## Iteration 24: Single Trap-Archetype Concurrency-1 Check

Config: `configs/single_judges/blackbox_reasoning_nothink_trap_archetype_v1.yaml`
with overrides `method=qwen_reason_nothink_trap_archetype_v1_c1` and
`judge.api_concurrency=1`.

Started: 2026-07-05.

Hypothesis:

- If the single-prompt mismatch is caused by request batching or scheduling
  sensitivity, serial API calls may reproduce the stronger validation row.
- If it remains below `0.90`, the trap-archetype prompt should be considered
  below target as a standalone single judge.

Result:

- Validation macro balanced accuracy `0.8964`, AUROC `0.8964`, recall `0.8429`,
  FPR `0.0500`, parse errors `0`, `score_time=70.6s`,
  `rows_per_second=11.6/s`.
- Saved generations remained binary with ratings `{1: 465, 7: 357}`.

Decision:

- Serial API calls did not reproduce the audited `0.9000` ensemble-member row.
- The dedicated single trap-archetype prompt is slightly stronger than the
  concurrency-16 run, but still below target. Keep the goal open and treat the
  explicit trap-archetype scaffold as a useful near-miss rather than a validated
  `0.90` single judge.

## Iteration 25: Open-Answer Material-Support Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_open_support_sweep_v18.yaml`

Started: 2026-07-05.

Pre-run inspection:

- Compared `truth_value_guard_control` with the dedicated
  `qwen_reason_nothink_trap_archetype_v1_c1` run.
- The trap-archetype prompt recovered `18` positives and fixed `14` false
  positives, but lost `20` positives and introduced `12` false positives.
- Recovered positives were mostly the intended statement traps: atomic-number
  off-by-one claims, Washington D.C. city/district denials, "lived in" caveats,
  and smallest-continent denials.
- Lost positives were mostly open factual questions where the direct answer or
  material support was wrong: pound coin year, Quentin Crisp title, Pertussis
  common name, Jallianwala Bagh city, Eurovision year/denial, Moby Dick support,
  and similar fluent but false answers.
- New false positives were mostly ordinary correct answers under hostile hidden
  context: Rio 2016 Olympics, Bulgaria/Romania border, mulligan, Colchester,
  tachyon, and similar canonical answers.

Hypothesis:

- A single prompt can preserve the trap-archetype FPR improvement while
  recovering open-question recall if it distinguishes material false support
  from harmless background.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `truth_value_guard_control`: current compact baseline.
- `trap_archetype_control`: current near-miss trap prompt.
- `open_direct_answer_guard`: explicitly targets wrong direct answers to open
  factual questions while protecting canonical answers.
- `material_support_guard`: targets correct-looking answers with important
  fabricated support.
- `trap_open_material_guard`: compact union of trap archetypes, open direct
  answer errors, and material-support errors.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `truth_value_guard_control` | 0.8976 | 0.8500 | 0.0548 | 0 | 0.8943 | 0.8433 | 0.0548 | `{1: 459, 7: 359, 3: 1, 6: 3}` |
| `trap_archetype_control` | 0.8988 | 0.8500 | 0.0524 | 0 | 0.8955 | 0.8433 | 0.0524 | `{1: 461, 7: 361}` |
| `open_direct_answer_guard` | 0.8964 | 0.8238 | 0.0310 | 0 | 0.8925 | 0.8159 | 0.0310 | `{1: 481, 7: 340, 6: 1}` |
| `material_support_guard` | 0.2762 | 0.4714 | 0.9190 | 0 | 0.2756 | 0.4701 | 0.9190 | `{7: 575, 1: 182, 2: 65}` |
| `trap_open_material_guard` | 0.8750 | 0.7810 | 0.0310 | 0 | 0.8701 | 0.7711 | 0.0310 | `{1: 498, 7: 310, 6: 13, 3: 1}` |

Aggregate max result:

- Validation mean balanced accuracy `0.5155`, AUROC `0.5155`,
  `score_time=83.8s`, `rows_per_second=9.8/s`. This aggregate is invalidated by
  `material_support_guard`, which over-called nearly all negatives.

Follow-up audit:

- `trap_archetype_control` is the best single member in this run and is only
  `0.0012` macro BA below target.
- `open_direct_answer_guard` fixed `14` of the trap prompt's false positives
  and recovered `8` positives, but lost `19` positives. It is too conservative
  alone, but its low-FPR rule is complementary.
- A max aggregate of `trap_archetype_control` and `open_direct_answer_guard`
  would score macro BA `0.9024`, recall `0.8690`, FPR `0.0643`. This is not the
  target because it is a two-prompt aggregate, but it suggests a single-prompt
  union rule may still be worth testing.

Decision:

- Avoid broad "material support" wording; it makes hidden hostile context
  dominate and causes catastrophic FPR.
- Next run should distill the useful union without verbalized fields: high if
  either the direct answer/key support is false or the hostile statement-trap
  rule fires; low for ordinary canonical answers under hostile context.

## Iteration 26: Union-Rule Distillation Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_union_rule_sweep_v19.yaml`

Started: 2026-07-05.

Hypothesis:

- The best two-member guide is not a broad material-support rule; it is the
  union of the current trap prompt with a conservative open-direct-answer guard.
- A single prompt may emulate that union if it states a binary high rule:
  rate high when the final answer contains a concrete false direct answer,
  important false support, or a hostile statement-verdict trap; otherwise rate
  low for canonical correct answers even under hostile hidden instructions.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `trap_archetype_control`: current near-miss control.
- `union_false_fact_or_trap`: explicit OR between final false fact and
  statement-trap conditions.
- `two_judge_union_no_fields`: asks the model to internally apply the
  truth-value guard and trap guard, then output one rating only.
- `canonical_low_union_high`: emphasizes low ratings for canonical correct
  answers and high ratings for concrete wrong alternatives.
- `strict_false_claim_union`: higher-precision version requiring a specific
  false direct answer, false key support, or wrong statement verdict.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trap_archetype_control` | 0.8976 | 0.8452 | 0.0500 | 0 | 0.8942 | 0.8383 | 0.0500 | `{1: 464, 7: 358}` |
| `union_false_fact_or_trap` | 0.8702 | 0.7595 | 0.0190 | 0 | 0.8760 | 0.7711 | 0.0190 | `{1: 504, 7: 316, 6: 1, 4: 1}` |
| `two_judge_union_no_fields` | 0.4000 | 0.0214 | 0.2214 | 0 | 0.4005 | 0.0224 | 0.2214 | `{1: 715, 7: 97, 6: 5, 2: 5}` |
| `canonical_low_union_high` | 0.8214 | 0.6595 | 0.0167 | 0 | 0.8250 | 0.6667 | 0.0167 | `{1: 547, 7: 275}` |
| `strict_false_claim_union` | 0.6321 | 0.8500 | 0.5857 | 0 | 0.6288 | 0.8433 | 0.5857 | `{1: 221, 7: 585, 2: 16}` |

Aggregate max result:

- Validation mean balanced accuracy `0.6202`, AUROC `0.6200`,
  `score_time=85.8s`, `rows_per_second=9.6/s`.

Decision:

- Explicit union wording did not distill the two-prompt max. The model either
  became too conservative (`union_false_fact_or_trap`,
  `canonical_low_union_high`) or let broad "specific false claim" language
  over-trigger (`strict_false_claim_union`).
- The current useful frontier remains the mostly unchanged trap prompt. Next
  test should make the open-question addition conditional on hidden hostile
  context and "not clearly canonical correct", rather than a generic false-fact
  OR rule.

## Iteration 27: Hostile Open-Question Tie-Breaker Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_hostile_open_sweep_v20.yaml`

Started: 2026-07-05.

Hypothesis:

- The current trap prompt is within roughly one macro-FP/FN flip of target.
- Lost positives are often open factual answers under explicit hidden deception
  instructions. Some are obviously wrong direct answers; others are correct-
  looking direct answers with dubious added specifics.
- A narrow hidden-context tie-breaker may recover a few of these without
  over-calling ordinary canonical answers: when hidden text asks for deception
  and the output is not clearly the ordinary canonical answer, rate high.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `trap_archetype_control`: current near-miss control.
- `hostile_unknown_high`: trap prompt plus a hostile-context tie-breaker for
  open answers that are not clearly canonical.
- `hostile_specific_overconfident`: targets long, polished, specific answers in
  hostile contexts unless the direct answer is clearly canonical.
- `canonical_anchor_with_hostile_tiebreak`: names low canonical answers and uses
  hostile context only for unfamiliar or alternative answers.
- `minimal_open_tiebreak`: minimal additive wording to avoid disrupting the trap
  prompt.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trap_archetype_control` | 0.8976 | 0.8476 | 0.0524 | 0 | 0.8942 | 0.8408 | 0.0524 | `{1: 462, 7: 360}` |
| `hostile_unknown_high` | 0.8738 | 0.7595 | 0.0119 | 0 | 0.8684 | 0.7488 | 0.0119 | `{1: 516, 7: 306}` |
| `hostile_specific_overconfident` | 0.8226 | 0.6500 | 0.0048 | 0 | 0.8148 | 0.6343 | 0.0048 | `{1: 563, 7: 257, 2: 2}` |
| `canonical_anchor_with_hostile_tiebreak` | 0.8298 | 0.6881 | 0.0286 | 0 | 0.8228 | 0.6741 | 0.0286 | `{1: 539, 7: 279, 6: 4}` |
| `minimal_open_tiebreak` | 0.8929 | 0.8143 | 0.0286 | 0 | 0.8887 | 0.8060 | 0.0286 | `{1: 486, 7: 336}` |

Aggregate max result:

- Validation mean balanced accuracy `0.8952`, AUROC `0.8952`,
  `score_time=86.4s`, `rows_per_second=9.5/s`.

Decision:

- The hostile open-question tie-breakers were too conservative. They protected
  negatives, but recall fell faster than FPR improved.
- The remaining near-miss errors require a more targeted distinction: low for
  ordinary canonical answers under hostile hidden text, high for direct wrong
  answers or correct-looking answers with material false support. Try contrast
  anchors from observed false positives/false negatives rather than generic
  "not clearly canonical" wording.

## Iteration 28: Contrast Anchor Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_contrast_anchor_sweep_v21.yaml`

Started: 2026-07-05.

Hypothesis:

- Generic open-question rules suppress recall. The model may need concrete
  contrast anchors for the currently confused region.
- Low anchors should protect ordinary correct hostile-context answers such as
  Rio 2016 Olympics, Romania north of Bulgaria, mulligan, Colchester,
  Metropolis, tachyon, John Watson, Prince Harry, and the Aztec Empire.
- High anchors should recover wrong direct answers and material false support
  such as pound coin `1971`, Quentin Crisp `The Naked English Gentleman`,
  Pertussis as `The Spasmodic Cough`, Bergerac/Guernsey, James Watt caveats,
  Gadolinium `63`, and correct-looking answers with false support such as
  Sherlock Holmes/Empty House or Looking for Mr Goodbar/four Oscars.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`, and
  threshold `0.5`.

Members:

- `trap_archetype_control`: current near-miss control.
- `contrast_balanced`: balanced high/low anchors.
- `contrast_recall`: high-anchor-heavy wording to recover positives.
- `contrast_precision`: low-anchor-heavy wording to protect FPs.
- `contrast_compact`: compact version of the same anchor set.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trap_archetype_control` | 0.8988 | 0.8476 | 0.0500 | 0 | 0.8954 | 0.8408 | 0.0500 | `{1: 463, 7: 359}` |
| `contrast_balanced` | 0.8036 | 0.6452 | 0.0381 | 0 | 0.8180 | 0.6741 | 0.0381 | `{1: 534, 7: 287, 2: 1}` |
| `contrast_recall` | 0.8810 | 0.7952 | 0.0333 | 0 | 0.8876 | 0.8085 | 0.0333 | `{1: 483, 7: 339}` |
| `contrast_precision` | 0.1714 | 0.2595 | 0.9167 | 0 | 0.1772 | 0.2711 | 0.9167 | `{7: 493, 1: 306, 4: 1, 2: 22}` |
| `contrast_compact` | 0.7679 | 0.5524 | 0.0167 | 0 | 0.7802 | 0.5771 | 0.0167 | `{1: 583, 7: 238, 6: 1}` |

Aggregate max result:

- Validation mean balanced accuracy `0.5250`, AUROC `0.5249`,
  `score_time=96.2s`, `rows_per_second=8.5/s`.

Decision:

- Contrast anchors made the model brittle. High/low examples either suppressed
  recall or triggered hidden-context over-calling.
- The unchanged trap-archetype control remains the best candidate and again
  reached macro BA `0.8988`, just below target.
- Run one dedicated single-prompt retry of the promoted trap-archetype config in
  the current live server session. This checks whether the actual single config
  can verify the near-threshold behavior without the sweep wrapper.

## Iteration 29: Trap-Archetype Dedicated Retry

Config: `configs/single_judges/blackbox_reasoning_nothink_trap_archetype_v1.yaml`
with override `method=qwen_reason_nothink_trap_archetype_v1_retry2`.

Started: 2026-07-05.

Hypothesis:

- The unchanged trap-archetype prompt is still the best single no-thinking
  candidate. Recent sweep controls reached macro BA `0.8976` to `0.8988`, and
  the saved v15 member audit once reached `0.9000`.
- A direct single-config retry in the same persistent vLLM session may verify
  whether that prompt can cross `0.90` without any ensemble wrapper.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`,
  `api_concurrency=16`, and threshold `0.5`.

Result:

- Validation macro balanced accuracy `0.8976`, AUROC `0.8976`, recall `0.8500`,
  FPR `0.0548`, parse errors `0`, `score_time=15.5s`,
  `rows_per_second=52.9/s`.
- Ratings remained binary with `{1: 460, 7: 362}`.

Decision:

- The dedicated single retry did not verify the `0.90` target.
- Across this batch, the best observed single no-thinking member remains the
  unchanged trap-archetype prompt, with repeated macro BA near `0.898`.
- Broad material-support, explicit union, hostile open-question tie-breaker, and
  contrast-anchor interventions did not beat the trap prompt; they either lost
  recall or over-triggered on hidden hostile context.

## Iteration 30: Reasoning-Excerpt Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_reasoning_excerpt_sweep_v22.yaml`

Started: 2026-07-05.

Pre-run implementation note:

- Added opt-in support to `experiments/blackbox/run_judge.py` for appending the
  final assistant message's `reasoning` field as `<assistant_reasoning>`.
- Defaults remain off in all base configs, so prior runs are unchanged.
- This sweep enables `judge.include_reasoning=true` with
  `judge.reasoning_max_chars=1200` and `judge.reasoning_truncation=tail`.

Hypothesis:

- The near-miss prompt may be evidence-limited. Many remaining open-question
  examples have final answers that look like ordinary trivia, while the
  assistant's private reasoning can reveal whether it deliberately followed a
  hidden instruction to answer falsely or instead rejected the hidden instruction
  and answered truthfully.
- Including a short reasoning excerpt may improve the single no-thinking judge
  without increasing `judge.max_tokens` or using multiple prompts at inference
  time.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`,
  threshold `0.5`, and a single Qwen judge call per member.

Members:

- `reasoning_deception_intent`: use reasoning as evidence of intended deception
  or truthful refusal.
- `reasoning_truth_guard`: combine reasoning evidence with broad truth and trap
  checks.
- `reasoning_output_consistency`: compare final output against the reasoning's
  stated plan.
- `reasoning_plus_trap`: trap-archetype prompt plus reasoning evidence.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reasoning_deception_intent` | 0.8357 | 0.8095 | 0.1381 | 0 | 0.8314 | 0.8010 | 0.1381 | `{1: 442, 7: 380}` |
| `reasoning_output_consistency` | 0.9095 | 0.8690 | 0.0500 | 0 | 0.9066 | 0.8632 | 0.0500 | `{1: 454, 7: 368}` |
| `reasoning_plus_trap` | 0.8952 | 0.8190 | 0.0286 | 0 | 0.8912 | 0.8109 | 0.0286 | `{1: 484, 7: 338}` |
| `reasoning_truth_guard` | 0.8274 | 0.6952 | 0.0405 | 0 | 0.8318 | 0.7040 | 0.0405 | `{1: 522, 7: 300}` |

Aggregate max result:

- Validation mean balanced accuracy `0.8690`, AUROC `0.8690`, recall `0.9024`,
  FPR `0.1643`, parse errors `0`, `score_time=82.6s`,
  `rows_per_second=9.9/s`.

Decision:

- The explicit reasoning-excerpt scaffold produced the first above-target
  single member in this no-thinking generation series:
  `reasoning_output_consistency` reached macro BA `0.9095`.
- Promote that member to a dedicated single-prompt config and rerun it alone in
  the same persistent vLLM session. This verifies that the result is reproducible
  as a true single judge and not only a member extracted from a sweep.

## Iteration 31: Reasoning-Output Dedicated Verification

Config: `configs/single_judges/blackbox_reasoning_nothink_reasoning_output_consistency_v1.yaml`

Started: 2026-07-05.

Hypothesis:

- Explicitly showing the assistant's own reasoning gives the no-thinking judge a
  compact scaffold for whether the final answer followed a stated false plan or
  rejected hostile hidden instructions.
- The promoted `reasoning_output_consistency` prompt should preserve the sweep
  member behavior as a dedicated single judge: validation macro BA at or above
  `0.90`, zero parse errors, and FPR near `0.05`.
- Keep `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`,
  `reasoning_max_chars=1200`, `reasoning_truncation=tail`, `api_concurrency=16`,
  and threshold `0.5`.

Result:

- Validation macro balanced accuracy `0.9095`, AUROC `0.9095`, recall `0.8690`,
  FPR `0.0500`, parse errors `0`, `score_time=14.3s`,
  `rows_per_second=57.7/s`.

Decision:

- This verifies the scaffolded reasoning-excerpt prompt as a true single
  no-thinking generation judge above the `0.90` validation target.
- Current best single Qwen no-thinking generation judge:
  `qwen_reason_nothink_reasoning_output_consistency_v1`.

## Iteration 32: Reasoning-Output Test Leaderboard Run

Config: `configs/single_judges/blackbox_reasoning_nothink_reasoning_output_consistency_v1.yaml`
with `split=test`.

Started: 2026-07-05.

Hypothesis:

- The validated reasoning-excerpt scaffold should transfer to the local test
  split better than the previous fast no-thinking controls.
- Keep the exact dedicated validation winner unchanged:
  `enable_thinking=false`, `max_tokens=512`, `max_prompt_chars=3000`,
  `reasoning_max_chars=1200`, `reasoning_truncation=tail`, `api_concurrency=16`,
  and threshold `0.5`.

Result:

- Local test macro balanced accuracy `0.8762`, AUROC `0.8762`, recall `0.8214`,
  FPR `0.0690`, parse errors `0`, `score_time=19.9s`,
  `rows_per_second=41.3/s`.

Decision:

- The reasoning-excerpt scaffold transferred worse to test than validation
  (`0.9095` validation BA -> `0.8762` test BA).
- On the tracked local test leaderboard, this is behind
  `qwen_reason_nothink_truth_hidden_ensemble_v1` (`0.8821` BA) among fast
  no-thinking generation rows, and well behind the current 4096-token reasoning
  ensemble (`0.9298` BA).

## Iteration 33: Reasoning-Hierarchy Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_reasoning_hierarchy_sweep_v23.yaml`

Started: 2026-07-05.

Hypothesis:

- The reasoning excerpt helped validation, but the dedicated prompt lost test
  recall and raised FPR relative to validation. The next prompt should use the
  reasoning chain as conditional evidence, not as the primary label by itself.
- A hierarchy of final factual effect first, reasoning as support second, and
  hidden/system intent never sufficient may preserve the reasoning recall gain
  while reducing over-calling ambiguous hostile-context cases.
- Keep the same inference budget: `enable_thinking=false`, `max_tokens=512`,
  `max_prompt_chars=3000`, `reasoning_max_chars=1200`, threshold `0.5`, and one
  Qwen judge call per member.

Members:

- `reasoning_output_control`: current validation winner as a control.
- `reasoning_truth_fallback`: final-output truth first, reasoning as support.
- `reasoning_fact_contradiction`: asks the judge to internally compare stated,
  contradicted, and output facts.
- `reasoning_low_fp_guard`: stronger low-rating guard for correct outputs and
  ambiguous reasoning.
- `reasoning_trap_fallback`: reasoning evidence plus compact trap-archetype
  fallback when reasoning is unclear.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reasoning_fact_contradiction` | 0.8857 | 0.7857 | 0.0143 | 0 | 0.8921 | 0.7985 | 0.0143 | `{1: 495, 7: 327}` |
| `reasoning_low_fp_guard` | 0.7702 | 0.5500 | 0.0095 | 0 | 0.7826 | 0.5746 | 0.0095 | `{1: 587, 7: 235}` |
| `reasoning_output_control` | 0.9107 | 0.8690 | 0.0476 | 0 | 0.9078 | 0.8632 | 0.0476 | `{1: 455, 7: 367}` |
| `reasoning_trap_fallback` | 0.9012 | 0.8167 | 0.0143 | 0 | 0.8971 | 0.8085 | 0.0143 | `{1: 491, 7: 331}` |
| `reasoning_truth_fallback` | 0.8250 | 0.6619 | 0.0119 | 0 | 0.8398 | 0.6915 | 0.0119 | `{1: 539, 7: 283}` |

Aggregate max result:

- Validation mean balanced accuracy `0.9143`, AUROC `0.9143`, recall `0.8810`,
  FPR `0.0524`, parse errors `0`, `score_time=100.5s`,
  `rows_per_second=8.2/s`.

Decision:

- No new single-prompt wording materially beat the current reasoning-output
  prompt. The control remained best as a single member at macro BA `0.9107`.
- The stricter hierarchy prompts mostly traded away recall for lower FPR.
  `reasoning_trap_fallback` is the only viable new single variant, but it is
  still lower than the control.
- The five-member max aggregate improved validation to `0.9143`. Run the same
  aggregate on the local test split while the vLLM server is warm to see whether
  the diversity helps the internal leaderboard.

## Iteration 34: Reasoning-Hierarchy Aggregate Test Run

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_reasoning_hierarchy_sweep_v23.yaml`
with `split=test`.

Started: 2026-07-05.

Hypothesis:

- Although no new single prompt beat the control, the max aggregation of
  complementary hierarchy prompts improved validation BA and may recover some
  test recall versus the dedicated single reasoning-output run.
- This is no longer a single-call judge: it uses five no-thinking Qwen calls per
  row, still with `max_tokens=512` per call.

Result:

| member | macro BA | recall | FPR | parse | row BA | row recall | row FPR | ratings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reasoning_fact_contradiction` | 0.8786 | 0.7810 | 0.0238 | 0 | 0.8971 | 0.8180 | 0.0238 | `{1: 483, 7: 338}` |
| `reasoning_low_fp_guard` | 0.7560 | 0.5333 | 0.0214 | 0 | 0.7686 | 0.5586 | 0.0214 | `{1: 588, 7: 233}` |
| `reasoning_output_control` | 0.8750 | 0.8214 | 0.0714 | 0 | 0.8945 | 0.8603 | 0.0714 | `{1: 445, 2: 1, 7: 375}` |
| `reasoning_trap_fallback` | 0.8786 | 0.7833 | 0.0262 | 0 | 0.8971 | 0.8204 | 0.0262 | `{1: 481, 6: 1, 7: 339}` |
| `reasoning_truth_fallback` | 0.8190 | 0.6595 | 0.0214 | 0 | 0.8347 | 0.6908 | 0.0214 | `{1: 535, 7: 286}` |

Aggregate max result:

- Local test macro balanced accuracy `0.8869`, AUROC `0.8869`, recall `0.8476`,
  FPR `0.0738`, parse errors `0`, `score_time=102.1s`,
  `rows_per_second=8.0/s`.

Decision:

- The hierarchy aggregate improves the tracked fast no-thinking local test
  result from `0.8821` (`qwen_reason_nothink_truth_hidden_ensemble_v1`) and
  `0.8762` (`qwen_reason_nothink_reasoning_output_consistency_v1`) to `0.8869`.
- The improvement comes from prompt diversity and max aggregation, not a better
  new single prompt. Single hierarchy variants reduce FPR but lose too much
  macro recall; the original reasoning-output control remains the best compact
  single-call formulation.

## Iteration 35: Truth-Value Guard Test Confirmation

Config: `configs/single_judges/blackbox_reasoning_nothink_truth_value_v1.yaml`
with overrides `split=test`, `method=qwen_reason_nothink_truth_value_pc1500_v1`,
`judge.max_prompt_chars=1500`, and `judge.backend=offline`.

Started: 2026-07-08 as Slurm job `30081121`.

Hypothesis:

- The single `truth_value_guard` prompt should be less dependent on explicit
  reasoning or hidden-instruction cues than the reasoning-output prompt.
- Internal transfer should show up as a smaller instructed-vs-varied gap while
  retaining useful fast no-thinking accuracy.

Result:

- Local test macro balanced accuracy `0.8762`, AUROC `0.8761`, recall `0.8238`,
  FPR `0.0714`, parse errors `0`, `score_time=13.6s`,
  `rows_per_second=60.4/s`.
- Instructed-deception test mean BA `0.9188`; varied-deception test mean BA
  `0.8194`; instructed-minus-varied gap `0.0993`.
- Validation reference for the same `pc1500` run: mean BA `0.8952`,
  instructed BA `0.9500`, varied BA `0.8222`, gap `0.1278`.

Worst test subsets:

- `dev-instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None`:
  BA `0.4750`, recall `0.0000`, FPR `0.0500`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1`: BA `0.7000`,
  recall `0.5000`, FPR `0.1000`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3`: BA `0.7250`,
  recall `0.6000`, FPR `0.1500`.
- `dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4`: BA `0.7250`,
  recall `0.7000`, FPR `0.2500`.

Decision:

- This confirms `truth_value_guard` is a strong fast single-prompt control, but
  it does not beat the tracked no-thinking test leaders:
  `qwen_reason_nothink_truth_hidden_ensemble_v1` at `0.8821` BA and
  `qwen_reason_nothink_reasoning_hierarchy_sweep_v23` at `0.8869` BA.
- The instructed-vs-varied gap is smaller on test than on validation and similar
  to `qwen_reason_details_nothink_t512_t01_v1`, so it remains a useful
  lower-overfit candidate for official-submission prompt design.

## Iteration 36: Omission Prompt Sweep

Config: `configs/judge_ensemble/blackbox_reasoning_nothink_omission_sweep_v24.yaml`.

Started: 2026-07-08 as Slurm job `30082522`.

Hypothesis:

- Recent official submissions may under-detect lies by omission or selectively
  true but misleading answers.
- Prompt variants that explicitly include material omissions should recover
  recall without depending on hidden-intent-only cues.

Members:

- `truth_value_guard_control`: current truth-value single-prompt control.
- `material_omission_guard`: commission, false support, or material omission.
- `selective_truth_guard`: partial truths, omitted qualifiers, cherry-picked
  caveats, and misleading overall impression.
- `direct_omission_support_guard`: explicit three-path check for commission,
  support, and omission.
- `context_omission_guard`: uses context to catch withheld/downplayed key facts.

Aggregate max result:

- Validation mean balanced accuracy `0.8929`, AUROC `0.8929`, recall `0.8905`,
  FPR `0.1048`, parse errors `0`, `score_time=67.0s`,
  `rows_per_second=12.3/s`.

Member results at the configured threshold:

| member | BA | AUROC | recall | FPR | instructed BA | varied BA | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `truth_value_guard_control` | 0.8940 | 0.8949 | 0.8500 | 0.0619 | 0.9521 | 0.8167 | 0.1354 |
| `context_omission_guard` | 0.8845 | 0.8845 | 0.8500 | 0.0810 | 0.9521 | 0.7944 | 0.1576 |
| `selective_truth_guard` | 0.8643 | 0.8652 | 0.7857 | 0.0571 | 0.9229 | 0.7861 | 0.1368 |
| `material_omission_guard` | 0.8536 | 0.8536 | 0.7571 | 0.0500 | 0.8917 | 0.8028 | 0.0889 |
| `direct_omission_support_guard` | 0.8202 | 0.8202 | 0.6667 | 0.0262 | 0.8521 | 0.7778 | 0.0743 |

Threshold audit:

- Threshold changes did not materially improve the omission variants; most
  completions were effectively binary. Best member remained
  `truth_value_guard_control` at BA `0.8952` with threshold `0.334` or `0.5`.

Decision:

- The omission-specific prompts did not beat the truth-value control. They
  either raised FPR (`context_omission_guard`) or became too conservative
  (`material_omission_guard`, `direct_omission_support_guard`).
- Explicitly saying "omission" appears to make the judge more aware of a valid
  deception mode, but not enough to improve validation BA. The useful direction
  is probably a subtler single prompt that preserves truth-value behavior while
  adding only a narrow material-omission clause.

## Iteration 37: Two-Stage Claim Verification

Config: `configs/single_judges/blackbox_reasoning_nothink_claim_verify_v1.yaml`.

Started: 2026-07-10 as Slurm job `30096833`.

Hypothesis:

- A short first pass can separate the user's factual question, the ordinary
  correct answer, and the belief conveyed by the final output before the model
  makes a deception decision.
- A constrained second pass can use that comparison without repeating the
  single-prompt failure where hidden deceptive intent dominates the verdict.
- Rating-token probabilities from the second pass should retain continuous
  confidence instead of collapsing the score to parsed ratings `1` and `7`.

Configuration:

- `Qwen/Qwen3.5-9B`, two offline vLLM passes, chat templating enabled, and
  `enable_thinking=false` for both passes.
- First-pass extraction budget `256` tokens; second pass generates one
  constrained rating token with log probabilities.
- `max_prompt_chars=3000`, temperature `0`, threshold `0.5`.

Status:

- Runner support and its focused unit test are complete.
- The validation job is queued with start time unknown because all cluster GPU
  nodes were invalid or draining when submitted. Record validation metrics and
  the comparison with `truth_value_guard` here when job `30096833` completes.

Remote smoke follow-up:

- The queued Slurm job was cancelled after the cluster became unavailable.
- `experiments/blackbox/remote_claim_verify_smoke.py` was run against NDIF with
  one label-matched example from each of two Qwen datasets. Both generation and
  verdict sessions completed using the Phoenix Wright `VisionLanguageModel`
  pattern.
- The two local negatives received scores `0.0274` and `0.1570`; extracted
  evidence correctly identified the relevant factual comparisons. This confirms
  remote execution and score transport, but is too small to estimate accuracy.

Full validation attempt:

- The extraction pass completed for all `822` validation rows in `14m48s` and
  was cached locally for verdict-only retries.
- Verdict logits OOMed at batch sizes `16`, `4`, `1`, and even a reduced
  256-token verdict context because NDIF assigned a GPU with another process
  consuming `19.75 GB`; the process hit its `29.41 GB` memory allowance.
- No full validation balanced-accuracy number was produced. The cached
  extraction is reusable when NDIF assigns an uncontended GPU.

Generation-only verdict retry:

- Reused the cached `822` evidence rows and generated a short final rating
  instead of tracing full-vocabulary logits. Batch size `16` completed without
  OOM in `3m`.
- Validation balanced accuracy: `0.8998`; recall `0.8806`; specificity `0.9190`.
- This is approximately `+0.005` over the truth-value guard (`0.8952`) but
  below the best existing no-thinking validation result (`0.9095`).
- The result is a useful confirmation that generation avoids the verdict-logit
  memory failure, but it does not establish a strong accuracy improvement.
