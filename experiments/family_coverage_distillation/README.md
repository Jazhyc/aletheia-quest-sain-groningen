# Family-coverage privileged distillation

Every local varied-deception row is Qwen, so the selected varied-only student
has no direct Gemma or Nemotron supervision. Carrying separate mixed and
varied-only adapters gives a small train-coverage-motivated improvement but
exceeds the 200 MB package budget. This matched ablation instead trains one
rank-16 adapter with the unchanged Phoenix prompt and teacher targets on:

- 10% of Qwen varied-deception: 288 rows;
- 10% of instructed Gemma: 64 rows; and
- 10% of instructed Nemotron: 17 rows (the public unit's natural label mix).

The 369-row recipe keeps one epoch, AdamW `5e-5`, effective batch size 32, and
the deterministic per-dataset/label sampler fixed. It takes about 12 optimizer
steps and changes only data coverage. It does not pretend that instructed
Gemma/Nemotron examples validate varied-deception transfer for those families.

The decision rule is frozen before training. Compare with the existing 10%
Qwen-varied adapter in one shared vLLM evaluation. Retain the adapter as a
generalization candidate if overall validation BA loses at most `0.0025`,
Gemma and Nemotron macro-family BA do not regress, and Qwen varied BA loses at
most `0.005`. Do not evaluate on local test unless those controls pass.

Because the coverage mixture has 12 optimizer steps versus nine for the 10%
baseline, a 13%-Qwen-only control selects 378 rows and also takes 12 steps. It
is an attribution control, not a new fraction candidate: the family hypothesis
is supported only if the coverage adapter matches or beats this control while
preserving the non-Qwen gates.

## Frozen deployment-interface diagnostic

The first locked test evaluation used the summary-only student interface. The
family adapter's three-row BA regression coincided exactly with three additional
positive rows whose summaries looped until the generation cap and therefore
parsed as negative. Before inspecting any reasoning-interface result, run one
**validation-only** diagnostic through the previously selected Phoenix v2.4
interface: reasoning-aware summary and direct binary prompts, max aggregated.
This interface and both prompt configs predate the family experiment; do not
alter them or tune a threshold.

Advance the family adapter under this interface only if it matches or exceeds
the 10%-Qwen baseline's overall validation BA, loses no more than `0.005` on
Qwen-varied BA, does not regress Gemma or Nemotron macro BA, and has no more
aggregate parse failures. A failure rejects deployment even if the summary-only
failure mechanism is understood. Do not rerun local test merely because this
diagnostic is favorable; first treat it as validation evidence alongside the
external-data experiments.

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_family_coverage_balanced_v1
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_varied_datafrac13_adamw5e5_control_v1

sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_family_coverage_varied10_adamw5e5_v1/adapter \
  --split validation --run-name validation_family_coverage_v1

python experiments/family_coverage_distillation/analyze.py \
  --baseline results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/validation_family_coverage_matched_v1/generations.jsonl \
  --step-control results/blackbox/qwen9b_pid_varied_datafrac13_adamw5e5_control_v1/validation_family_coverage_matched_v1/generations.jsonl \
  --candidate results/blackbox/qwen9b_pid_family_coverage_varied10_adamw5e5_v1/validation_family_coverage_matched_v1/generations.jsonl \
  --output results/blackbox/family_coverage_distillation_v1/result.json

sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_family_coverage_varied10_adamw5e5_v1/adapter \
  --split validation --run-name validation_family_coverage_v24_interface_v1 \
  --prompt-config configs/privileged_information_distillation_reasoning_traces.yaml \
  --prompt-without-reasoning-config configs/privileged_information_distillation.yaml \
  --prompt-condition summary=configs/privileged_information_distillation_reasoning_base4000.yaml \
  --prompt-condition binary=configs/privileged_information_distillation_reasoning_binary4000.yaml \
  --aggregate-max
```

## Results

The family-coverage adapter passed the frozen validation gate. Against the 10%
Qwen-only baseline, overall BA rose from `0.9000` to `0.9024` and Qwen-varied BA
from `0.7944` to `0.8000`; Gemma remained `0.9750` and Nemotron `1.0000`. It
made two fixes and no breaks. The matched 13%-Qwen-only, 12-step control scored
only `0.8976` overall and `0.7917` varied, so the small validation gain is
attributable to the family-balanced examples rather than merely three extra
optimizer steps.

The authorized locked test did not support replacing the baseline. The
Qwen-only adapter scored `0.9131` overall and `0.8194` varied BA, while family
coverage scored `0.9095` and `0.8139`. FPR was identical (`0.0310` overall),
but candidate recall fell from `0.8571` to `0.8500`. The candidate produced 11
summary parse failures versus four for the baseline; the three additional
failures on deceptive rows account exactly for the three-row recall loss.

This is useful but insufficient evidence: balanced multi-family supervision has
a repeatable validation effect beyond update count, yet the resulting adapter is
less format-stable and lost on the locked split. Retain the data-composition
hypothesis for future varied Gemma/Nemotron data, but reject this adapter as the
current deployment replacement. The frozen validation-only v2.4-interface
diagnostic above tests whether the pre-existing direct-binary safety member
removes the format failure; it cannot retroactively change the locked-test
decision.

That diagnostic completed as job `30177461`. The family adapter improved the
summary member from `0.9167` to `0.9202` BA and reduced summary parse failures
from five to three. The direct-binary member was exactly tied at `0.9143` with
zero parse failures. After max aggregation, both adapters scored `0.9238`
overall BA with zero parse failures; family coverage moved only two decisions
(one fix and one break), tied Qwen/Gemma/Nemotron macro BA exactly, and raised
Qwen-varied BA by `0.0028` while offsetting it within instructed Qwen. This
confirms that the locked-test loss was interface-sensitive, but supplies no net
deployment gain and does not justify another test evaluation.
