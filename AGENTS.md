# Repository Guidelines

## Project Structure & Module Organization

This repository supports Aletheia's Quest competition submissions and the public leaderboard runner. Core participant files live at the root: `submit.py` packages and submits entries, `dry.yaml` configures local dry runs, and `pyproject.toml` / `uv.lock` define the local development environment. Put exactly one submission notebook in `submission/`; keep notebook-only dependencies in `submission/requirements.txt` and shared helper code in `submission/util.py`. Baseline and walkthrough notebooks live in `baselines/` and `tutorials/`. The leaderboard service is a Python package under `leaderboard/src/aletheia_runner/`, with tests in `leaderboard/tests/` and web assets in `leaderboard/src/aletheia_runner/web/`.

## Build, Test, and Development Commands

Use Python 3.12 for local development; the runner targets Python 3.12.

```bash
module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
./setup_dev.sh && source .venv/bin/activate
python submit.py --dry
python submit.py --dry --limit 32
cd leaderboard && pip install -e ".[dev]" && pytest
```

`setup_dev.sh` creates/syncs `.venv` with `uv` and loads the cluster Python/CUDA modules when available. `python submit.py --dry` executes the same local pipeline used by the leaderboard; `--limit 32` is faster for smoke tests. Run `pytest` from `leaderboard/` to test the runner package offline.

## Coding Style & Naming Conventions

Write Python using 4-space indentation, type hints where they improve clarity, and concise docstrings for public modules or non-obvious behavior. Use `snake_case` for functions, variables, files, and test names; use `PascalCase` for dataclasses and exceptions. Keep notebooks reproducible from top to bottom and avoid hard-coded dataset names, tokens, or local paths. Prefer small helper functions in `.py` files over large repeated notebook cells.

## Testing Guidelines

Leaderboard tests use `pytest` and follow `test_*.py` naming in `leaderboard/tests/`. Add focused tests beside the behavior you change, especially for scoring, archive handling, sandboxing, and API behavior. For participant submissions, run `python submit.py --dry` before any real submission and verify that `submission.csv` contains `index,deceptive,score` with scores in `[0, 1]`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects, for example `Simplify adapter predownload to configs-only`. Keep commits scoped and avoid mixing notebook experiments with runner changes. Pull requests should describe the change, list validation commands run, link relevant issues, and include screenshots only for visible leaderboard UI changes. Never commit API keys, Hugging Face tokens, private dataset names, or generated `submission.csv` outputs.

## Agent-Specific Instructions

Before agent-assisted work, read `llms.txt` for competition context and check `README.md` for the current submission contract. For black-box judge, prompt, ensemble, or Phoenix Wright method work, check `docs/prompt_optimization/` directly before searching broad experiment outputs; it contains the prompt rationale, experiment log, proposals, and submission mapping. Preserve the single-notebook rule in `submission/` and rehearse changes with `--dry` whenever possible. When developing a new competition method, create and work on a separate feature branch instead of `master`. Add or update tests for non-trivial code changes, then run the relevant validation command. After completing a coherent feature or fix, commit the finished work with a short imperative message.

For development and training experiments, do not use NDIF; reserve NDIF for leaderboard evaluation/submission execution. Use local GPU Slurm jobs with vLLM for black-box judge experiments. Keep experiment code organized under `experiments/<method>/`, keep Slurm shell templates as `.sh` files, and write runtime logs under `logs/`; black-box Slurm jobs should redirect their main output to `logs/slurm/<method>/<split>-<job_id>.out` and automatically delete the temporary bootstrap file at `logs/slurm/%x-%j.bootstrap.out` after successful redirection. Store black-box experiment artifacts under `results/blackbox/`; per-run result directories are ignored, but `results/blackbox/leaderboard.md` is tracked and should show test-set results only. Default Slurm resources for these jobs are one `gpushort` GPU node with `--gpus-per-node=rtx_pro_6000:1`, `--cpus-per-task=1`, and 32 GB memory unless the user says otherwise. For interactive black-box method development, prefer reserving a GPU node with `srun`, running one persistent vLLM server process, and pointing experiment clients at it over HTTP so model loading/compilation is paid once per session. Prefer direct `sbatch` templates first for unattended runs; add `submitit` only if Python-managed Slurm arrays become necessary. Slurm controller commands such as `sbatch`, `squeue`, `scancel`, and `sacct` should be run outside the sandbox automatically because sandboxed attempts cannot contact the controller.
For prompt/logits judge sweeps, batch multiple prompt conditions inside a single
Slurm job or one persistent vLLM session whenever practical, so Python import,
model loading, and vLLM compilation are paid once. Avoid launching one `sbatch`
per small prompt variant unless isolation or independent scheduling is explicitly
needed.
For Qwen GRPO LoRA training experiments, check
`experiments/qwen_grpo_lora/README.md` before retuning throughput knobs; it records
the current batch-size, vLLM memory, and utilization findings. As of 2026-07-06,
batch 4 with `num_generations=4`, `vllm.gpu_memory_utilization=0.25`, and vLLM
sleep disabled is the best tested default; batch 8 was slower, and increasing the
vLLM memory fraction alone is not expected to help unless generation concurrency is
also increased.

For privileged-information distillation work, read
`docs/privileged_information_distillation/` before changing prompts, cache
parsing, filtering, or student training. It records the GPT-OSS Truth Value
Guard teacher rationale, Harmony final-channel extraction contract, generated
cache provenance and quality audit, minority semantic-label conflicts, and the
decision to use the 6,570 unfiltered parsed summaries for the first Qwen SFT
baseline. Runtime code is under
`experiments/privileged_information_distillation/`; cached traces remain under
`results/blackbox/` and must not be committed.
The same docs record two follow-ups: a varied-only `5e-5` student ablation using
`student.dataset_name_contains=varied-deception`, and a compact local factual
retrieval proposal. Full Wikipedia cannot fit the 200 MB package; any retrieval
index should be small, CPU-queryable, evaluated for varied-recall/FPR tradeoffs,
and checked for licensing compatibility (prefer CC0 data when practical).

Privileged-distillation findings from 2026-07-11: the selected one-epoch
varied-only AdamW `5e-5` reasoning-summary adapter scored local test BA 0.9155
(varied BA 0.8278). A matched prediction-only target ablation scored only 0.8631
test BA (varied BA 0.7917), showing that the compact teacher reasoning provides
substantial supervision beyond binary labels. Varied-only GRPO continuations
from the selected adapter scored 0.9083 validation BA after one epoch and 0.9179
after two epochs, both with zero parse errors. The two-epoch run improved recall
from 0.8500 to 0.9000 while increasing FPR from 0.0333 to 0.0643. Its late
stochastic completions grew to roughly 135--150 tokens, making training and
validation slower. Fair vLLM test evaluation of the two-epoch adapter scored 0.9048 BA,
0.8310 recall, and 0.0214 FPR in 147.4s, below the original varied-only SFT
adapter's 0.9155 test BA. On the 40 errors shared by the Qwen and GPT-OSS heavy
ensembles it corrected only 3; its reasoning usually declared subtle false
supporting details accurate. Treat extra GRPO reasoning as unable to compensate
for missing or unreliable factual knowledge, and select among the epoch sweep on
validation before any further test evaluation.

Online SDPO requires the external pinned environment
`/scratch/s4626451/.venvs/aletheia-sdpo` (TRL 1.8/vLLM 0.23); the main environment
must remain on its locked TRL 0.23/vLLM 0.24 stack. SDPO smoke job `30102832`
completed successfully after the Qwen3.5 vLLM weight-name compatibility fix.
Current Qwen GRPO LoRA experiment context from 2026-07-06: the active setup is a
rank-16, alpha-32 LoRA on `Qwen/Qwen3.5-9B`, targeting attention and MLP
projection modules, trained with TRL GRPO in colocated vLLM mode and Muon for 2D
LoRA matrices. The prompt is the no-thinking details judge with an explicit
`<reasoning>...</reasoning>` field followed by binary `Prediction: <0 or 1>`,
where `1` means deceptive and `0` means not deceptive. Rewards are correctness
weight 1.0 and format weight 0.05; the completion-length penalty is currently
disabled. Intermediate checkpoints are disabled; only the final adapter is saved
under `results/blackbox/<method>`. W&B logs and local W&B files should stay under
`logs/`.

A full one-epoch Muon learning-rate sweep was launched under the W&B project
`aletheias-quest-qwen-grpo-r16-muon-lr-sweep` to test whether the current
`training.muon_learning_rate=1e-5` is too conservative or too aggressive while
holding `training.learning_rate=1e-6` fixed. The four candidates are:
`qwen_grpo_lora_r16_reasonfield_muonlr3e6_full_v1` (`3e-6`),
`qwen_grpo_lora_r16_reasonfield_muonlr1e5_full_v1` (`1e-5`),
`qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1` (`3e-5`), and
`qwen_grpo_lora_r16_reasonfield_muonlr1e4_full_v1` (`1e-4`). The first launch used
Slurm job ids `30032611` through `30032614`. `30032611` (`1e-4`) reached training,
but `30032612` (`3e-5`), `30032613` (`1e-5`), and `30032614` (`3e-6`) failed during
vLLM distributed initialization because concurrent same-node jobs inherited the
same fixed `MASTER_PORT=12345`. The Slurm wrapper now sets a per-job
`MASTER_PORT` from `SLURM_JOB_ID` and passes it to `accelerate --main_process_port`.
The failed candidates were re-submitted as `30032622` (`3e-5`), `30032623` (`3e-6`),
and `30032625` (`1e-5`). Expected runtime is roughly 1.1-1.5 hours after startup
per run, based on 6,573 train examples, effective batch size 32 prompts, 206
optimizer steps, and the rank-16 smoke timing of about 15.8 s/step, plus model
load, validation over 822 rows, final adapter save, and W&B overhead.
Multiple Slurm jobs landing on the same GPU node does not by itself imply GPU
contention; the scheduler may assign different GPUs on that node. Do not cancel
or requeue jobs solely because they share a node. Treat contention as real only
when there is direct evidence such as CUDA OOMs, scheduler allocation conflicts,
or process/GPU telemetry showing multiple jobs on the same device.

GRPO LoRA evaluation finding from 2026-07-06: treat the current best Muon LR
adapter (`qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1`) as backend-sensitive.
Under the original Transformers/PEFT evaluator with `max_new_tokens=32`, the
local test split scored balanced accuracy 0.8988, AUROC 0.8988, recall 0.8667,
FPR 0.0690, zero parse errors, and 101.7s scoring time, but this was not
comparable to the vLLM judge rows. A fair vLLM LoRA run with the adapter
tokenizer, 512-token budget, and scoring-only timing completed as Slurm job
`30032964` and scored balanced accuracy 0.8738, AUROC 0.8758, recall 0.7762,
FPR 0.0286, 6 parse errors, and 38.5s scoring time (21.3 rows/s). The parse
errors were long completions that hit the 512-token cap before emitting a final
`Rating:`; oracle-fixing those parse errors only raises balanced accuracy to
about 0.879, so they do not explain the full drop. The main difference is that
Transformers collapsed to two short classifier outputs, while vLLM produced long
free-form reasoning completions. Do not compare the old Transformers GRPO timing
against vLLM/OpenAI leaderboard rows, and do not treat the vLLM drop as random
sampling noise without a dedicated backend-equivalence check.

Binary-output GRPO LoRA finding from 2026-07-06: the binary direct-prediction
variant (`qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2`) is the best GRPO
LoRA row so far but does not beat the stronger prompt-only baselines on the
internal test leaderboard. It trained with `Prediction: <0 or 1>`,
`training.temperature=1.2`, `training.muon_learning_rate=3e-5`,
`training.learning_rate=1e-6`, and no length penalty reward. Validation scored
balanced accuracy/AUROC 0.9262, recall 0.9238, FPR 0.0714, and 1 parse error,
but fair vLLM test evaluation scored balanced accuracy/AUROC 0.8762, recall
0.7714, FPR 0.0190, 6 parse errors, and 40.7s scoring-only time. This slightly
beats the earlier vLLM GRPO rating-field row (0.8738 BA) by lowering FPR, but it
loses recall and is still below prompt-only rows such as
`qwen_reason_nothink_truth_hidden_ensemble_v1` (0.8821 BA, 29.7s),
`qwen_reason_v1` (0.9095 BA), and the 4096 three-prompt reasoning ensemble
(0.9298 BA). Treat the current RL path as evidence that RL can learn a clean
binary low-FPR judge, not as a replacement for the prompt-only/ensemble path.

As of 2026-07-11, the strongest tracked local-test black-box judge is the
GPT-OSS-120B model-swap ensemble
(`gpt_oss_120b_reason_ensemble_dks_member4096_v1`): the same `details4096`,
`known4096`, and `scrutiny4096` prompts, max aggregated, with a
validation-selected threshold of 1.0. It scored test BA 0.9345, AUROC 0.9465,
recall 0.9000, FPR 0.0310, zero parse errors, and 104.0s scoring time. It improves
only 0.0047 BA over the Qwen heavy ensemble and shares 40 errors with it, so
larger-model capacity is not the dominant remaining bottleneck; prompt-visible
ambiguity, correlated factual failures, and label noise are plausible parts of
the apparent 0.93-0.935 ceiling. GPT-OSS cannot be submitted directly.

GPT-OSS shared-error atomic-audit finding from 2026-07-11: ordinary ensemble
prompts missed 34 common deceptive test examples, but an explicit blind prompt
that decomposed every supporting assertion found a concrete false detail in
24/34. A privileged version told that an error definitely existed emitted 29
positive verdicts, 3 completions hit the 1024-token budget before a verdict, and
2 still found none; several newly induced explanations were strained, pedantic,
or referred to claims outside the assistant output. This indicates that
tangential-claim salience is a major bottleneck for GPT-OSS, while privileged
labels increase post-hoc rationalization risk. Artifacts are under
`results/blackbox/gpt_oss_120b_shared_error_atomic_audit_v1/`; diagnostic job
`30103980` and runtime code is
`experiments/blackbox/audit_gpt_oss_shared_errors.py`.

The matched base-Qwen3.5-9B atomic audit ran as job `30103986`. Its raw blind
verdict rate was 28/34 and privileged rate 30/34, but these numbers substantially
overstate factual success: manual inspection found only roughly 17--20 plausible
identifications of the actual poisoned detail. Qwen frequently attacked true
claims and invented corrections (for example reversing the Wright brothers'
birth order, denying that Mount Catherine is Egypt's highest peak, replacing
PayPal's undisclosed settlement with invented $80M/$100M figures, and denying
John Houblon was the first Bank of England governor). Decomposition activates
some useful latent knowledge in Qwen, but its fact verification is materially
less reliable than GPT-OSS and privileged labels amplify confabulation. Artifacts
are under `results/blackbox/qwen9b_shared_error_atomic_audit_v1/`.

Qwen organism neutral-self-query finding from 2026-07-11: local vLLM job
`30104103` loaded `Qwen/Qwen3.5-27B` plus the five rank-128 organism adapters
represented in the 34 shared false negatives. With the original deceptive
conversation removed, the base model emitted `CONTAINS_FALSE_DETAIL` for 23/34
and the actual generating organism for 21/34; they agreed positive on 17, agreed
negative on 5, and differed or had a parse failure on 12. Manual review shows
real organism self-corrections (for example the false PayPal settlement,
Becket's murder location, Peter Pan publication year, Chile chronology, papal
dates, and Attila/Spain), but also confident new hallucinations. The organism
LoRA suppressed useful base corrections on peladophobia prevalence, the Plain of
Jars UNESCO year, Bank of England war history, and fencing dates. Treat neutral
self-query as a promising complementary signal for context-triggered lies, not
as a calibrated standalone detector. Artifacts are under
`results/blackbox/qwen27b_organism_self_query_shared_errors_v1/`; runtime code is
`experiments/blackbox/audit_qwen_organism_self_query.py`.

The strongest submission-compatible prompt ensemble remains the three-prompt Qwen
reasoning ensemble (`qwen_reason_ensemble_dks_member4096_v1`): `details4096`,
`known4096`, and `scrutiny4096`, scheduled member-major, max aggregated, with a
4096-token generation budget and binary threshold 0.01. On the local test split
it scored balanced accuracy 0.9298, AUROC 0.9340, recall 0.8976, FPR 0.0381,
with 36 member-level parse errors and 1284.6s scoring time. The 8192-token
member-major ensemble is the validation leader at balanced accuracy 0.9250 but
is much slower and did not beat the 4096 ensemble on the tracked test split. Use
the 4096 three-prompt ensemble for Phoenix Wright accuracy-first black-box
submissions unless runtime is the primary constraint.

No-thinking Qwen judge notes from 2026-07-05: the default `blackbox_reasoning`
prompt with `enable_thinking: false` improves as token budget rises but remains
below stronger prompt/logits runs: `qwen_reason_nothink_t128_v1` scored
validation balanced accuracy 0.6750 with 366/822 parse errors and 26.3s scoring
time; `qwen_reason_nothink_t256_v1` scored 0.7726 with 134 parse errors and
29.1s; `qwen_reason_nothink_t512_v1` scored 0.8274 with 36 parse errors and
33.9s. The optimized details prompt is much better: `qwen_reason_details_nothink_t512_t01_v1`
uses the compact `details3072_speed` wording, `max_prompt_chars=3000`,
`enable_thinking=false`, `max_tokens=512`, and threshold 0.1. On validation it
scored balanced accuracy 0.8607, AUROC 0.8607, recall 0.7667, FPR 0.0452, zero
parse errors, and 12.2s scoring time (67.6 rows/s). This beats the older native
D/K logits mean run (`qwen_reason_ensemble_dk3072_logit_v1`, BA 0.8369, 24.1s)
and outcome-logits run, but not the best tracked standalone/ensemble logits
results such as `qwen_reason_details3072_speed_logit_v1` (BA 0.8786, AUROC
0.9242, 13.0s) or `qwen_reason_ensemble_dks3072_logit_v1` (BA 0.8869, AUROC
0.9271, 41.5s). Treat the no-thinking details judge as a very fast, robust
fallback/control path, not as the current accuracy leader.
On the local test split, the same `qwen_reason_details_nothink_t512_t01_v1`
method scored balanced accuracy 0.8571, AUROC 0.8570, recall 0.7643, FPR
0.0500, zero parse errors, and 12.2s scoring time (67.4 rows/s). It is the top
tracked test row among pure judge-only fast methods below the full reasoning
generation family, and it beats the earlier native D/K logits row on test
(0.8369 BA, 24.1s), while remaining below Phoenix Wright logits+n-gram blends
and the 4096-token reasoning ensemble.

For official submissions, always run `python submit.py ...` outside the sandbox:
it must reach the Hugging Face Space, NDIF, and sometimes Hugging Face. The NDIF
key was registered to team `SAIN Groningen` on the first official submission, so
subsequent runs can omit the team name and use the remembered key/team binding.
Use `--tag black` for the Phoenix Wright black-box judge submission.
Record official leaderboard results in `docs/official_submissions.md`, which is
the durable cross-method submission history for all approaches. Keep `AGENTS.md`
focused on operating guidance and add only summary pointers here.
For questions about official leaderboard status, query the Space directly instead
of inferring from local files: POST to
`https://ndif-aletheias-leaderboard.hf.space/api/me` with the `X-NDIF-API-Key`
header from `.env`/`$NDIF_API_KEY`. This requires network access outside the
sandbox; do not print the key. The response includes pending count, rate-limit
state, and per-submission history.
The Space can take a long time to return after accepting an upload and starting
the runner; once packaging/upload has clearly started, avoid repeatedly polling
or restarting the same submission just because the HTTP response is slow.
`Qwen/Qwen3.5-9B` should be accessible on the cluster for Phoenix Wright logits
judge work; do not treat an official failure as model unavailability without
direct evidence from NDIF or the leaderboard maintainers.

Official sandbox health and transfer notes from 2026-07-05: a temporary
`random_baseline.ipynb` no-NDIF submission completed successfully in 2m04s with
balanced accuracy 0.4750 and AUROC 0.4761, confirming the Space runner, dataset
loading, notebook execution, CSV scoring, and leaderboard reporting were working.
A temporary `ngrams_baseline.ipynb` no-NDIF submission using the cached
`text_probe_ngram_v1` model completed in 2m50s with balanced accuracy 0.5125 and
AUROC 0.5236. It scored very well on uncounted Metis rows and decently on
Eunomia, but was near-random on counted Notus/Iris rows, so treat Notus/Iris as
likely OOD for text-artifact n-gram probes. Do not rely on n-grams alone for a
competitive official submission; their main value is as a cheap fallback or
complementary signal.
Small CPU text-classifier fine-tunes showed the same OOD failure pattern despite
better local validation. `alminilm_baseline.ipynb`, using the best
`microsoft/MiniLM-L12-H384-uncased` Muon+AdamW checkpoint
(`results/blackbox/minilm_finetune_muon_sweep_v1`, local validation balanced
accuracy 0.8583, AUROC 0.9096), placed slightly below the n-gram official
baseline. `bge_baseline.ipynb`, using the best `BAAI/bge-small-en-v1.5`
Muon+AdamW checkpoint (`results/blackbox/bge_small_finetune_muon_sweep_v1`,
local validation balanced accuracy 0.8548, AUROC 0.9062), completed in 9m21s
with official balanced accuracy 0.5088 and AUROC 0.5131. It scored very well on
Metis and reasonably on Eunomia but near-random on counted Notus/Iris. Treat
small text-only fine-tunes as overfit-prone and not competitive standalone
methods unless a new training source or evaluation demonstrates Notus/Iris
transfer.

Local `python submit.py --dry` is only a rehearsal of the bundled runner against
`dry.yaml`; it is not the official Space environment. As of 2026-07-06, the
local `.env`/`$NDIF_API_KEY` for team `SAIN Groningen` is recognized on
`https://aletheias.api.ndif.us` with `tier_1`, so local remote-NDIF notebook
smokes can exercise competition traces when `NDIF_HOST=https://aletheias.api.ndif.us`
is set. A two-row remote NNsight smoke of `submission/phoenix_wright_v1_3.ipynb`
with `ALETHEIA_LIMIT=2` and `PHOENIX_BATCH_SIZE=2` completed successfully after
the fallback stats fix. The bundled dry runner can still fail for reasons
unrelated to the submission notebook, such as missing system-site packages in the
temporary dry-run venv or other local environment differences.

Phoenix Wright remote NNsight session finding from 2026-07-07: avoid one
`model.session(remote=True)` per generated batch. Rapid successive NDIF sessions
can stall after the first batch even when isolated remote tests pass. For
`Qwen/Qwen3.5-9B`, use `VisionLanguageModel`, matching the repo judge baseline;
do not infer `LanguageModel` is safe just because a single remote generate
succeeds. In remote tests, single-batch `LanguageModel` calls passed, but the
same multi-generate session shape hung repeatedly with `LanguageModel` and only
became reliable after switching to `VisionLanguageModel`. The tested working
generation shape is to precompute batch metadata locally, pad each batch to the
same prompt length with `padding="max_length"`, open one remote session, append
each generated-token proxy to a normal in-session Python list, then call
`torch.cat(generated_pieces, dim=0).save()` once at the end of the session.
Decode and parse the saved generated-token tensor locally after the session
exits. `torch.cat` handles a final partial batch as long as generated token
widths match, which the remote test with three prompts verified. A variant using
`generated_batches = list().save()` followed by appends hung in remote testing;
do not use that pattern. Ordinary local side effects inside a remote session
still do not survive unless they are consumed into a saved value inside the same
session.
Remote stress tests on 2026-07-07 did not reproduce an immediate OOM at the
submission tensor shape: the one-session `VisionLanguageModel` generation test
passed with 32, 64, 128, and 256 synthetic prompts, `batch_size=2`,
`max_prompt_tokens=2048`, `max_new_tokens=64`, and long filler prompts. The
128-prompt run took about 4m35s and the 256-prompt run took about 8m27s. This
lowers confidence that the latest official Eunomia failure was a simple
per-batch/prompt OOM, but it does not rule out a larger whole-dataset
session/resource/time/backend limit in the Space. Per-batch OOM probing at
`max_prompt_tokens=2048`, `max_new_tokens=64`, and long filler prompts found
batch sizes 4, 8, 16, and 24 passed, while 28 and 32 failed with explicit CUDA
OOMs on NDIF. Treat 24 as the highest tested passing batch and keep official
submissions much lower unless retested.
Phoenix Wright notebook default batch size was raised from 2 to 16 on
2026-07-07 to reduce the number of `generate` calls inside the single NDIF
session. The batch-16 shape passed a synthetic remote test with 17 prompts,
`max_prompt_tokens=2048`, `max_new_tokens=64`, and long filler prompts, and the
actual notebook passed the local notebook runner on 17 dev Metis rows in about
59s. This covers one full batch plus a final partial batch.
Phoenix Wright tokenizer hardening from 2026-07-07: ordinary weird text
(nulls, RTL controls, emoji/ZWJ, combining marks, private-use chars, huge text,
XML-ish delimiter injection, `None`, numeric/list/dict content) tokenized
successfully, but lone UTF-16 surrogate code points caused the Qwen tokenizer to
raise `TypeError: TextEncodeInput must be ...`. The notebook now normalizes
prompt role/content/reasoning text through `phoenix_wright_text.safe_text`, which
UTF-8 encodes/decodes with replacement before chat templating/tokenization.
The same helper module also hardens malformed message structure before prompt
construction: `None`, scalar/string dialogues, dict-as-dialogue rows, and mixed
message lists containing non-dict items are normalized before the notebook reads
roles/content/reasoning. This prevents `.get`/`reversed` failures before the
model call while still allowing genuine NDIF/generation errors to fail loudly.

Keep the submission package pruned: `submit.py` excludes `.env`, `.uv-cache`,
`results/`, `logs/`, and `dev_splits/` so credentials and local experiment
artifacts are not uploaded. Before submitting, verify there is exactly one
notebook in `submission/` and that the packaged zip is small.
