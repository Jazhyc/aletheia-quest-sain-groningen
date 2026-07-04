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
Multiple Slurm jobs landing on the same GPU node does not by itself imply GPU
contention; the scheduler may assign different GPUs on that node. Do not cancel
or requeue jobs solely because they share a node. Treat contention as real only
when there is direct evidence such as CUDA OOMs, scheduler allocation conflicts,
or process/GPU telemetry showing multiple jobs on the same device.

As of 2026-07-04, the strongest tracked black-box judge is the three-prompt Qwen
reasoning ensemble (`qwen_reason_ensemble_dks_member4096_v1`): `details4096`,
`known4096`, and `scrutiny4096`, scheduled member-major, max aggregated, with a
4096-token generation budget and binary threshold 0.01. On the local test split
it scored balanced accuracy 0.9298, AUROC 0.9340, recall 0.8976, FPR 0.0381,
with 36 member-level parse errors and 1284.6s scoring time. The 8192-token
member-major ensemble is the validation leader at balanced accuracy 0.9250 but
is much slower and did not beat the 4096 ensemble on the tracked test split. Use
the 4096 three-prompt ensemble for Phoenix Wright accuracy-first black-box
submissions unless runtime is the primary constraint.

For official submissions, always run `python submit.py ...` outside the sandbox:
it must reach the Hugging Face Space, NDIF, and sometimes Hugging Face. The NDIF
key was registered to team `SAIN Groningen` on the first official submission, so
subsequent runs can omit the team name and use the remembered key/team binding.
Use `--tag black` for the Phoenix Wright black-box judge submission.
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

Local `python submit.py --dry` is only a rehearsal of the bundled runner against
`dry.yaml`; it is not the official Space environment. On this cluster it may fail
or hang for reasons unrelated to the submission notebook, including local NDIF key
scope, missing system-site packages in the temporary dry-run venv, or network
access differences. Prefer static notebook checks plus the real Space submission
when the method depends on competition-only NDIF execution.

Keep the submission package pruned: `submit.py` excludes `.env`, `.uv-cache`,
`results/`, `logs/`, and `dev_splits/` so credentials and local experiment
artifacts are not uploaded. Before submitting, verify there is exactly one
notebook in `submission/` and that the packaged zip is small.
