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

Before agent-assisted work, read `llms.txt` for competition context and check `README.md` for the current submission contract. Preserve the single-notebook rule in `submission/` and rehearse changes with `--dry` whenever possible. When developing a new competition method, create and work on a separate feature branch instead of `master`. Add or update tests for non-trivial code changes, then run the relevant validation command. After completing a coherent feature or fix, commit the finished work with a short imperative message.

For development and training experiments, do not use NDIF; reserve NDIF for leaderboard evaluation/submission execution. Use local GPU Slurm jobs with vLLM for black-box judge experiments. Keep experiment code organized under `experiments/<method>/`, keep Slurm shell templates as `.sh` files, and write runtime logs under `logs/` (`logs/slurm/%x-%j.out` for Slurm output). Store black-box experiment artifacts under `results/blackbox/`; per-run result directories are ignored, but `results/blackbox/leaderboard.md` is tracked and should show test-set results only. Default Slurm resources for these jobs are one `gpushort` GPU node with `--gpus-per-node=rtx_pro_6000:1`, `--cpus-per-task=1`, and 32 GB memory unless the user says otherwise. For interactive black-box method development, prefer reserving a GPU node with `srun`, running one persistent vLLM server process, and pointing experiment clients at it over HTTP so model loading/compilation is paid once per session. Prefer direct `sbatch` templates first for unattended runs; add `submitit` only if Python-managed Slurm arrays become necessary. Slurm controller commands such as `sbatch`, `squeue`, `scancel`, and `sacct` should be run outside the sandbox automatically because sandboxed attempts cannot contact the controller.
