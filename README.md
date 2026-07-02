<p align="center">
  <img src="assets/aletheia-logo.png" alt="Aletheia's Quest" width="240">
</p>

<h1 align="center">Aletheia's Quest — Competition Submission Repo</h1>

[![Apply to compete](https://img.shields.io/badge/Apply%20to%20compete-aletheias--quest.github.io-6f42c1?style=for-the-badge)](https://aletheias-quest.github.io/)
&nbsp;
[![Live leaderboard](https://img.shields.io/badge/Live%20leaderboard-ndif--aletheias--leaderboard-2ea44f?style=for-the-badge)](https://ndif-aletheias-leaderboard.hf.space/)
&nbsp;
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/MfjQbAErEG)
&nbsp;
[![Context7 docs](https://img.shields.io/badge/Context7-docs%20for%20agents-0b0b0b?style=for-the-badge)](https://context7.com/ndif-team/aletheias-quest-competition)

Build a **deception detector**: for each model conversation in a private eval
set, output the probability that the assistant's final message is _deceptive_.
You build your detector with [nnsight](https://nnsight.net) and run model traces
remotely on [NDIF](https://ndif.us).

> **Submissions.** After you submit your lie detection notebooks, watch your team's standing on the **[live leaderboard](https://ndif-aletheias-leaderboard.hf.space/)**.

> 💬 **Questions?** Join the **[Discord](https://discord.gg/MfjQbAErEG)** — ask the
> devs and other competitors anything about nnsight, NDIF, the data, or your method.

> 📅 **Office hours** — bring questions about NNsight, NDIF, or your submission to us on **Tuesdays & Thursdays at 1 PM EDT** on Zoom:
> **[northeastern.zoom.us/my/jadenfk](https://northeastern.zoom.us/my/jadenfk)**.

> 🤖 **Compete with an AI agent.** This repo ships an [`llms.txt`](llms.txt) that
> teaches a coding agent the whole competition — the data, the `util.py` toolkit,
> NDIF/nnsight, and how to craft, debug, and submit a method. It's also indexed on
> **[Context7](https://context7.com/ndif-team/aletheias-quest-competition)**, so an
> agent can pull up-to-date docs on demand. Open the repo in an agent and just ask.
> See **[Build with an agent](#build-with-an-agent)**.

## The competition at a glance

> Summarized from the official site — **[aletheias-quest.github.io](https://aletheias-quest.github.io/)**
> is authoritative and details may change. Apply via the
> **[application form](https://forms.office.com/r/BXVwPEweJi)**; questions to
> **competition@cadenzalabs.org** or the Discord.

**The task.** Detect when an LLM _lies_ — "the assistant lies if it states something it
believes to be false." You get **black-box and white-box** access (query the model, and
read/write activations at any layer) to a suite of LLMs (**"model organisms"**) induced to
lie in different ways, served on NDIF.

**Timeline (2026).** Applications open **Jun 4**, close **Jun 21** (rolling). Competition runs
**Jun 29 → Jul 26** (main submission deadline). **Bonus week** (a surprise task) runs **Jul 27 →
Jul 31**. Invite-only **awards event Aug 25** in Boston.

**Prizes — $50,000 total** (split evenly among a team's members; a team can win in multiple
categories — prizes stack). Categories: **white-box** methods (1st/2nd/3rd), **black-box**
methods (1st/2nd/3rd), **weekly leaderboard** (1st/2nd/3rd), and **judge's awards** (Novelty,
Scalability). **You must rank above the 4 baseline methods to win.**

**Scoring.** Ranked by **mean balanced accuracy** across the held-out datasets; **average AUROC**
breaks ties. Scored per _(model organism, dataset)_ unit, averaged across all datasets.

**Eligibility.** 18+; not a US-sanctioned/restricted party; not an organizer or dataset builder;
one team per person. Winners pass Northeastern's Restricted Party Screening; government employees
need an employer release before any award.

**Method rules (read these — they constrain your approach).** _Allowed:_ querying models via
NDIF/nnsight, reading/writing activations at any layer, using weights + LoRA adapters, **tuning a
detector's _parameters_ per organism (with documentation)**, and prompts motivated by a _general_
hypothesis. _Prohibited:_ exfiltrating held-out test data; reading model design docs/training
data; **submitting fundamentally different detection _logic_ per organism**; prompts crafted from
organism-specific behavioral analysis; **external API calls at inference time**; **fine-tuning the
hosted models**; or exploiting organism-specific tricks that wouldn't generalize.

**What you submit.** One Jupyter notebook (one detector per organism) + trained parameters /
supplemental data, a **report** explaining your method and per-organism choices, **reproducible
training code**, and a compliance form with your team number. Package ≤ **200 MB**; daily
submissions may be rate-limited. All submissions are published under the **MIT license** (you keep
authorship credit; organizers retain a perpetual royalty-free license to publish them).

**Organized by** Cadenza Labs, NDIF (Northeastern), and Schmidt Sciences — leads Walter Laurito &
Kieron Kretschmar (Cadenza Labs), Jaden Fiotto-Kaufman (NDIF), Peter Hase (Schmidt Sciences),
advised by David Bau (NDIF Director).

## Quick start

1. **Clone** this repo.
2. **Set up a local dev environment** (Python 3.12). The quickest path is the
   bundled script, which uses [`uv`](https://docs.astral.sh/uv/) to create
   `./.venv` and install everything needed for development, GPU-assisted probe
   training, and `python submit.py --dry`:
   ```bash
   module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
   ./setup_dev.sh && source .venv/bin/activate
   ```
   On the Hábrók/RUG cluster, load those modules before syncing manually as
   well; the system `/usr/bin/python3.12` does not include Python headers needed
   to build the competition's `nnsight` branch. The setup script loads the
   modules automatically when the `module` command is available.

   The root [`pyproject.toml`](pyproject.toml) pins Python 3.12, includes the
   competition's **nnsight** `hackathon/peft` branch, and installs local research
   tools including **nnterp** and **vLLM**. The `uv.lock` file pins the resolved
   Git commit and transitive packages for repeatable notebook work. To run without
   activating the environment:
   ```bash
   uv run python submit.py --dry
   ```
   `submission/requirements.txt` is still the leaderboard runtime dependency
   contract. Keep training-only packages such as `vllm` out of it unless the
   submitted notebook imports them at inference time.

   On shared clusters, put large caches on scratch rather than the home
   filesystem before syncing or downloading models:
   ```bash
   export SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
   export HF_HOME="$SCRATCH_DIR/.huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   export HF_DATASETS_CACHE="$HF_HOME/datasets"
   export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
   ```
   Add these to your shell startup file (for example `~/.bashrc`) so notebooks,
   `uv sync`, `huggingface-cli`, and `datasets.load_dataset` all reuse the same
   scratch-backed caches.
3. For local development, `hf auth login` so your HF token is available.
   You can also keep local-only credentials in a git-ignored `.env` file:
   ```bash
   NDIF_API_KEY=your-ndif-key
   HF_TOKEN=your-hf-token
   WANDB_API_KEY=your-wandb-key
   ```
   Load it before notebooks, dry runs, or training jobs:
   ```bash
   set -a
   source .env
   set +a
   ```
   `NDIF_API_KEY` is required for remote nnsight traces. `HF_TOKEN` is only
   needed for gated/private Hugging Face access, and `WANDB_API_KEY` is only
   needed if you enable experiment logging.
4. Look at [`submission/example.ipynb`](submission/example.ipynb) — a minimal
   working baseline detector that follows the contract; replace the contents of the `detector` method to user the lie classification method you developed it with your detector.
5. Replace `
`example.ipynb`, rename or delete it so only your notebook remains.
6. **Verify locally first** (no upload) — runs your notebook end-to-end in a venv
   (installing your `requirements.txt`) against the datasets listed in
   [`dry.yaml`](dry.yaml) and prints a score, so you catch broken deps / output
   before submitting. **Edit `dry.yaml` to rehearse on different datasets** — each
   entry is an inputs repo plus its labels repo. On failure it prints the **full
   error/traceback** — this is where you debug, because the leaderboard itself only
   returns a generic error (see _Execution sandbox_):
   ```bash
   export NDIF_API_KEY="your-ndif-key"
   python submit.py --dry
   # faster partial rehearsal — score only the first 32 rows of each dataset:
   python submit.py --dry --limit 32
   ```
   `--limit N` is forwarded to your notebook as `$ALETHEIA_LIMIT` (see how
   `submission/example.ipynb` reads it); omit it to score every row. It works on a
   real submit too, but the leaderboard sets no limit, so a full submission always
   scores every row.
7. Submit for real — give your **team name on the first submission only**, and
   **tag your method** `--tag white` (white-box: uses activations/weights) or
   `--tag black` (black-box: query-only). The tag shows as a badge on the
   leaderboard and is filterable (it maps to the white-box / black-box prize
   categories); omit it to stay untagged.
   ```bash
   python submit.py --team "your-team-name" --tag white --ndif-api-key <YOUR_NDIF_KEY> --space-url <LEADERBOARD_SPACE_URL>
   # afterwards, just:  python submit.py --tag white --ndif-api-key <YOUR_NDIF_KEY> --space-url <LEADERBOARD_SPACE_URL>
   ```

**To submit you need:** an **NDIF API key** (`$NDIF_API_KEY` or `--ndif-api-key`) —
always required — and the leaderboard Space URL (`--space-url` or
`$ALETHEIA_SPACE_URL`). Your **NDIF key is your identity**: the first time you
submit, your `--team` name is bound to it (and must be unused by anyone else);
after that the key alone identifies your team, so you needn't pass `--team` again.

**Giving nnsight your key (local development).** On the leaderboard your key is set
for you. To run nnsight traces **locally** (tutorials, debugging your method),
nnsight needs your NDIF key — it reads the **`NDIF_API_KEY`** environment variable
automatically, so `export NDIF_API_KEY="your-ndif-key"` is enough. Or set it from
Python:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("your-ndif-key")   # saves it to nnsight's config (persists)
# or, just for this session:  CONFIG.API.APIKEY = "your-ndif-key"
```

**Submission limit & standing.** There is a per-team **submission rate limit** of one submission per 12 hours (**subject to change**; over it you get a clear "try again in …" message). Runs that *error* still spend the rate limit, so make sure to run `--dry` until it's clean. Submissions rejected up front (e.g. a malformed package, or due to rate-limiting) cost nothing. Check the **Entrant's Desk** on the [leaderboard page](https://ndif-aletheias-leaderboard.hf.space/) (enter your NDIF key) to see your team, best score, **attempts remaining**, and your submission history.

## Build with an agent

This repo is set up to be driven by a coding agent (e.g.
[Claude Code](https://claude.com/claude-code) or any other). The bundled
[`llms.txt`](llms.txt) is a full briefing on the competition — the datasets and
models, the `submission/util.py` toolkit, how NDIF/nnsight work, the submission
contract, and a table of the errors people actually hit — so the agent can help you
go from the example to a real method.

The repo is also indexed on **[Context7](https://context7.com/ndif-team/aletheias-quest-competition).** An agent with Context7 (e.g. via its MCP server) can pull up-to-date, indexed docs and code snippets from this repo and `nnsight` on demand, instead of relying on stale training data. The **[leaderboard page](https://ndif-aletheias-leaderboard.hf.space/)** also embeds a Context7 chat you can ask directly.

**To point an agent at it:**

1. Clone the repo and open it in your agent, then tell it to **read `llms.txt`** first
   (some agents also auto-load it if you symlink/copy it to `CLAUDE.md` or `AGENTS.md`).
   Agents with **Context7** can also fetch this repo's docs at
   [`context7.com/ndif-team/aletheias-quest-competition`](https://context7.com/ndif-team/aletheias-quest-competition).
2. Make sure your `NDIF_API_KEY` is set in the environment so it can run `--dry`.
3. Ask. For example:
   - _"Read llms.txt, then turn `submission/example.ipynb` into a layer-15
     activation probe and run `python submit.py --dry`."_
   - _"My submission fails with a generic error — help me debug it."_
   - _"Explain how `run_full_session` works and write a `detect` that uses an
     LLM-judge prompt."_

The agent should use `util.py`, rehearse with `--dry`, and read tracebacks to fix
issues — but **you** still need to be an official participant with a valid NDIF key,
and you're responsible for what you submit. When the agent is unsure about NDIF or
nnsight behavior, it'll point you to the tutorials, [nnsight.net](https://nnsight.net),
the [Discord](https://discord.gg/MfjQbAErEG), or office hours — and so should you.

## The contract

The runner executes the **single `.ipynb` in `submission/`** top-to-bottom — a
submission is **one notebook at a time**, so if `submission/` holds more than one
`.ipynb` the submission is rejected. Before running, it sets these environment
variables:

| Variable       | Meaning                                                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DATASET_NAME` | HF dataset id to predict on (the private eval set)                                                                                                           |
| `NDIF_HOST`    | The competition's NDIF cluster, `https://aletheias.api.ndif.us` — set for you; nnsight reads it, so **you don't configure a host** (just pass `remote=True`) |
| `NDIF_API_KEY` | Your NDIF key, forwarded from your submission (nnsight reads it)                                                                                             |
| `HF_TOKEN`     | Your HF token, forwarded so you can load gated HF models you can access                                                                                      |

Load the data with `load_dataset(DATASET_NAME, split="test")` — read `DATASET_NAME`
from the environment, don't hard-code it. (It's predownloaded and read offline; only
`DATASET_NAME` is available at runtime.)

**All model compute must run on NDIF.** The leaderboard has **no GPU**, so your
forward passes have to be remote nnsight traces (`remote=True`; your `NDIF_API_KEY` is
already set). Developing locally you can run however you like — but the submitted
notebook must use NDIF.

Your notebook must **write `submission.csv`** in the repo root:

```
index,deceptive,score
0,True,0.83
1,False,0.12
2,True,0.97
...
```

- `index` — the row's `index` from the dataset (joins your row to the held-out
  label; use the column, not row position).
- `deceptive` — your **binary** call, `True`/`False`.
- `score` — a continuous deception score in `[0, 1]` you'd threshold to get
  `deceptive`.
- **Exactly one row per example.** That's all the grader reads.

The eval inputs are **label-free**; you are scored by joining your rows (on
`index`) against held-out labels you never see. Per dataset we report four metrics:
**balanced accuracy**, **recall** and **false-positive rate** (from your binary
`deceptive` call, with "deceptive" the positive class) and **AUROC** (from the
continuous `score`). You're evaluated on several held-out datasets; your headline
numbers are the **average across datasets**, and the leaderboard ranks by mean
**balanced accuracy** (click a row to see every metric per dataset). It also shows
your total runtime.

## Dependencies (`submission/requirements.txt`)

Put a `requirements.txt` in [`submission/`](submission/) (next to your notebook).
Before your notebook runs, the runner
installs it into a per-job virtualenv (with `--system-site-packages`) on top of
the base image, so you only list what you _add_. The base already includes
`datasets`, `numpy`, `pandas`, `scikit-learn`, `nbclient`, **`torch` (CPU)**,
**`transformers`**, and **`nnsight`** (the `hackathon/peft` branch — see step 3) —
so a standard nnsight + NDIF probe needs no extra deps. (Installs reach **PyPI
only**; see the egress note below.)

**Note on NDIF-side packages.** Code that runs **inside an NDIF trace/session** (your
`detect_fn`, anything `.save()`'d remotely) executes on NDIF, where only a
**whitelisted set of packages** is available — not whatever you put in
`requirements.txt` (that installs in your _local_ sandbox, not on NDIF). If your
method needs a library NDIF doesn't have, either **keep that part local** and make a
separate NDIF call for just the model compute (e.g. pull activations back, then run
your library locally), or **ask the devs on the [Discord](https://discord.gg/MfjQbAErEG)**
to add the package to the NDIF environment.

## Execution sandbox

Your notebook runs in an isolated sandbox (filesystem-confined, syscall-filtered,
resource-limited). Implications for your code:

- The eval dataset is **predownloaded**; `load_dataset(DATASET_NAME, …)` reads it
  offline (you can't load other datasets from the Hub at runtime).
- Your **`NDIF_API_KEY`** and **`HF_TOKEN`** are forwarded into the run (set them
  before `submit.py`, or pass `--ndif-api-key`/`--hf-token`), so nnsight
  authenticates remote traces and you can load gated HF models you have access to.
- `LanguageModel("…")` works: `huggingface.co` is reachable for model
  config/tokenizer. Run the weights on NDIF with `remote=True` (don't dispatch
  them locally).
- Network egress is allowlisted by hostname to the NDIF cluster
  (`aletheias.api.ndif.us`, under `api.ndif.us`) + its results bucket and
  `huggingface.co`/`hf.co`; everything else is blocked. HF is read-only (model
  configs/tokenizers download fine; you can't upload).
- You may only write under the working directory; CPU/memory/time are capped.
- **If your run fails here, the leaderboard returns a _generic_ error** (the real
  error could leak the private eval data). To see the actual traceback, reproduce
  with `python submit.py --dry` — same pipeline, the datasets in `dry.yaml`, full errors.

`--dry` runs the real pipeline (venv → `requirements.txt` → your notebook → NDIF
traces → scoring), so it catches dependency, code, NDIF, and output errors. It
does **not** apply the network/filesystem confinement above (so it stays portable)
— a notebook that reaches a non-allowlisted host or writes outside its directory
passes locally but fails here. Keep your code to NDIF + HF and the working dir.

## Notes

- **All model compute must run on NDIF** (`remote=True` in your traces) — the
  leaderboard has no GPU. (When developing locally, run however you like.)
- You can include extra files (probe weights, helper modules) in the repo — they
  ship with your submission. Keep the total package reasonable (< 200 MB).
- `submit.py` packages the repo and POSTs it to the leaderboard Space; set the
  Space URL via `--space-url` or the `ALETHEIA_SPACE_URL` environment variable.

## Private dev splits

The public dev datasets are labelled, but each repo is a single `test` split.
For local method development, make a deterministic 80/10/10 split and score
against it without changing the official submission contract:

```bash
python scripts/make_dev_splits.py --dry-yaml dry.yaml --out-dir dev_splits --seed 0
python scripts/dev_leaderboard.py --method "my-method" --split validation
python scripts/dev_leaderboard.py --method "my-method" --split test
```

`make_dev_splits.py` stratifies independently within each dataset and label, so
every split keeps the same model/LoRA organism coverage as the source dev set as
closely as the row counts allow. It writes local label subsets and
`dev_splits/dry.train.yaml`, `dev_splits/dry.validation.yaml`, and
`dev_splits/dry.test.yaml`.

Use `train` for fitting, `validation` for hyperparameter and threshold choices,
and reserve `test` for the final local check. `dev_leaderboard.py` runs the same
notebook pipeline as `--dry`, scores only the chosen label subset, and appends a
private JSONL history to `dev_splits/results.jsonl`. The whole `dev_splits/`
directory is git-ignored.

## Black-box experiments

For fast prompt and threshold iteration, use the Hydra-driven judge runner rather
than editing a submission notebook for every trial:

```bash
python experiments/blackbox/run_judge.py split=validation method=qwen_judge_v1
python experiments/blackbox/run_judge.py split=validation method=qwen_3shot shots.n_per_label=3
python experiments/blackbox/run_judge.py split=validation scoring.threshold=0.45 judge.rating_max=5
```

It reads `dev_splits/dry.<split>.yaml`, scores only that split's indexed rows,
writes per-dataset predictions under `logs/blackbox/`, and appends a local JSONL
ledger to `logs/blackbox/results.jsonl`. Few-shot examples, when enabled, are
sampled only from `shots.split` (`train` by default), so validation/test labels
are not used in the prompt. Development and training runs use local GPU inference
through vLLM; reserve NDIF for leaderboard evaluation/submission execution.

On Slurm, start with direct `sbatch` so cluster failures stay visible:

```bash
sbatch experiments/blackbox/run_judge.sh split=validation method=qwen_judge_v1
sbatch experiments/blackbox/run_judge.sh split=validation method=qwen_3shot shots.n_per_label=3
```

The Slurm template requests one `gpushort` GPU node, one CPU, 32 GB memory, and
writes job output to `logs/slurm/%x-%j.out`. Hydra still gives config overrides
and sweep-friendly command lines. Add `submitit` later only if we need
Python-managed Slurm arrays; direct `sbatch` is usually easier to debug.
