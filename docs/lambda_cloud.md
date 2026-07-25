# Lambda Cloud Campaign Runner

`scripts/lambda_cloud.py` manages a named Lambda Cloud GPU instance that stays
alive across an experiment campaign. The script never automatically terminates
an instance after an experiment. Termination is a separate, explicit command
that requires `--yes`.

Lambda's REST API is used for instance lifecycle operations. SSH and `rsync` are
used for experiment execution and file transfer, so `LAMBDA_API_KEY` remains on
the local system and is never copied to the cloud instance.

## Local configuration

Put credentials in the root `.env`, which is ignored by Git:

```bash
LAMBDA_API_KEY=...
LAMBDA_SSH_KEY_PATH=/home/example/.ssh/id_ed25519
LAMBDA_SSH_KEY_NAME=aletheia-habrok
```

The latter two settings are optional. The private-key path defaults to
`~/.ssh/id_ed25519`, and the script can infer a uniquely matching registered key
name.

Register this system's public key once:

```bash
.venv/bin/python scripts/lambda_cloud.py register-key \
  --name aletheia-habrok

.venv/bin/python scripts/lambda_cloud.py doctor
```

Only the public key is uploaded. The private key remains local.

## Discover and launch

Capacity is dynamic, so inspect it immediately before launching:

```bash
.venv/bin/python scripts/lambda_cloud.py types \
  --available-only --min-gpu-memory 40
```

Inspect available base images in the target region as well:

```bash
.venv/bin/python scripts/lambda_cloud.py images --region us-west-2
```

Launch exactly one named campaign after reviewing the returned hourly price:

```bash
.venv/bin/python scripts/lambda_cloud.py launch \
  --campaign qwen-prompt-sweep \
  --instance-type gpu_1x_a100_sxm4 \
  --region us-west-2 \
  --image-family lambda-stack-24-04 \
  --yes

.venv/bin/python scripts/lambda_cloud.py wait \
  --campaign qwen-prompt-sweep

.venv/bin/python scripts/lambda_cloud.py probe \
  --campaign qwen-prompt-sweep

.venv/bin/python scripts/lambda_cloud.py compute-probe \
  --campaign qwen-prompt-sweep
```

`launch` is idempotent by campaign name: if the campaign instance already
exists, it reports that instance instead of launching another billable GPU.
Non-x86 instance types require `--allow-non-x86` because Python and CUDA wheel
compatibility must be checked explicitly.

## Connect and transfer files

Open an interactive connection:

```bash
.venv/bin/python scripts/lambda_cloud.py ssh \
  --campaign qwen-prompt-sweep
```

Run a non-interactive command:

```bash
.venv/bin/python scripts/lambda_cloud.py ssh \
  --campaign qwen-prompt-sweep -- nvidia-smi
```

Copy the committed `HEAD` snapshot to `~/Aletheias-Quest-Competition`:

```bash
.venv/bin/python scripts/lambda_cloud.py sync-commit \
  --campaign qwen-prompt-sweep

.venv/bin/python scripts/lambda_cloud.py bootstrap \
  --campaign qwen-prompt-sweep
```

`sync-commit` excludes all uncommitted and ignored content, including `.env`,
environments, caches, datasets, logs, and results. This is the preferred
reproducible campaign transfer. `sync-code` transfers the working tree but
refuses a dirty tree unless `--include-uncommitted` is explicitly supplied.

`bootstrap` installs a managed Python 3.12 with `uv`, performs `uv sync
--locked`, and checks that PyTorch and vLLM can see the GPU. Copy any required
ignored input artifact explicitly:

```bash
.venv/bin/python scripts/lambda_cloud.py push \
  --campaign qwen-prompt-sweep \
  --local-path results/blackbox/source-cache
```

Pull a completed result directory back into the same repository path:

```bash
.venv/bin/python scripts/lambda_cloud.py pull \
  --campaign qwen-prompt-sweep \
  --remote-path results/blackbox/qwen-example-v1 \
  --directory
```

Runtime logs should use `logs/lambda/<method>/` on both systems. Experiment
artifacts remain under `results/blackbox/<method>/`.

## Selected credentials

Most experiments need only a Hugging Face read token and, optionally, W&B:

```bash
.venv/bin/python scripts/lambda_cloud.py sync-secrets \
  --campaign qwen-prompt-sweep \
  --name HF_TOKEN \
  --name WANDB_API_KEY
```

The command sends values over SSH standard input and writes them to
`~/.config/aletheia/secrets.env` with mode `600`. Values are never included in
command arguments or output. Source that file before an experiment:

```bash
source ~/.config/aletheia/secrets.env
```

Only `HF_TOKEN`, `WANDB_API_KEY`, and `WIKIMEDIA_ACCESS_TOKEN` are allowlisted.
The Lambda lifecycle key and the competition's NDIF key are always rejected.

## End the campaign

First pull and inspect all required artifacts. Then explicitly terminate the
campaign:

```bash
.venv/bin/python scripts/lambda_cloud.py terminate \
  --campaign qwen-prompt-sweep \
  --yes
```

Linux shutdown or poweroff does not stop Lambda billing. Only API or console
termination ends instance billing. Instance-local storage is erased when the
instance terminates.

## References

- [Lambda Cloud API](https://docs.lambda.ai/public-cloud/cloud-api/)
- [Connecting with SSH](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)
- [Importing and exporting data](https://docs.lambda.ai/public-cloud/importing-exporting-data/)
- [Instance lifecycle](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/)
