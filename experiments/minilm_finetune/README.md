# MiniLM Fine-Tune

Black-box text classifier experiment for `microsoft/MiniLM-L12-H384-uncased`.
It fine-tunes on `dev_splits/dry.train.yaml`, selects hyperparameters, epoch,
and threshold on `dev_splits/dry.validation.yaml`, and writes artifacts under
`results/blackbox/minilm_finetune_v1/`.

Run on Slurm:

```bash
sbatch experiments/minilm_finetune/run_minilm_finetune.sh
```

Useful smaller smoke run:

```bash
python experiments/minilm_finetune/run_minilm_finetune.py \
  --cpu \
  --views output \
  --max-lengths 128 \
  --learning-rates 2e-5 \
  --epochs-grid 1 \
  --train-batch-size 8 \
  --eval-batch-size 16
```

Default Slurm resources follow the repository black-box convention: one
`gpushort` node with `--gpus-per-node=rtx_pro_6000:1`, one CPU, and 32 GB RAM.
The trainer uses bf16 autocast on CUDA unless `--no-bf16` is passed.
