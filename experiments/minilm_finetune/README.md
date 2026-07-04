# MiniLM Fine-Tune

Black-box text classifier experiment for `microsoft/MiniLM-L12-H384-uncased`.
It fine-tunes on `dev_splits/dry.train.yaml`, selects hyperparameters, epoch,
and threshold on `dev_splits/dry.validation.yaml`, and writes artifacts under
`results/blackbox/minilm_finetune_v1/`.

Run on Slurm:

```bash
sbatch experiments/minilm_finetune/run_minilm_finetune.sh
```

Broader AdamW sweep over views, sequence lengths, learning rates, epochs, and
class weighting:

```bash
sbatch experiments/minilm_finetune/run_minilm_finetune.sh \
  --output-dir results/blackbox/minilm_finetune_adamw_sweep_v2 \
  --views output,dialogue,output_context \
  --max-lengths 128,256,384,512 \
  --learning-rates 3e-5,4e-5,5e-5,6e-5 \
  --epochs-grid 2,3 \
  --class-weight-options true,false \
  --optimizers adamw
```

Muon plus AdamW comparison. Muon is applied to hidden 2D weight matrices; AdamW
handles embeddings, classifier head, biases, and normalization weights:

```bash
sbatch experiments/minilm_finetune/run_minilm_finetune.sh \
  --output-dir results/blackbox/minilm_finetune_muon_sweep_v1 \
  --views output,dialogue,output_context \
  --max-lengths 256,384,512 \
  --learning-rates 1e-5,3e-5 \
  --muon-learning-rates 3e-4,1e-3,3e-3 \
  --epochs-grid 2,3 \
  --class-weight-options true,false \
  --optimizers muon_adamw
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
