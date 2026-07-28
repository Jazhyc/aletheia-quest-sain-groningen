# Phoenix backend parity

This diagnostic scores one full 400-row public varied-deception organism,
`aletheias-quest/dev-varied-deception-Qwen3.5-27B-None`, with the exact Phoenix
Wright 4.0 visible-message renderer and direct `Prediction:` margin.

Its local train, validation, and test label partitions are recombined only for
this backend-parity diagnostic.

The vLLM run scores base Qwen, the pre-migration Phoenix 2.0 revision
`cb1d515...`, migrated `main`, and the re-keyed local Phoenix v2.1 adapter
(`1407d885...`) in one model process. This explicitly tests whether each layout
is applied or silently ignored. The NDIF run uses migrated `main`.

```bash
sbatch experiments/phoenix_backend_parity/run_vllm.sh

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
python experiments/phoenix_backend_parity/run.py --backend ndif

python experiments/phoenix_backend_parity/compare.py
python experiments/phoenix_backend_parity/compare_historical.py

# Optional exact current-loader replay of the historical validation adapter.
sbatch experiments/phoenix_backend_parity/run_historical_validation.sh
```

`compare_historical.py` joins the current re-keyed local-v2.1 condition against
the archived Phoenix 4.0 validation/test direct-margin rows, reports calibration
and rank correlations, and counts positive-negative pair-order transitions.
The optional Slurm replay reconstructs the historical adapter directory with
the converted current-loader weights before invoking the frozen continuous
margin evaluator.
