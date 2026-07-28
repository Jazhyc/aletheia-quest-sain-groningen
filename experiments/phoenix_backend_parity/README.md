# Phoenix backend parity

This diagnostic scores one full 400-row public varied-deception organism,
`aletheias-quest/dev-varied-deception-Qwen3.5-27B-None`, with the exact Phoenix
Wright 4.0 visible-message renderer and direct `Prediction:` margin.

Its local train, validation, and test label partitions are recombined only for
this backend-parity diagnostic.

The vLLM run scores base Qwen, the pre-migration Phoenix 2.0 revision
`cb1d515...`, and migrated `main` in one model process. This explicitly tests
whether either layout is applied or silently ignored. The NDIF run uses
migrated `main`.

```bash
sbatch experiments/phoenix_backend_parity/run_vllm.sh

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
python experiments/phoenix_backend_parity/run.py --backend ndif

python experiments/phoenix_backend_parity/compare.py
```
