# Bundled runner dependencies

`transformers-5.15.0.dev0-py3-none-any.whl` was built from the Hugging Face
Transformers repository at commit
`c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d`, the exact revision deployed on
NDIF and announced by the organizers on 2026-07-23.

`nnsight-0.7.1.dev41+gd901da3ed-cp312-cp312-linux_x86_64.whl` was built with
Python 3.12 from the NNsight `hackathon/peft` branch at commit
`d901da3ed772c815d0d146136de6f7e35f913221`, which includes the remote LoRA
fix announced in the same update.

Wheel SHA-256 values:

- Transformers: `0dbdde6331c4562d429467c41c9d2dac641db6a8bd36dab97771a7ebfb6db83b`
- NNsight: `831bdcd9ad90a8e5ed94348f36e011d8aaa0002d9e570c955169efcd60d0b6ae`

The wheels are bundled because earlier leaderboard dependency-install attempts
could not reach GitHub. Their upstream licenses are included inside the wheels.
