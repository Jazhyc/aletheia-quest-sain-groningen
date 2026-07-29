#!/usr/bin/env python3
"""Publish the current Phoenix 6.3 main adapter to Hugging Face.

This compatibility entry point delegates to the validated Phoenix 6.3 uploader.
It deliberately no longer targets the historical Luna repository now that the
bundled ``main/`` checkpoint contains the Qwen-397B soft-distillation adapter.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.upload_qwen397_tvg_adapter import (
    main,
)


if __name__ == "__main__":
    main()
