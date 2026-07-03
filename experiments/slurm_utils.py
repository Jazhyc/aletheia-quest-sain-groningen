"""Common utilities for SLURM job submission."""

from __future__ import annotations

import subprocess
from pathlib import Path


def create_logs_dir() -> Path:
    """Ensure the logs directory exists for SLURM output."""
    logs_dir = Path("logs/slurm")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def create_method_logs_dir(method: str) -> Path:
    """Ensure a method-specific SLURM logs directory exists."""
    safe_method = "".join(c if c.isalnum() or c in "_.=-" else "_" for c in method)
    logs_dir = Path("logs/slurm") / safe_method
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def submit_sbatch(template_path: str, export_vars: dict) -> str:
    """Submit a SLURM job using sbatch and return its job id.

    SLURM's ``--export`` value is comma-delimited, so literal commas in exported
    values silently truncate variables. Callers should encode commas out, for
    example by using ``|`` and decoding inside the bash template.
    """
    bad = {k: v for k, v in export_vars.items() if "," in str(v)}
    if bad:
        raise ValueError(
            "submit_sbatch: SLURM --export cannot pass values containing commas "
            "(they delimit variables in the export string). Re-encode these "
            f"values without commas and decode in the bash template: {list(bad)}"
        )

    create_logs_dir()
    export_str = ",".join(f"{key}={value}" for key, value in export_vars.items())
    result = subprocess.run(
        ["sbatch", f"--export={export_str}", template_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split()[-1]


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 60)


def print_job_summary(job_ids: list) -> None:
    """Print submitted-job monitoring commands."""
    if not job_ids:
        return

    print("\nMonitor jobs with:")
    print("  squeue -u $USER")
    print("\nCheck specific job:")
    print(f"  squeue -j {job_ids[0][-1]}")
    print("\nCancel all jobs:")
    all_ids = " ".join(str(item[-1]) for item in job_ids)
    print(f"  scancel {all_ids}")
