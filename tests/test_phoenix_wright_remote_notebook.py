import os
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "submission" / "phoenix_wright_v1_3.ipynb"
SUBMISSION_CSV = REPO_ROOT / "submission.csv"
DEV_METIS = (
    "aletheias-quest/"
    "dev-instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None"
)
NDIF_HOST = "https://aletheias.api.ndif.us"


def _load_dotenv_key() -> None:
    if os.environ.get("NDIF_API_KEY"):
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "NDIF_API_KEY" and value:
            os.environ[key] = value.strip().strip("'\"")
            return


def test_remote_notebook_scores_survive_nnsight_session(monkeypatch):
    _load_dotenv_key()
    assert os.environ.get("NDIF_API_KEY"), "NDIF_API_KEY must be set or present in .env"

    monkeypatch.setenv("NDIF_HOST", NDIF_HOST)
    monkeypatch.setenv("DATASET_NAME", DEV_METIS)
    monkeypatch.setenv("ALETHEIA_LIMIT", "17")
    monkeypatch.setenv("PHOENIX_BATCH_SIZE", "16")
    monkeypatch.delenv("PHOENIX_NNSIGHT_LOCAL", raising=False)

    if SUBMISSION_CSV.exists():
        SUBMISSION_CSV.unlink()

    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        NotebookClient(
            nb,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        ).execute()

        assert SUBMISSION_CSV.exists(), "notebook did not write submission.csv"
        df = pd.read_csv(SUBMISSION_CSV)
        assert list(df.columns) == ["index", "deceptive", "score"]
        assert len(df) == 17
        assert df["score"].between(0.0, 1.0).all()
        assert df["score"].tolist()[:2] == [0.0, 1.0]
        assert df["deceptive"].tolist()[:2] == [False, True]
    finally:
        if SUBMISSION_CSV.exists():
            SUBMISSION_CSV.unlink()
