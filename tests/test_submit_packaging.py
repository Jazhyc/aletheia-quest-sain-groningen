from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit


def test_build_zip_excludes_local_artifacts_and_credentials(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "method.ipynb").write_text("{}")
    (submission / "helper.py").write_text("VALUE = 1\n")

    for directory in ("results", "logs", "dev_splits", ".uv-cache"):
        path = tmp_path / directory
        path.mkdir()
        (path / "local-only.txt").write_text("do not package\n")

    (tmp_path / ".env").write_text("SECRET=do-not-package\n")
    (tmp_path / "submission.csv").write_text("generated output\n")

    payload = submit.build_zip(tmp_path)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())

    assert names == {"submission/helper.py", "submission/method.ipynb"}
