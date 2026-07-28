from __future__ import annotations

import io
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit


def test_build_zip_excludes_local_artifacts_and_credentials(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "method.ipynb").write_text("{}")
    (submission / "helper.py").write_text("VALUE = 1\n")
    (submission / "test_helper.py").write_text("raise AssertionError\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_method.py").write_text("raise AssertionError\n")

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


def test_run_dry_stages_the_filtered_upload_instead_of_repository(
    tmp_path,
    monkeypatch,
):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "method.ipynb").write_text("{}")
    (submission / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "dry.yaml").write_text(
        "datasets:\n"
        "  - name: example/data\n"
        "    labels_uri: example/labels\n"
    )
    runner_src = tmp_path / "leaderboard" / "src"
    runner_src.mkdir(parents=True)

    for directory in ("results", "logs", ".venv", ".git"):
        path = tmp_path / directory
        path.mkdir()
        (path / "large-local-only.bin").write_bytes(b"x" * 1024)
    (tmp_path / ".env").write_text("SECRET=do-not-stage\n")

    captured = {}

    class StopAfterInspection(RuntimeError):
        pass

    def fake_dry_run(root, *args, **kwargs):
        staged = Path(root)
        captured["root"] = staged
        captured["files"] = {
            path.relative_to(staged).as_posix()
            for path in staged.rglob("*")
            if path.is_file()
        }
        raise StopAfterInspection

    package = types.ModuleType("aletheia_runner")
    package.__path__ = []
    dryrun = types.ModuleType("aletheia_runner.dryrun")
    dryrun.dry_run = fake_dry_run
    monkeypatch.setitem(sys.modules, "aletheia_runner", package)
    monkeypatch.setitem(sys.modules, "aletheia_runner.dryrun", dryrun)

    with pytest.raises(StopAfterInspection):
        submit.run_dry(tmp_path, "ndif-key", "hf-token", limit=1)

    assert captured["root"] != tmp_path
    assert captured["files"] == {
        "dry.yaml",
        "submission/helper.py",
        "submission/method.ipynb",
    }
    assert not captured["root"].exists()
