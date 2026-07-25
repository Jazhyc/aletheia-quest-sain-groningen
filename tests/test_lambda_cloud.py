from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/lambda_cloud.py"
SPEC = importlib.util.spec_from_file_location("lambda_cloud", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lambda_cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lambda_cloud)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_client_sends_bearer_token_without_putting_it_in_url() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse({"data": []})

    client = lambda_cloud.LambdaCloudClient(
        "top-secret", base_url="https://example.invalid/api/v1"
    )
    with patch.object(lambda_cloud.urllib.request, "urlopen", fake_urlopen):
        assert client.instances() == []

    assert captured == {
        "url": "https://example.invalid/api/v1/instances",
        "authorization": "Bearer top-secret",
        "timeout": 30.0,
    }
    assert "top-secret" not in captured["url"]


def test_launch_payload_uses_one_named_campaign_instance() -> None:
    response = FakeResponse({"data": {"instance_ids": ["instance-1"]}})
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return response

    client = lambda_cloud.LambdaCloudClient("key")
    with patch.object(lambda_cloud.urllib.request, "urlopen", fake_urlopen):
        ids = client.launch(
            {
                "region_name": "us-west-2",
                "instance_type_name": "gpu_1x_a100_sxm4",
                "ssh_key_names": ["aletheia-habrok"],
                "file_system_names": [],
                "quantity": 1,
                "name": "aletheia-prompt-campaign",
            }
        )

    assert ids == ["instance-1"]
    request = requests[0]
    assert request.method == "POST"
    assert json.loads(request.data) == {
        "region_name": "us-west-2",
        "instance_type_name": "gpu_1x_a100_sxm4",
        "ssh_key_names": ["aletheia-habrok"],
        "file_system_names": [],
        "quantity": 1,
        "name": "aletheia-prompt-campaign",
    }


def test_campaign_matching_is_exact_and_validated() -> None:
    instances = [
        {"name": "aletheia-prompt-campaign", "id": "wanted"},
        {"name": "aletheia-prompt-campaign-old", "id": "other"},
    ]
    assert lambda_cloud.matching_instances(instances, "prompt-campaign") == [
        {"name": "aletheia-prompt-campaign", "id": "wanted"}
    ]
    with pytest.raises(lambda_cloud.LambdaCloudError, match="campaign must start"):
        lambda_cloud.campaign_instance_name("../bad")


def test_parser_requires_explicit_confirmation_for_launch() -> None:
    parser = lambda_cloud.build_parser()
    args = parser.parse_args(
        [
            "launch",
            "--campaign",
            "prompt-campaign",
            "--instance-type",
            "gpu_1x_a100_sxm4",
            "--region",
            "us-west-2",
        ]
    )
    assert args.yes is False
    assert args.allow_non_x86 is False


def test_sync_code_requires_explicit_uncommitted_opt_in() -> None:
    parser = lambda_cloud.build_parser()
    args = parser.parse_args(["sync-code", "--campaign", "prompt-campaign"])
    assert args.include_uncommitted is False
    committed = parser.parse_args(["sync-commit", "--campaign", "prompt-campaign"])
    assert committed.revision == "HEAD"


def test_public_key_fingerprint_matches_openssh_shape() -> None:
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEqHxWB8sExampleOnly"
    assert lambda_cloud.public_key_fingerprint(public_key).startswith("SHA256:")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside", "results/../../outside"],
)
def test_remote_path_rejects_absolute_or_parent_traversal(path: str) -> None:
    with pytest.raises(lambda_cloud.LambdaCloudError):
        lambda_cloud.ensure_relative_remote_path(path)


def test_safe_api_error_does_not_include_authorization_header() -> None:
    error = urllib.error.HTTPError(
        "https://example.invalid/api/v1/instances",
        401,
        "Unauthorized",
        {"Authorization": "Bearer top-secret"},
        None,
    )
    assert lambda_cloud.safe_api_error(error) == "Unauthorized"
