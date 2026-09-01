"""Bash Agent 对应逻辑的快速单元测试。"""

import pytest

from saas_bench.agents.bash_agent import cli
from tests.support.harness import TEST_CONFIG


def test_main_requires_explicit_config_or_resume():
    with pytest.raises(SystemExit):
        cli.main([])

def test_main_starts_new_run_from_explicit_config(monkeypatch):
    calls = []
    config_path = TEST_CONFIG

    class Runner:
        def run(self, verbose):
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        cli,
        "create_new_runner",
        lambda path: calls.append(path) or Runner(),
    )
    monkeypatch.setattr(
        cli,
        "create_resumed_runner",
        lambda value: pytest.fail("resume path should not be used"),
    )

    cli.main(["--config", str(config_path)])

    assert calls == [config_path]

def test_main_resume_uses_only_saved_run_identity(monkeypatch):
    calls = []

    class Runner:
        def run(self, verbose):
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        cli,
        "create_new_runner",
        lambda path: pytest.fail("current TOML should not be read during resume"),
    )
    monkeypatch.setattr(
        cli,
        "create_resumed_runner",
        lambda value: calls.append(value) or Runner(),
    )

    cli.main(["--resume", "existing"])

    assert calls == ["existing"]

@pytest.mark.parametrize(
    "args",
    [
        ["--model", "other-model"],
        ["--days", "7"],
        ["--temperature", "0.1"],
        ["--resume", "existing", "--config", "config/experiment.toml"],
    ],
)
def test_main_rejects_cli_configuration_overrides(args):
    with pytest.raises(SystemExit):
        cli.main(args)
