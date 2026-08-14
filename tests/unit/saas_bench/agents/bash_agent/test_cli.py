"""Bash Agent 对应逻辑的快速单元测试。"""

from pathlib import Path

import pytest

from saas_bench.agents.bash_agent import run_test


PROJECT_ROOT = Path(__file__).resolve().parents[5]

def test_main_starts_new_run_from_the_default_toml(monkeypatch):
    calls = []
    assert run_test.DEFAULT_EXPERIMENT_CONFIG == PROJECT_ROOT / "experiments/experiment.toml"
    assert run_test.DEFAULT_EXPERIMENT_CONFIG.is_file()

    class Runner:
        def run(self, verbose):
            assert verbose is True
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        run_test,
        "_new_experiment_runner",
        lambda path: calls.append(path) or Runner(),
    )
    monkeypatch.setattr(
        run_test,
        "_resume_runner",
        lambda value: pytest.fail("resume path should not be used"),
    )

    run_test.main([])

    assert calls == [run_test.DEFAULT_EXPERIMENT_CONFIG]

def test_main_starts_new_run_from_explicit_config(monkeypatch):
    calls = []
    config_path = PROJECT_ROOT / "experiments/smoke-deepseek.toml"

    class Runner:
        def run(self, verbose):
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        run_test,
        "_new_experiment_runner",
        lambda path: calls.append(path) or Runner(),
    )
    monkeypatch.setattr(
        run_test,
        "_resume_runner",
        lambda value: pytest.fail("resume path should not be used"),
    )

    run_test.main(["--config", str(config_path)])

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
        run_test,
        "_new_experiment_runner",
        lambda path: pytest.fail("current TOML should not be read during resume"),
    )
    monkeypatch.setattr(
        run_test,
        "_resume_runner",
        lambda value: calls.append(value) or Runner(),
    )

    run_test.main(["--resume", "existing"])

    assert calls == ["existing"]

@pytest.mark.parametrize(
    "args",
    [
        ["--model", "other-model"],
        ["--days", "7"],
        ["--temperature", "0.1"],
        ["--resume", "existing", "--config", "experiments/full.toml"],
    ],
)
def test_main_rejects_cli_configuration_overrides(args):
    with pytest.raises(SystemExit):
        run_test.main(args)
