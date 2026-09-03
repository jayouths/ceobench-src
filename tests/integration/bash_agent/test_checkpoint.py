"""按职责拆分的 Harness 回归测试。"""

import json

import pytest

from saas_bench.agents.bash_agent.analysis.models import (
    Role,
    AnalysisCallKind,
    RoleCallUsage,
    EvidenceCard,
    RoleReport,
    RoleReportsArtifact,
)
from saas_bench.agents.bash_agent.runner import BashAgentRunner


from tests.support.harness import (
    EMPTY_ANALYSIS_USAGE,
    EMPTY_ENVIRONMENT_LLM_USAGE,
    make_checkpoint_runner as _checkpoint_runner,
)



def test_checkpoint_json_references_the_exact_database(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }

    runner._save_checkpoint(7)

    checkpoint = json.loads((runner.workspace_dir / "checkpoint.json").read_text())
    checkpoint_db = runner.workspace_dir / checkpoint["database"]["file"]
    assert checkpoint["day"] == 7
    assert checkpoint["cash"] == pytest.approx(900_000.0)
    assert checkpoint_db.read_bytes() == b"persisted-database"
    assert checkpoint["database"] == {
        "file": str(checkpoint_db.relative_to(runner.workspace_dir))
    }
    assert (runner.workspace_dir / "world.nmdb").read_bytes() == b"persisted-database"
    runtime = checkpoint["runtime"]
    conversation = runner.workspace_dir / runtime["conversation"]["file"]
    assert conversation.is_file()
    assert set(runtime["conversation"]) == {"file"}
    assert (
        runner.workspace_repository.git("rev-parse", "HEAD", check=True).stdout.strip()
        == runtime["workspace_commit"]
    )
    assert runtime["runner_log_offsets"] == {
        "trajectory": 0,
        "performance": 0,
    }
    assert runtime["server_log_offsets"] == {"history": 0, "event_log": 0}
    assert runtime["environment_llm"] == EMPTY_ENVIRONMENT_LLM_USAGE
    assert runtime["analysis"] == EMPTY_ANALYSIS_USAGE


def test_checkpoint_can_be_loaded_before_runner_session_is_initialized(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    runner._session_id = None

    checkpoint = runner._load_checkpoint()

    assert checkpoint["session_id"] == "session-1"


def test_checkpoint_persists_role_report_usage_from_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    prefixes = {
        Role.MARKET: "MAR",
        Role.FINANCE: "FIN",
        Role.PRODUCT: "PRO",
        Role.CUSTOMER: "CUS",
    }
    reports = [
        RoleReport(
            role=role,
            day=7,
            key_evidence_ids=[f"{prefixes[role]}-001"],
            evidence=[EvidenceCard(
                id=f"{prefixes[role]}-001",
                metric=f"{role.value}.test_metric",
                meaning="test metric",
                fact="current value 1",
                window="current point",
            )],
            hypotheses=[],
            risks=[],
        )
        for role in Role
    ]
    calls = [
        RoleCallUsage(
            role=role,
            attempt=1,
            call_kind=AnalysisCallKind.INITIAL,
            requested_model="channel-model",
            served_model="served-model",
            pricing_model="official-model",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=2,
            reasoning_tokens=1,
            elapsed_seconds=0.1,
            cost_amount=0.01,
            currency="USD",
        )
        for role in Role
    ]
    artifact = RoleReportsArtifact(day=7, reports=reports, calls=calls)
    artifact_path = runner.workspace_dir / "analysis/day_007/role_reports.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(artifact.model_dump_json())
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }

    checkpoint = runner._save_checkpoint(7)
    analysis = checkpoint["runtime"]["analysis"]

    assert analysis["role_report_days"] == [7]
    assert analysis["state_portrait_days"] == []
    assert analysis["call_count"] == 4
    assert analysis["input_tokens"] == 40
    assert analysis["reasoning_tokens"] == 4
    assert analysis["cost_by_currency"] == {"USD": pytest.approx(0.04)}
    assert analysis["by_role"]["finance"]["output_tokens"] == 5

def test_failed_database_persistence_keeps_previous_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    checkpoint_file = runner.workspace_dir / "checkpoint.json"
    checkpoint_file.write_text('{"day": 0}')
    runner._http_post = lambda path, data, timeout: {
        "success": False,
        "error": "week_advance_not_stable",
    }

    with pytest.raises(RuntimeError, match="persistence failed"):
        runner._save_checkpoint(7)

    assert checkpoint_file.read_text() == '{"day": 0}'
    assert not (runner.workspace_dir / ".checkpoint_dbs").exists()

def test_runtime_snapshot_failure_keeps_previous_checkpoint_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    old_db = runner.workspace_dir / ".checkpoint_dbs" / "old.nmdb"
    old_runtime = runner.workspace_dir / ".checkpoint_runtime" / "conversation_old.json"
    old_db.parent.mkdir()
    old_runtime.parent.mkdir()
    old_db.write_bytes(b"old-database")
    old_runtime.write_text("old-conversation")
    checkpoint_file = runner.workspace_dir / "checkpoint.json"
    checkpoint_file.write_text(json.dumps({
        "day": 0,
        "database_file": ".checkpoint_dbs/old.nmdb",
        "runtime": {"conversation_file": ".checkpoint_runtime/conversation_old.json"},
    }))
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner.workspace_repository.capture_checkpoint_commit = lambda day: (_ for _ in ()).throw(
        RuntimeError("git failed")
    )

    with pytest.raises(RuntimeError, match="git failed"):
        runner._save_checkpoint(7)

    assert json.loads(checkpoint_file.read_text())["day"] == 0
    assert old_db.read_bytes() == b"old-database"
    assert old_runtime.read_text() == "old-conversation"
