"""四角色报告生成、修复和周产物测试。"""

import json

import pytest

from saas_bench.agents.bash_agent import run_test
from saas_bench.agents.bash_agent.analysis.models import (
    Role,
    RoleCallKind,
    RoleCallUsage,
    RoleReportsArtifact,
)
from saas_bench.agents.bash_agent.analysis.role_prompts import build_role_prompts
from saas_bench.agents.bash_agent.analysis.role_reports import (
    RoleCallOutcome,
    RoleReportGenerationError,
    RoleReportGenerator,
)
from saas_bench.agents.bash_agent.analysis.signals import SignalCollector
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.public_week_snapshot import build_public_week_snapshot
from saas_bench.llm_provider import TextLLMResult


_PREFIX = {
    Role.MARKET: "MAR",
    Role.FINANCE: "FIN",
    Role.PRODUCT: "PRO",
    Role.CUSTOMER: "CUS",
}

_METRIC = {
    Role.MARKET: ("market.effective_leads.individual", "insufficient_data"),
    Role.FINANCE: ("finance.current_cash", "flat"),
    Role.PRODUCT: ("product.configuration.current.tier_a", "flat"),
    Role.CUSTOMER: (
        "customer.customer_base.active_individual_accounts",
        "flat",
    ),
}


def _direct_query(conn):
    def query(sql):
        return [dict(row) for row in conn.execute(sql).fetchall()]

    return query


def _valid_response(role: Role) -> str:
    evidence_id = f"{_PREFIX[role]}-1"
    metric, direction = _METRIC[role]
    return json.dumps({
        "evidence": [{
            "id": evidence_id,
            "observation": "当前公开信号支持一项经营事实",
            "metric": metric,
            "direction": direction,
            "strength": 0.8,
            "lag_note": "无明显滞后",
        }],
        "hypotheses": [{
            "cause": "一种尚待验证的原因",
            "evidence_ids": [evidence_id],
            "confidence": 0.6,
            "validation": "观察下一周同一公开指标",
        }],
        "risks": [],
    }, ensure_ascii=False)


def _usage(role: Role, attempt: int, call_kind: RoleCallKind) -> RoleCallUsage:
    return RoleCallUsage(
        role=role,
        attempt=attempt,
        call_kind=call_kind,
        requested_model="analysis-model",
        served_model="served-analysis-model",
        pricing_model="official-analysis-model",
        input_tokens=100,
        output_tokens=20,
        cached_tokens=10,
        reasoning_tokens=5,
        elapsed_seconds=0.25,
        cost_amount=0.001,
        currency="USD",
    )


@pytest.fixture
def day_zero_signals(make_initialized_sim):
    conn, _, _ = make_initialized_sim(seed=42)
    return SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )


def test_role_prompts_only_include_the_selected_role(day_zero_signals):
    system_prompt, user_prompt = build_role_prompts(day_zero_signals, Role.MARKET)

    payload = json.loads(user_prompt.split("：\n", 1)[1])
    assert payload["signals"] == day_zero_signals.market.model_dump(mode="json")
    assert "finance" not in payload
    assert "不得输出行动建议" in system_prompt
    assert "metric 必须填写输入中的精确字段路径" in system_prompt


def test_generator_repairs_invalid_json_with_self_contained_context(day_zero_signals):
    observed = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        observed.append((day, role, attempt, call_kind, user_prompt))
        text = "not-json" if role is Role.MARKET and attempt == 1 else _valid_response(role)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=1,
    ).generate(day_zero_signals)

    assert [report.role for report in artifact.reports] == list(Role)
    assert len(artifact.calls) == 5
    market_repair = observed[1]
    assert market_repair[3] is RoleCallKind.REPAIR
    assert "not-json" in market_repair[4]
    assert "程序校验错误" in market_repair[4]
    assert '"signals"' in market_repair[4]


def test_generator_fails_after_configured_repair_limit(day_zero_signals):
    calls = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        calls.append((role, attempt, call_kind))
        return RoleCallOutcome(
            text="[]",
            usage=_usage(role, attempt, call_kind),
        )

    with pytest.raises(RoleReportGenerationError, match="market.*2 call"):
        RoleReportGenerator(call_model, max_schema_retries=1).generate(
            day_zero_signals
        )

    assert calls == [
        (Role.MARKET, 1, RoleCallKind.INITIAL),
        (Role.MARKET, 2, RoleCallKind.REPAIR),
    ]


def test_generator_repairs_unknown_metric_path(day_zero_signals):
    observed = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        observed.append(user_prompt)
        text = _valid_response(role)
        if role is Role.MARKET and attempt == 1:
            payload = json.loads(text)
            payload["evidence"][0]["metric"] = "market.invented.metric"
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=1,
    ).generate(day_zero_signals)

    assert len(artifact.calls) == 5
    assert "unknown metric path" in observed[1]


def test_generator_repairs_metric_direction_mismatch(day_zero_signals):
    observed = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        observed.append(user_prompt)
        text = _valid_response(role)
        if role is Role.MARKET and attempt == 1:
            payload = json.loads(text)
            payload["evidence"][0]["direction"] = "up"
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=1,
    ).generate(day_zero_signals)

    assert len(artifact.calls) == 5
    assert "metric direction mismatch" in observed[1]


def test_runner_writes_reuses_and_summarizes_role_reports(
    tmp_path,
    day_zero_signals,
):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    runner.analysis_enabled = True
    runner.analysis_module_config = {"max_schema_retries": 1}
    calls = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        calls.append(role)
        return RoleCallOutcome(
            text=_valid_response(role),
            usage=_usage(role, attempt, call_kind),
        )

    runner._call_analysis_role_model = call_model
    artifact, generated = runner._ensure_analysis_role_reports(day_zero_signals)
    path = tmp_path / "analysis" / "day_000" / "role_reports.json"

    assert generated is True
    assert path.is_file()
    assert RoleReportsArtifact.model_validate_json(path.read_text()) == artifact
    assert calls == list(Role)

    runner._call_analysis_role_model = lambda *args: (_ for _ in ()).throw(
        AssertionError("completed report must be reused")
    )
    reused, generated = runner._ensure_analysis_role_reports(day_zero_signals)
    usage = runner._analysis_usage_summary(0)

    assert generated is False
    assert reused == artifact
    assert usage["completed_days"] == [0]
    assert usage["call_count"] == 4
    assert usage["input_tokens"] == 400
    assert usage["by_role"]["market"]["reasoning_tokens"] == 5
    assert usage["cost_by_currency"]["USD"] == pytest.approx(0.004)


def test_runner_records_raw_response_timing_and_official_cost(tmp_path, monkeypatch):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.analysis_client = object()
    runner.analysis_model_config = {
        "api_type": "openai_chat_completions",
        "model": "channel-model",
        "max_output_tokens": 1000,
        "temperature": 0.2,
        "top_p": None,
        "reasoning_effort": "none",
        "request_options": {},
        "tasks": {},
        "pricing_model_map": {"served-model": "official-model"},
        "pricing": {
            "official-model": {
                "currency": "USD",
                "uncached_input_cost_per_million": 1.0,
                "cached_input_cost_per_million": 0.1,
                "output_cost_per_million": 2.0,
            }
        },
    }
    runner.run_id = "test"
    runner.response_log_file = tmp_path / "raw.jsonl"
    runner.timing_log_file = tmp_path / "timing.jsonl"
    runner._timing_queue = None
    runner._dashboard_url = ""

    monkeypatch.setattr(
        run_test,
        "call_text_model",
        lambda **kwargs: TextLLMResult(
            text=_valid_response(Role.MARKET),
            model="served-model",
            input_tokens=100,
            output_tokens=20,
            cached_tokens=10,
            reasoning_tokens=5,
            raw_response={"id": "raw-response"},
        ),
    )

    outcome = runner._call_analysis_role_model(
        7,
        Role.MARKET,
        1,
        RoleCallKind.INITIAL,
        "system",
        "user",
    )

    raw = json.loads(runner.response_log_file.read_text())
    timing = json.loads(runner.timing_log_file.read_text())
    assert outcome.usage.pricing_model == "official-model"
    assert outcome.usage.cost_amount == pytest.approx(0.000131)
    assert raw["component"] == "analysis"
    assert raw["raw_response"] == {"id": "raw-response"}
    assert timing["event"] == "analysis_llm_call"
    assert timing["reasoning_tokens"] == 5


def test_resume_removes_only_role_reports_not_confirmed_by_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    for day in (0, 7, 14):
        directory = tmp_path / "analysis" / f"day_{day:03d}"
        directory.mkdir(parents=True)
        (directory / "signals.json").write_text("signals")
        (directory / "role_reports.json").write_text("reports")

    runner._prune_analysis_artifacts_after(7, {0})

    assert (tmp_path / "analysis" / "day_000" / "role_reports.json").is_file()
    assert not (tmp_path / "analysis" / "day_007" / "role_reports.json").exists()
    assert (tmp_path / "analysis" / "day_007" / "signals.json").is_file()
    assert not (tmp_path / "analysis" / "day_014").exists()
