"""四角色报告生成、修复和周产物测试。"""

import json
import threading

import pytest

from saas_bench.agents.bash_agent.analysis import pipeline as pipeline_module
from saas_bench.agents.bash_agent.analysis.models import (
    Direction,
    Role,
    AnalysisCallKind,
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
from saas_bench.agents.bash_agent.analysis.signal_models import (
    ChangeDirection,
    DataStatus,
    MetricComparison,
    NumericObservation,
    SocialPost,
)
from saas_bench.simulator.public_week_snapshot import build_public_week_snapshot
from saas_bench.experiment.llm_provider import TextLLMResult
from tests.support.harness import make_analysis_pipeline


_PREFIX = {
    Role.MARKET: "MAR",
    Role.FINANCE: "FIN",
    Role.PRODUCT: "PRO",
    Role.CUSTOMER: "CUS",
}

_METRIC = {
    Role.MARKET: "market.effective_leads.individual",
    Role.FINANCE: "finance.current_cash",
    Role.PRODUCT: "product.configuration.model_tier.A",
    Role.CUSTOMER: "customer.customer_base.active_individual_accounts",
}


def _direct_query(conn):
    def query(sql):
        return [dict(row) for row in conn.execute(sql).fetchall()]

    return query


def _valid_response(role: Role) -> str:
    evidence_id = f"{_PREFIX[role]}-1"
    return json.dumps({
        "evidence": [{
            "id": evidence_id,
            "observation": "当前公开信号支持一项经营事实",
            "metric": _METRIC[role],
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


def _usage(role: Role, attempt: int, call_kind: AnalysisCallKind) -> RoleCallUsage:
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
    assert payload["market"] == day_zero_signals.market.model_dump(mode="json")
    assert "signals" not in payload
    assert "finance" not in payload
    assert "不得输出行动建议" in system_prompt
    assert "metric 必须从输入 JSON 的 market 顶层键开始" in system_prompt
    assert "windows 只用于判断数据完整性" in system_prompt
    assert "必须描述同一个信号" in system_prompt
    assert "时点值、文本、列表元素及数据不足的比较必须省略 direction" in system_prompt


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
    assert market_repair[3] is AnalysisCallKind.REPAIR
    assert "not-json" in market_repair[4]
    assert "程序校验错误" in market_repair[4]
    assert '"market"' in market_repair[4]


def test_generator_runs_independent_roles_concurrently_in_stable_order(
    day_zero_signals,
):
    barrier = threading.Barrier(len(Role))

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        barrier.wait(timeout=2)
        return RoleCallOutcome(
            text=_valid_response(role),
            usage=_usage(role, attempt, call_kind),
        )

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=0,
        role_report_concurrency=4,
    ).generate(day_zero_signals)

    assert [report.role for report in artifact.reports] == list(Role)
    assert [call.role for call in artifact.calls] == list(Role)


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
        (Role.MARKET, 1, AnalysisCallKind.INITIAL),
        (Role.MARKET, 2, AnalysisCallKind.REPAIR),
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
    assert "不得只替换路径" in observed[1]


def test_generator_reports_all_invalid_metric_paths_in_one_repair(day_zero_signals):
    observed = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        observed.append(user_prompt)
        text = _valid_response(role)
        if role is Role.MARKET and attempt == 1:
            payload = json.loads(text)
            second = dict(payload["evidence"][0])
            payload["evidence"][0]["metric"] = "signals.first.invalid"
            second["id"] = "MAR-2"
            second["metric"] = "market.second.invalid"
            payload["evidence"].append(second)
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    RoleReportGenerator(call_model, max_schema_retries=1).generate(day_zero_signals)

    repair_prompt = observed[1]
    assert "signals.first.invalid" in repair_prompt
    assert "market.second.invalid" in repair_prompt


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


def test_generator_rejects_invented_direction_for_point_in_time_metric(
    day_zero_signals,
):
    observed = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        observed.append(user_prompt)
        text = _valid_response(role)
        if role is Role.FINANCE and attempt == 1:
            payload = json.loads(text)
            payload["evidence"][0]["direction"] = "up"
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=1,
    ).generate(day_zero_signals)

    assert len(artifact.calls) == 5
    assert "expected None, got 'up'" in observed[2]


def test_generator_accepts_direction_for_individual_configuration_metric(
    day_zero_signals,
):
    comparison = MetricComparison(
        current=NumericObservation(value=2, status=DataStatus.AVAILABLE),
        previous=NumericObservation(value=1, status=DataStatus.AVAILABLE),
        absolute_change=1,
        relative_change=1.0,
        direction=ChangeDirection.UP,
        comparison_status=DataStatus.AVAILABLE,
    )
    model_tier = day_zero_signals.product.configuration.model_tier.model_copy(
        update={"A": comparison}
    )
    configuration = day_zero_signals.product.configuration.model_copy(
        update={
            "model_tier": model_tier,
        }
    )
    signals = day_zero_signals.model_copy(
        update={
            "product": day_zero_signals.product.model_copy(
                update={"configuration": configuration}
            )
        }
    )

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        text = _valid_response(role)
        if role is Role.PRODUCT:
            payload = json.loads(text)
            payload["evidence"][0].update({
                "metric": "product.configuration.model_tier.A",
                "direction": "up",
            })
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=0,
    ).generate(signals)

    product = next(report for report in artifact.reports if report.role is Role.PRODUCT)
    assert product.evidence[0].direction is Direction.UP


def test_generator_accepts_array_metric_path_without_direction(day_zero_signals):
    post = SocialPost(post_id=1, day=0, content="公开客户反馈")
    social_feedback = day_zero_signals.market.social_feedback.model_copy(
        update={"current_posts": [post]}
    )
    market = day_zero_signals.market.model_copy(
        update={"social_feedback": social_feedback}
    )
    signals = day_zero_signals.model_copy(update={"market": market})

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        text = _valid_response(role)
        if role is Role.MARKET:
            payload = json.loads(text)
            payload["evidence"][0]["metric"] = (
                "market.social_feedback.current_posts[0].content"
            )
            text = json.dumps(payload, ensure_ascii=False)
        return RoleCallOutcome(text=text, usage=_usage(role, attempt, call_kind))

    artifact = RoleReportGenerator(
        call_model,
        max_schema_retries=0,
    ).generate(signals)

    market_report = next(
        report for report in artifact.reports if report.role is Role.MARKET
    )
    assert market_report.evidence[0].direction is None
    assert "direction" not in market_report.evidence[0].model_dump(mode="json")


@pytest.mark.parametrize(
    "path",
    [
        "social_feedback.current_posts[1].content",
        "social_feedback.current_posts[-1].content",
        "social_feedback.current_posts[abc].content",
    ],
)
def test_metric_path_rejects_invalid_array_index(path):
    payload = {
        "social_feedback": {
            "current_posts": [{"content": "公开客户反馈"}],
        }
    }

    found, target = RoleReportGenerator._resolve_metric_path(payload, path)

    assert found is False
    assert target is None


def test_pipeline_writes_reuses_and_summarizes_role_reports(
    tmp_path,
    day_zero_signals,
):
    pipeline = make_analysis_pipeline(tmp_path)
    calls = []

    def call_model(day, role, attempt, call_kind, system_prompt, user_prompt):
        calls.append(role)
        return RoleCallOutcome(
            text=_valid_response(role),
            usage=_usage(role, attempt, call_kind),
        )

    pipeline.call_role_model = call_model
    assert pipeline.ensure_role_reports(day_zero_signals) == (None, False)
    assert calls == []
    assert not (tmp_path / "analysis" / "day_000" / "role_reports.json").exists()

    day_seven_signals = day_zero_signals.model_copy(
        update={"day": 7, "week": 1}
    )
    artifact, generated = pipeline.ensure_role_reports(day_seven_signals)
    path = tmp_path / "analysis" / "day_007" / "role_reports.json"

    assert generated is True
    assert path.is_file()
    assert RoleReportsArtifact.model_validate_json(path.read_text()) == artifact
    assert calls == list(Role)

    pipeline.call_role_model = lambda *args: (_ for _ in ()).throw(
        AssertionError("completed report must be reused")
    )
    reused, generated = pipeline.ensure_role_reports(day_seven_signals)
    usage = pipeline.usage_summary(7)

    assert generated is False
    assert reused == artifact
    assert usage["role_report_days"] == [7]
    assert usage["state_portrait_days"] == []
    assert usage["call_count"] == 4
    assert usage["input_tokens"] == 400
    assert usage["by_role"]["market"]["reasoning_tokens"] == 5
    assert usage["state_reconstruction"]["call_count"] == 0
    assert usage["cost_by_currency"]["USD"] == pytest.approx(0.004)


def test_pipeline_records_analysis_call_in_trajectory_with_official_cost(
    tmp_path, monkeypatch
):
    model_config = {
        "provider": "openai",
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
    events = []
    pipeline = make_analysis_pipeline(
        tmp_path,
        model_config=model_config,
        client=object(),
        log_trajectory=lambda event_type, day, **fields: events.append({
            "event_type": event_type,
            "day": day,
            **fields,
        }),
    )

    monkeypatch.setattr(
        pipeline_module,
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

    outcome = pipeline.call_role_model(
        7,
        Role.MARKET,
        1,
        AnalysisCallKind.INITIAL,
        "system",
        "user",
    )

    event = events[0]
    assert outcome.usage.pricing_model == "official-model"
    assert outcome.usage.cost_amount == pytest.approx(0.000131)
    assert event["event_type"] == "llm_call"
    assert event["component"] == "analysis"
    assert event["raw_response"] == {"id": "raw-response"}
    assert event["reasoning_tokens"] == 5


def test_resume_prunes_llm_artifacts_at_their_independent_checkpoint_boundaries(tmp_path):
    pipeline = make_analysis_pipeline(tmp_path)
    for day in (0, 7, 14):
        directory = tmp_path / "analysis" / f"day_{day:03d}"
        directory.mkdir(parents=True)
        (directory / "signals.json").write_text("signals")
        (directory / "role_reports.json").write_text("reports")
        (directory / "state_portrait.json").write_text("portrait")
        (directory / "STRATEGY_BRIEF.md").write_text("brief")

    pipeline.prune_artifacts_after(7, {0, 7}, {0})

    assert (tmp_path / "analysis" / "day_000" / "role_reports.json").is_file()
    assert (tmp_path / "analysis" / "day_000" / "state_portrait.json").is_file()
    assert (tmp_path / "analysis" / "day_007" / "role_reports.json").is_file()
    assert not (tmp_path / "analysis" / "day_007" / "state_portrait.json").exists()
    assert not (tmp_path / "analysis" / "day_007" / "STRATEGY_BRIEF.md").exists()
    assert (tmp_path / "analysis" / "day_007" / "signals.json").is_file()
    assert not (tmp_path / "analysis" / "day_014").exists()
