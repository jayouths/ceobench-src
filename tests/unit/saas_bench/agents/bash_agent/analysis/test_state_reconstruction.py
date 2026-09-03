"""状态重构 Prompt、证据校验与修复测试。"""

import json

import pytest

from saas_bench.agents.bash_agent.analysis.brief import render_strategy_brief
from saas_bench.agents.bash_agent.analysis.models import (
    Role,
    AnalysisCallKind,
    RoleCallUsage,
    RoleReport,
    RoleReportsArtifact,
    StateCallUsage,
)
from saas_bench.agents.bash_agent.analysis.state_prompts import build_state_prompts
from saas_bench.agents.bash_agent.analysis.state_reconstruction import (
    StateCallOutcome,
    StateReconstructionError,
    StateReconstructor,
)
from tests.support.harness import make_analysis_pipeline


_PREFIX = {
    Role.MARKET: "MAR",
    Role.FINANCE: "FIN",
    Role.PRODUCT: "PRO",
    Role.CUSTOMER: "CUS",
}


def _role_reports() -> RoleReportsArtifact:
    reports = []
    calls = []
    for role in Role:
        evidence_id = f"{_PREFIX[role]}-1"
        reports.append(RoleReport.model_validate({
            "role": role,
            "day": 7,
            "evidence": [{
                "id": evidence_id,
                "observation": f"{role.value} 本周存在一项公开经营事实",
                "metric": f"{role.value}.metric",
                "direction": "flat",
                "strength": 0.8,
                "lag_note": "无明显滞后",
            }],
            "hypotheses": [],
            "risks": [],
        }))
        calls.append(RoleCallUsage(
            role=role,
            attempt=1,
            call_kind=AnalysisCallKind.INITIAL,
            requested_model="analysis-model",
            served_model="served-model",
            pricing_model="official-model",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            reasoning_tokens=0,
            elapsed_seconds=0.1,
            cost_amount=0.001,
            currency="USD",
        ))
    return RoleReportsArtifact(day=7, reports=reports, calls=calls)


def _valid_assessment() -> str:
    dimensions = [
        ("cash_health", "watch", "FIN-1"),
        ("demand_momentum", "stable", "MAR-1"),
        ("unit_economics", "marginal", "FIN-1"),
        ("service_pressure", "balanced", "PRO-1"),
        ("customer_health", "watch", "CUS-1"),
    ]
    return json.dumps({
        "diagnosis": "当前经营整体稳定，但现金与客户健康需要观察",
        "dimensions": [{
            "dimension": dimension,
            "label": label,
            "confidence": 0.7,
            "evidence_ids": [evidence_id],
            "rationale": "对应角色报告提供了直接证据",
        } for dimension, label, evidence_id in dimensions],
        "facts": [{
            "statement": "经营状态同时受到财务和客户信号支持",
            "evidence_ids": ["FIN-1", "CUS-1"],
            "confidence": 0.8,
        }],
        "hypotheses": [{
            "cause": "需求和交付状态共同影响客户健康",
            "evidence_for": ["MAR-1", "PRO-1"],
            "evidence_against": [],
            "competing_causes": ["客户结构发生变化"],
            "confidence": 0.6,
            "validation_test": "观察下一周需求、服务和客户信号",
        }],
        "latent_risks": [{
            "risk": "客户健康可能进一步恶化",
            "evidence_ids": ["CUS-1"],
            "early_indicator": "客户角色的公开周度信号",
            "horizon_weeks": 2,
            "severity": 3,
        }],
        "causal_chain": [{
            "cause": "服务状态变化",
            "effect": "客户健康承压",
            "evidence_ids": ["PRO-1", "CUS-1"],
            "confidence": 0.6,
        }],
    }, ensure_ascii=False)


def _usage(attempt: int, call_kind: AnalysisCallKind) -> StateCallUsage:
    return StateCallUsage(
        attempt=attempt,
        call_kind=call_kind,
        requested_model="analysis-model",
        served_model="served-model",
        pricing_model="official-model",
        input_tokens=100,
        output_tokens=30,
        cached_tokens=10,
        reasoning_tokens=5,
        elapsed_seconds=0.2,
        cost_amount=0.002,
        currency="USD",
    )


def test_state_prompt_only_contains_business_reports():
    reports = _role_reports()
    system_prompt, user_prompt = build_state_prompts(reports)
    payload = json.loads(user_prompt.split("：\n", 1)[1])

    assert len(payload["role_reports"]) == 4
    assert "calls" not in payload
    assert "state_label" not in system_prompt
    assert "不得补充外部事实、隐藏状态或行动建议" in system_prompt


def test_reconstructor_repairs_unknown_evidence_with_self_contained_context():
    reports = _role_reports()
    observed = []

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        observed.append(user_prompt)
        text = _valid_assessment()
        if attempt == 1:
            payload = json.loads(text)
            payload["facts"][0]["evidence_ids"] = ["MAR-9"]
            text = json.dumps(payload, ensure_ascii=False)
        return StateCallOutcome(text=text, usage=_usage(attempt, call_kind))

    artifact = StateReconstructor(
        call_model,
        max_schema_retries=1,
    ).generate(reports)

    assert artifact.day == 7
    assert len(artifact.calls) == 2
    assert "MAR-9" in observed[1]
    assert "unknown evidence ids" in observed[1]
    assert '"role_reports"' in observed[1]


def test_reconstructor_accepts_one_complete_json_code_fence():
    calls = []

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        calls.append(attempt)
        return StateCallOutcome(
            text=f"```json\n{_valid_assessment()}\n```",
            usage=_usage(attempt, call_kind),
        )

    artifact = StateReconstructor(
        call_model,
        max_schema_retries=1,
    ).generate(_role_reports())

    assert artifact.day == 7
    assert calls == [1]


def test_reconstructor_fails_after_configured_repair_limit():
    calls = []

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        calls.append((attempt, call_kind))
        return StateCallOutcome(
            text="[]",
            usage=_usage(attempt, call_kind),
        )

    with pytest.raises(StateReconstructionError, match="2 call"):
        StateReconstructor(call_model, max_schema_retries=1).generate(
            _role_reports()
        )

    assert calls == [
        (1, AnalysisCallKind.INITIAL),
        (2, AnalysisCallKind.REPAIR),
    ]


def test_pipeline_writes_reuses_and_summarizes_state_portrait(tmp_path):
    reports = _role_reports()
    reports_path = tmp_path / "analysis/day_007/role_reports.json"
    reports_path.parent.mkdir(parents=True)
    reports_path.write_text(reports.model_dump_json())

    pipeline = make_analysis_pipeline(tmp_path)
    calls = []

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        calls.append(attempt)
        return StateCallOutcome(
            text=_valid_assessment(),
            usage=_usage(attempt, call_kind),
        )

    pipeline.call_state_model = call_model
    artifact, generated = pipeline.ensure_state_portrait(reports)
    path = tmp_path / "analysis/day_007/state_portrait.json"

    assert generated is True
    assert path.is_file()
    assert calls == [1]

    pipeline.call_state_model = lambda *args: (_ for _ in ()).throw(
        AssertionError("completed state portrait must be reused")
    )
    reused, generated = pipeline.ensure_state_portrait(reports)
    usage = pipeline.usage_summary(7)

    assert generated is False
    assert reused == artifact
    assert usage["role_report_days"] == [7]
    assert usage["state_portrait_days"] == [7]
    assert usage["call_count"] == 5
    assert usage["state_reconstruction"]["call_count"] == 1
    assert usage["state_reconstruction"]["reasoning_tokens"] == 5
    assert usage["cost_by_currency"]["USD"] == pytest.approx(0.006)


def test_strategy_brief_is_deterministic_and_includes_evidence_index():
    reports = _role_reports()

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        return StateCallOutcome(
            text=_valid_assessment(),
            usage=_usage(attempt, call_kind),
        )

    portrait = StateReconstructor(
        call_model,
        max_schema_retries=0,
    ).generate(reports)

    first = render_strategy_brief(reports, portrait)
    second = render_strategy_brief(reports, portrait)

    assert first == second
    assert "# 本周经营状态简报" in first
    assert "## 五维经营状态" in first
    assert "## 已确认事实" in first
    assert "## 待验证假设" in first
    assert "## 潜在风险" in first
    assert "## 因果链" in first
    assert "## 证据索引" in first
    assert "**FIN-1** [finance.metric; flat; 强度 0.80]" in first
    assert "不包含行动指令" in first


def test_strategy_brief_omits_missing_evidence_direction():
    reports = _role_reports()
    reports.reports[0].evidence[0].direction = None

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        return StateCallOutcome(
            text=_valid_assessment(),
            usage=_usage(attempt, call_kind),
        )

    portrait = StateReconstructor(
        call_model,
        max_schema_retries=0,
    ).generate(reports)

    brief = render_strategy_brief(reports, portrait)

    assert "**MAR-1** [market.metric; 强度 0.80]" in brief


def test_strategy_brief_rejects_mismatched_days():
    reports = _role_reports()

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        return StateCallOutcome(
            text=_valid_assessment(),
            usage=_usage(attempt, call_kind),
        )

    portrait = StateReconstructor(
        call_model,
        max_schema_retries=0,
    ).generate(reports).model_copy(update={"day": 14})

    with pytest.raises(ValueError, match="same day"):
        render_strategy_brief(reports, portrait)


def test_pipeline_persists_reuses_and_injects_brief_only_when_enabled(tmp_path):
    reports = _role_reports()

    def call_model(day, attempt, call_kind, system_prompt, user_prompt):
        return StateCallOutcome(
            text=_valid_assessment(),
            usage=_usage(attempt, call_kind),
        )

    portrait = StateReconstructor(
        call_model,
        max_schema_retries=0,
    ).generate(reports)
    pipeline = make_analysis_pipeline(tmp_path)

    brief, generated = pipeline.ensure_brief(reports, portrait)
    path = tmp_path / "analysis/day_007/STRATEGY_BRIEF.md"

    assert generated is True
    assert path.read_text() == brief
    reused, generated = pipeline.ensure_brief(reports, portrait)
    assert generated is False
    assert reused == brief
    assert pipeline.decision_observation("dashboard", brief) == (
        f"dashboard\n\n---\n\n{brief}"
    )

    pipeline.enabled = False
    assert pipeline.decision_observation("dashboard", None) == "dashboard"
    assert pipeline.ensure_brief(reports, portrait) == (None, False)
