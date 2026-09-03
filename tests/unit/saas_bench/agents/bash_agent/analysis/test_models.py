"""Analysis 结构化数据契约的快速单元测试。"""

import pytest
from pydantic import ValidationError

from saas_bench.agents.bash_agent.analysis.models import (
    EvidenceCard,
    Role,
    RoleReport,
    RoleSelection,
    StateAssessment,
    StatePortrait,
)


def _card(evidence_id: str = "MAR-001") -> EvidenceCard:
    return EvidenceCard(
        id=evidence_id,
        metric="market.effective_leads.individual",
        meaning="个人有效线索",
        fact="当前值 10，前期值 8，方向为上升",
        window="最近7天与前7天",
        direction="up",
    )


def _selection(evidence_id: str = "MAR-001") -> RoleSelection:
    return RoleSelection.model_validate({
        "selected_evidence_ids": [evidence_id],
        "hypotheses": [{
            "cause": "获客曝光可能增加",
            "evidence_ids": [evidence_id],
            "confidence": 0.7,
            "validation": "观察下一周同一公开指标",
        }],
        "risks": [{
            "risk": "新增客户可能带来服务压力",
            "evidence_ids": [evidence_id],
            "early_indicator": "容量利用率",
            "horizon_weeks": 2,
            "severity": 3,
        }],
    })


def _assessment_payload() -> dict:
    dimensions = [
        ("cash_health", "healthy"),
        ("demand_momentum", "growing"),
        ("unit_economics", "marginal"),
        ("service_pressure", "pressured"),
        ("customer_health", "watch"),
    ]
    return {
        "diagnosis": "增长正在形成服务和利润压力",
        "dimensions": [{
            "dimension": dimension,
            "label": label,
            "confidence": 0.7,
            "evidence_ids": ["MAR-001"],
            "rationale": "当前经营证据支持这一判断",
        } for dimension, label in dimensions],
        "key_evidence_ids": ["MAR-001"],
        "hypotheses": [{
            "cause": "广告可能增加需求",
            "evidence_for": ["MAR-001"],
            "evidence_against": [],
            "competing_causes": [],
            "confidence": 0.6,
            "validation_test": "比较下一周不同获客来源",
        }],
        "latent_risks": [{
            "risk": "服务质量可能承压",
            "evidence_ids": ["MAR-001"],
            "early_indicator": "错误率",
            "horizon_weeks": 2,
            "severity": 4,
        }],
    }


def test_role_report_uses_program_owned_evidence_cards():
    report = RoleReport.from_selection(
        Role.MARKET,
        7,
        _selection(),
        [_card()],
    )

    assert report.role is Role.MARKET
    assert report.day == 7
    assert report.evidence == [_card()]


def test_role_selection_rejects_duplicate_core_evidence():
    payload = _selection().model_dump()
    payload["selected_evidence_ids"] = ["MAR-001", "MAR-001"]
    with pytest.raises(ValidationError, match="must be unique"):
        RoleSelection.model_validate(payload)


def test_role_report_rejects_unknown_selection_and_wrong_role_prefix():
    with pytest.raises(ValueError, match="unknown evidence ids"):
        RoleReport.from_selection(Role.MARKET, 7, _selection("MAR-999"), [_card()])

    with pytest.raises(ValidationError, match="must start with FIN-"):
        RoleReport.from_selection(Role.FINANCE, 7, _selection(), [_card()])

    selection = _selection()
    selection.hypotheses[0].evidence_ids = ["MAR-999"]
    with pytest.raises(ValueError, match="unknown evidence ids"):
        RoleReport.from_selection(Role.MARKET, 7, selection, [_card()])


def test_role_selection_requires_at_least_one_evidence():
    with pytest.raises(ValidationError, match="at least 1 item"):
        RoleSelection.model_validate({
            "selected_evidence_ids": [],
            "hypotheses": [],
            "risks": [],
        })


def test_state_portrait_adds_day_and_preserves_fixed_dimensions():
    assessment = StateAssessment.model_validate(_assessment_payload())
    portrait = StatePortrait.from_assessment(7, assessment)

    assert portrait.day == 7
    assert portrait.dimensions == assessment.dimensions
    assert "state_label" not in portrait.model_dump()
    assert "causal_chain" not in portrait.model_dump()


def test_state_assessment_requires_each_fixed_dimension_once():
    payload = _assessment_payload()
    payload["dimensions"][-1] = payload["dimensions"][0]

    with pytest.raises(ValidationError, match="each fixed dimension exactly once"):
        StateAssessment.model_validate(payload)


def test_operating_dimension_rejects_wrong_label_for_dimension():
    payload = _assessment_payload()
    payload["dimensions"][0]["label"] = "surging"

    with pytest.raises(ValidationError, match="invalid label for cash_health"):
        StateAssessment.model_validate(payload)


def test_analysis_models_do_not_coerce_string_numbers():
    payload = _selection().model_dump()
    payload["hypotheses"][0]["confidence"] = "0.7"

    with pytest.raises(ValidationError, match="valid number"):
        RoleSelection.model_validate(payload)
