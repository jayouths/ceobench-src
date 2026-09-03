"""Analysis 结构化数据契约的快速单元测试。"""

import pytest
from pydantic import ValidationError

from saas_bench.agents.bash_agent.analysis.models import (
    Role,
    RoleAnalysis,
    RoleReport,
    StateAssessment,
    StatePortrait,
)


def _evidence(evidence_id: str = "MAR-1") -> dict:
    return {
        "id": evidence_id,
        "observation": "Recent acquisition increased",
        "metric": "market.new_customers_wow",
        "direction": "up",
        "strength": 0.8,
        "lag_note": "Advertising effects can lag by one week",
    }


def _role_analysis() -> RoleAnalysis:
    return RoleAnalysis.model_validate({
        "evidence": [_evidence()],
        "hypotheses": [{
            "cause": "Higher advertising exposure",
            "evidence_ids": ["MAR-1"],
            "confidence": 0.7,
            "validation": "Compare acquisition by channel next week",
        }],
        "risks": [{
            "risk": "Acquisition may outpace service capacity",
            "early_indicator": "Capacity utilization",
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
        "diagnosis": "Growth is creating service and margin pressure",
        "dimensions": [{
            "dimension": dimension,
            "label": label,
            "confidence": 0.7,
            "evidence_ids": ["MAR-1"],
            "rationale": "Supported by the current operating signals",
        } for dimension, label in dimensions],
        "facts": [{
            "statement": "Acquisition increased",
            "evidence_ids": ["MAR-1"],
            "confidence": 0.8,
        }],
        "hypotheses": [{
            "cause": "Advertising increased demand",
            "evidence_for": ["MAR-1"],
            "evidence_against": [],
            "competing_causes": ["Organic word of mouth"],
            "confidence": 0.6,
            "validation_test": "Compare acquisition channels next week",
        }],
        "latent_risks": [{
            "risk": "Service degradation may increase churn",
            "evidence_ids": ["MAR-1"],
            "early_indicator": "Error rate",
            "horizon_weeks": 2,
            "severity": 4,
        }],
        "causal_chain": [{
            "cause": "Higher acquisition",
            "effect": "Higher service load",
            "evidence_ids": ["MAR-1"],
            "confidence": 0.6,
        }],
    }


def test_role_report_adds_program_owned_identity():
    report = RoleReport.from_analysis(Role.MARKET, 7, _role_analysis())

    assert report.role is Role.MARKET
    assert report.day == 7
    assert report.evidence[0].id == "MAR-1"


def test_evidence_without_direction_omits_field_when_serialized():
    payload = _evidence()
    payload.pop("direction")

    evidence = RoleAnalysis.model_validate({
        "evidence": [payload],
        "hypotheses": [],
        "risks": [],
    }).evidence[0]

    assert evidence.direction is None
    assert "direction" not in evidence.model_dump(mode="json")


def test_role_report_rejects_invalid_evidence_references_and_role_prefix():
    invalid_reference = _role_analysis().model_dump()
    invalid_reference["hypotheses"][0]["evidence_ids"] = ["MAR-2"]
    with pytest.raises(ValidationError, match="unknown evidence ids"):
        RoleAnalysis.model_validate(invalid_reference)

    with pytest.raises(ValidationError, match="must start with FIN-"):
        RoleReport.from_analysis(Role.FINANCE, 7, _role_analysis())


def test_role_analysis_rejects_empty_evidence_report():
    with pytest.raises(ValidationError, match="at least 1 item"):
        RoleAnalysis.model_validate({
            "evidence": [],
            "hypotheses": [],
            "risks": [],
        })


def test_state_portrait_adds_day_and_preserves_fixed_dimensions():
    assessment = StateAssessment.model_validate(_assessment_payload())
    portrait = StatePortrait.from_assessment(7, assessment)

    assert portrait.day == 7
    assert portrait.dimensions == assessment.dimensions
    assert "state_label" not in portrait.model_dump()


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
    payload = _role_analysis().model_dump()
    payload["evidence"][0]["strength"] = "0.8"

    with pytest.raises(ValidationError, match="valid number"):
        RoleAnalysis.model_validate(payload)
