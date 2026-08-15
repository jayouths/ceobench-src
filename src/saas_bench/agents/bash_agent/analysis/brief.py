"""将结构化经营画像确定性格式化为决策 Agent 可读简报。"""

from __future__ import annotations

from .models import RoleReportsArtifact, StatePortraitArtifact


_DIMENSION_NAMES = {
    "cash_health": "现金健康度",
    "demand_momentum": "需求动量",
    "unit_economics": "单位经济性",
    "service_pressure": "服务压力",
    "customer_health": "客户健康度",
}


def _table_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _ids(values: list[str]) -> str:
    return ", ".join(values) if values else "无"


def render_strategy_brief(
    role_reports: RoleReportsArtifact,
    portrait: StatePortraitArtifact,
) -> str:
    """只使用已校验产物生成 Markdown，不再引入模型判断。"""

    if role_reports.day != portrait.day:
        raise ValueError("role reports and state portrait must have the same day")

    lines = [
        "# 本周经营状态简报",
        "",
        f"模拟日：{portrait.day}",
        "",
        "> 本简报用于补充经营状态识别。事实、假设和风险已分区，不包含行动指令。",
        "",
        "## 核心诊断",
        "",
        portrait.diagnosis,
        "",
        "## 五维经营状态",
        "",
        "| 维度 | 状态 | 置信度 | 证据 | 判断依据 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for dimension in portrait.dimensions:
        lines.append(
            "| "
            + " | ".join((
                _DIMENSION_NAMES[dimension.dimension.value],
                dimension.label,
                f"{dimension.confidence:.2f}",
                _ids(dimension.evidence_ids),
                _table_cell(dimension.rationale),
            ))
            + " |"
        )

    lines.extend(["", "## 已确认事实", ""])
    if portrait.facts:
        for fact in portrait.facts:
            lines.append(
                f"- {fact.statement}（证据：{_ids(fact.evidence_ids)}；"
                f"置信度：{fact.confidence:.2f}）"
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## 待验证假设", ""])
    if portrait.hypotheses:
        for index, hypothesis in enumerate(portrait.hypotheses, start=1):
            lines.extend([
                f"{index}. {hypothesis.cause}（置信度：{hypothesis.confidence:.2f}）",
                f"   - 支持证据：{_ids(hypothesis.evidence_for)}",
                f"   - 反对证据：{_ids(hypothesis.evidence_against)}",
                f"   - 竞争性解释：{'；'.join(hypothesis.competing_causes)}",
                f"   - 验证方式：{hypothesis.validation_test}",
            ])
    else:
        lines.append("- 无")

    lines.extend(["", "## 潜在风险", ""])
    if portrait.latent_risks:
        for risk in portrait.latent_risks:
            lines.append(
                f"- {risk.risk}（证据：{_ids(risk.evidence_ids)}；"
                f"先行指标：{risk.early_indicator}；预计 {risk.horizon_weeks} 周；"
                f"严重度：{risk.severity}/5）"
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## 因果链", ""])
    if portrait.causal_chain:
        for step in portrait.causal_chain:
            lines.append(
                f"- {step.cause} -> {step.effect}（证据：{_ids(step.evidence_ids)}；"
                f"置信度：{step.confidence:.2f}）"
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## 证据索引", ""])
    for report in role_reports.reports:
        for evidence in report.evidence:
            lines.append(
                f"- **{evidence.id}** [{evidence.metric}; {evidence.direction.value}; "
                f"强度 {evidence.strength:.2f}] {evidence.observation} "
                f"（滞后：{evidence.lag_note}）"
            )
    if not any(report.evidence for report in role_reports.reports):
        lines.append("- 无")

    return "\n".join(lines) + "\n"
