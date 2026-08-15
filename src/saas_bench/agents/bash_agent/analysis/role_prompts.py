"""四个职能角色的 Prompt 和输入裁剪。"""

from __future__ import annotations

import json
from typing import Any

from .models import Role, RoleAnalysis
from .signal_models import AnalysisSignals


_ROLE_NAMES = {
    Role.MARKET: "市场分析",
    Role.FINANCE: "财务分析",
    Role.PRODUCT: "产品与技术分析",
    Role.CUSTOMER: "客户成功分析",
}

_ROLE_PREFIXES = {
    Role.MARKET: "MAR",
    Role.FINANCE: "FIN",
    Role.PRODUCT: "PRO",
    Role.CUSTOMER: "CUS",
}

_ROLE_GUIDANCE = {
    Role.MARKET: (
        "识别需求动量、获客来源结构、付费获客效率、公开市场反馈和宏观环境。"
        "不要分析内部成本、产品配置或提出营销动作。"
    ),
    Role.FINANCE: (
        "识别现金健康度、收入质量、成本结构、交付利润率和现金跑道。"
        "区分经常性成本与一次性投资，不要把模型 API 成本当作企业现金流。"
    ),
    Role.PRODUCT: (
        "识别使用量、容量压力、可靠性、配置变化、研发管线和当前交付质量。"
        "配置变化与服务波动只能形成待验证假设，不能直接断言因果。"
    ),
    Role.CUSTOMER: (
        "识别客户基础、新增付费、流失、工单压力和企业谈判状态。"
        "账户数与企业席位数不可直接混合，公开信息不足时必须明确保留不确定性。"
    ),
}


def role_signal_payload(signals: AnalysisSignals, role: Role) -> dict[str, Any]:
    """每个角色只接收本职信号，避免跨角色信息造成职责漂移。"""

    return {
        "day": signals.day,
        "week": signals.week,
        "windows": signals.windows.model_dump(mode="json"),
        "signals": getattr(signals, role.value).model_dump(mode="json"),
    }


def build_role_prompts(signals: AnalysisSignals, role: Role) -> tuple[str, str]:
    prefix = _ROLE_PREFIXES[role]
    schema = RoleAnalysis.model_json_schema()
    example = {
        "evidence": [{
            "id": f"{prefix}-1",
            "observation": "用一句话陈述信号直接支持的经营事实",
            "metric": f"{role.value}.精确字段路径",
            "direction": "up",
            "strength": 0.8,
            "lag_note": "说明该指标的反馈滞后或写无明显滞后",
        }],
        "hypotheses": [{
            "cause": "对事实原因的待验证解释",
            "evidence_ids": [f"{prefix}-1"],
            "confidence": 0.6,
            "validation": "下一步应观察什么公开指标来验证",
        }],
        "risks": [{
            "risk": "尚未充分暴露的潜在风险",
            "early_indicator": "应持续观察的公开先行指标",
            "horizon_weeks": 2,
            "severity": 3,
        }],
    }
    system_prompt = f"""你是企业经营状态识别系统中的{_ROLE_NAMES[role]}角色。

你的任务是解释当前公开经营信号，不是制定策略。{_ROLE_GUIDANCE[role]}

必须遵守：
1. 只能使用用户提供的 JSON 信号，不得补充外部事实或隐藏状态。
2. evidence 是可由输入直接核验的事实，最多 5 条；id 必须使用 {prefix}-1 至 {prefix}-5。
3. metric 必须填写输入中的精确字段路径，不能使用含糊名称。
4. direction 只能是 up、down、flat、insufficient_data；没有完整可比窗口时使用 insufficient_data。
5. strength 表示证据对 observation 的支持强度，不表示业务重要程度。
6. hypotheses 最多 3 条，必须引用已输出的 evidence id，并写出可由后续公开信号执行的验证方式。
7. risks 最多 3 条，只写已有早期指标支持但尚未充分暴露的风险。
8. 不得输出行动建议，不得把相关性写成已确认因果；数据不足时允许相应数组为空。
9. 只返回一个符合 Schema 的 JSON 对象，不要 Markdown、代码围栏或额外文字。

字段 Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}

最小格式示例（内容仅说明格式，不是本周事实）：
{json.dumps(example, ensure_ascii=False, indent=2)}"""
    user_prompt = (
        "请根据以下公开经营信号生成本周角色报告：\n"
        + json.dumps(
            role_signal_payload(signals, role),
            ensure_ascii=False,
            indent=2,
        )
    )
    return system_prompt, user_prompt


def build_repair_prompt(
    signals: AnalysisSignals,
    role: Role,
    invalid_response: str,
    validation_error: str,
) -> tuple[str, str]:
    """修复调用携带完整原输入、非法回答和错误，不能依赖对话历史。"""

    system_prompt, original_user_prompt = build_role_prompts(signals, role)
    repair_prompt = f"""{original_user_prompt}

上一份回答无法通过程序校验。请只修复格式和不符合约束的字段，不要杜撰新证据。

上一份回答：
{invalid_response}

程序校验错误：
{validation_error}

请重新返回一个完整、合法的 JSON 对象。"""
    return system_prompt, repair_prompt
