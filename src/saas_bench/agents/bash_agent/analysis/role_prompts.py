"""四个职能角色的 Prompt 和隔离输入。"""

from __future__ import annotations

import json

from .evidence_cards import build_evidence_cards
from .models import EvidenceCard, Role, RoleSelection
from .signal_models import AnalysisSignals


_ROLE_NAMES = {
    Role.MARKET: "市场分析",
    Role.FINANCE: "财务分析",
    Role.PRODUCT: "产品与技术分析",
    Role.CUSTOMER: "客户成功分析",
}

_ROLE_GUIDANCE = {
    Role.MARKET: (
        "判断需求、获客来源、付费获客效率、公开反馈和宏观环境中哪些信号最重要。"
        "不要分析内部成本、产品配置或提出营销动作。"
    ),
    Role.FINANCE: (
        "判断现金健康度、收入质量、成本结构、交付利润率和现金跑道。"
        "区分经常性成本与一次性投资，不要把模型 API 成本当作企业现金流。"
    ),
    Role.PRODUCT: (
        "判断使用量、容量压力、可靠性、配置变化、研发管线和交付质量。"
        "配置变化与服务波动只能形成待验证假设，不能直接断言因果。"
    ),
    Role.CUSTOMER: (
        "判断客户基础、新增付费、流失、工单压力和企业谈判状态。"
        "账户数与企业席位数不可混合，公开信息不足时必须保留不确定性。"
    ),
}


def role_input_payload(
    signals: AnalysisSignals,
    role: Role,
    cards: list[EvidenceCard] | None = None,
) -> dict:
    """角色只能看到本职证据卡片，不接触其他角色信号。"""

    role_cards = cards if cards is not None else build_evidence_cards(signals, role)
    return {
        "day": signals.day,
        "week": signals.week,
        "evidence_cards": [card.model_dump(mode="json") for card in role_cards],
    }


def build_role_prompts(
    signals: AnalysisSignals,
    role: Role,
    cards: list[EvidenceCard] | None = None,
) -> tuple[str, str]:
    role_cards = cards if cards is not None else build_evidence_cards(signals, role)
    first_id = role_cards[0].id
    schema = RoleSelection.model_json_schema()
    example = {
        "selected_evidence_ids": [first_id],
        "hypotheses": [{
            "cause": "该事实的一种待验证原因；不能写成已经确认的因果",
            "evidence_ids": [first_id],
            "confidence": 0.5,
            "validation": "说明未来观察哪个公开信号以及预期现象",
        }],
        "risks": [{
            "risk": "由当前事实支持、但后果尚未充分暴露的风险",
            "evidence_ids": [first_id],
            "early_indicator": "后续应持续观察的公开指标",
            "horizon_weeks": 2,
            "severity": 3,
        }],
    }
    system_prompt = f"""你是企业经营状态识别系统中的{_ROLE_NAMES[role]}角色。

输入中的 evidence_cards 已由程序根据公开经营数据确定性生成。每张卡片都是不可修改的事实。你的任务是选择最重要的事实，并谨慎提出可能原因和潜在风险；你不负责重新计算或改写事实。{_ROLE_GUIDANCE[role]}

必须遵守：
1. selected_evidence_ids 至少 1 个、最多 5 个，只能原样复制输入中的 id，表示应进入统一画像的核心事实，并按重要性排序。
2. 不要在输出中重新填写事实、指标路径、数值或变化方向。卡片已明确写出比较是否成立；比较数据不足时，禁止使用“上升、下降、改善、恶化、持续”等当前趋势判断。例如单周净现金流为负只能写“本周净流出”，不能写“持续净流出”。描述未来风险时可以写“若净流出持续”，但不能把尚未观察到的持续性写成既成事实。
3. hypotheses 最多 3 条。每条都必须引用输入中真实存在的证据卡片，使用“可能、推测、待验证”等表述，并给出能够通过后续 evidence_cards 中已有指标执行的验证方式；不得引入行业基准、客户满意度、工单分类等输入中不存在的数据。用于补充解释的证据不必重复加入 selected_evidence_ids。
4. risks 最多 3 条。每条都必须引用输入中真实存在的证据卡片，只写已有早期信号支持但后果尚未充分暴露的风险。用于支持风险的证据不必重复加入 selected_evidence_ids。
5. 不得输出行动建议，不得补充外部事实或隐藏状态。数据不足时 hypotheses 和 risks 可以为空。
6. 只返回一个符合 Schema 的 JSON 对象，不要 Markdown、代码围栏或额外文字。

字段含义：
- selected_evidence_ids：本角色认为最值得进入统一经营画像的确定性事实。
- hypotheses：对已选事实成因的待验证解释；confidence 是解释成立的置信度。
- risks：已有早期证据支持的潜在后果；severity 表示若风险发生时的影响程度，不表示发生概率。

输出 Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}

最小合法示例（内容只说明格式，不是本周结论）：
{json.dumps(example, ensure_ascii=False, indent=2)}"""
    user_prompt = (
        "请根据以下确定性证据卡片生成本周角色报告：\n"
        + json.dumps(
            role_input_payload(signals, role, role_cards),
            ensure_ascii=False,
            indent=2,
        )
    )
    return system_prompt, user_prompt


def build_repair_prompt(
    signals: AnalysisSignals,
    role: Role,
    cards: list[EvidenceCard],
    invalid_response: str,
    validation_error: str,
) -> tuple[str, str]:
    """修复时重新审查完整报告，避免只删除字段来绕过校验。"""

    system_prompt, original_user_prompt = build_role_prompts(signals, role, cards)
    repair_prompt = f"""{original_user_prompt}

上一份回答无法通过程序校验。请根据错误重新审查整份报告：受错误证据影响的选择、假设和风险都要同步修正，不能只删除字段或替换 id 来绕过校验，也不得杜撰新事实。

上一份回答：
{invalid_response}

程序校验错误：
{validation_error}

请重新返回一个完整、合法的 JSON 对象。"""
    return system_prompt, repair_prompt
