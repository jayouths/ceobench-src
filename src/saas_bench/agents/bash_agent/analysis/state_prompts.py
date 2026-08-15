"""统一经营状态重构的 Prompt。"""

from __future__ import annotations

import json

from .models import DIMENSION_LABELS, RoleReportsArtifact, StateAssessment


def state_input_payload(role_reports: RoleReportsArtifact) -> dict:
    """只向状态重构模型提供四份业务报告，不暴露调用元数据。"""

    return {
        "day": role_reports.day,
        "role_reports": [
            report.model_dump(mode="json") for report in role_reports.reports
        ],
    }


def build_state_prompts(
    role_reports: RoleReportsArtifact,
) -> tuple[str, str]:
    labels = {
        dimension.value: sorted(values)
        for dimension, values in DIMENSION_LABELS.items()
    }
    example = {
        "diagnosis": "用一句话概括当前最重要的经营状态，不提出行动建议",
        "dimensions": [{
            "dimension": "cash_health",
            "label": "watch",
            "confidence": 0.7,
            "evidence_ids": ["FIN-1"],
            "rationale": "说明标签与引用证据之间的关系",
        }],
        "facts": [{
            "statement": "可由报告证据直接支持的跨角色经营事实",
            "evidence_ids": ["FIN-1", "CUS-1"],
            "confidence": 0.8,
        }],
        "hypotheses": [{
            "cause": "尚待验证的原因解释",
            "evidence_for": ["PRO-1"],
            "evidence_against": ["CUS-2"],
            "competing_causes": ["另一种同样可能的解释"],
            "confidence": 0.6,
            "validation_test": "下一周应观察哪些公开证据来区分解释",
        }],
        "latent_risks": [{
            "risk": "尚未充分反映在头部指标中的风险",
            "evidence_ids": ["PRO-1", "CUS-1"],
            "early_indicator": "后续应观察的公开先行指标",
            "horizon_weeks": 2,
            "severity": 3,
        }],
        "causal_chain": [{
            "cause": "证据支持的起点",
            "effect": "可能产生的直接结果",
            "evidence_ids": ["MAR-1", "PRO-1"],
            "confidence": 0.6,
        }],
    }
    system_prompt = f"""你是企业经营状态识别系统的状态重构器。

你的任务是合并市场、财务、产品和客户四份报告，形成一份可追溯的经营画像。你只识别状态，不制定策略。

必须遵守：
1. 只能使用用户提供的四份角色报告，不得补充外部事实、隐藏状态或行动建议。
2. dimensions 必须且只能包含五个固定维度，每个维度恰好一次；标签必须来自对应枚举。
3. 非 insufficient_data 的维度必须引用输入中真实存在的 evidence id。
4. facts 只保留可直接支持的经营事实。优先使用至少两个独立证据；单一证据只能用于无歧义的财务或服务指标。
5. hypotheses 必须区分支持证据、反对证据和竞争性解释，不得把相关性写成已确认因果。
6. latent_risks 只保留已有早期证据、但尚未充分暴露的风险。
7. causal_chain 的每一步只表达一个直接的 cause → effect，并引用支持该步骤的证据。
8. 所有 evidence id 必须原样来自输入，不能新造、改写或引用角色报告中的 risk 文本代替证据。
9. 输入证据不足时使用 insufficient_data 或空数组，不得为了填满字段而杜撰内容。
10. 只返回一个符合 Schema 的 JSON 对象，不要 Markdown、代码围栏或额外文字。

五维标签枚举：
{json.dumps(labels, ensure_ascii=False, indent=2)}

字段 Schema：
{json.dumps(StateAssessment.model_json_schema(), ensure_ascii=False, indent=2)}

最小格式示例（只说明字段写法；实际输出必须包含全部五个维度）：
{json.dumps(example, ensure_ascii=False, indent=2)}"""
    user_prompt = (
        "请根据以下四份角色报告重构本周经营状态：\n"
        + json.dumps(
            state_input_payload(role_reports),
            ensure_ascii=False,
            indent=2,
        )
    )
    return system_prompt, user_prompt


def build_state_repair_prompt(
    role_reports: RoleReportsArtifact,
    invalid_response: str,
    validation_error: str,
) -> tuple[str, str]:
    """修复请求携带完整输入与错误，不依赖上一轮聊天历史。"""

    system_prompt, original_user_prompt = build_state_prompts(role_reports)
    user_prompt = f"""{original_user_prompt}

上一份回答无法通过程序校验。请只修复不符合约束的内容，不得增加输入中不存在的证据。

上一份回答：
{invalid_response}

程序校验错误：
{validation_error}

请重新返回一个完整、合法的 JSON 对象。"""
    return system_prompt, user_prompt
