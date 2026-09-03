"""统一经营状态重构的 Prompt。"""

from __future__ import annotations

import json

from .models import Role, RoleReportsArtifact, StateAssessment


_DIMENSION_RULES = """
- cash_health：综合现金余额、净现金流和现金跑道。healthy 表示现金与现金流均无明显压力；watch 表示当前现金尚可但已出现消耗信号；stressed 表示现金跑道已有可核验的明显压力；critical 表示接近无法持续经营；缺少直接财务证据时使用 insufficient_data。当现金为正、跑道因历史不足无法计算且只有一周负现金流时，最高只能判断为 watch，不能仅凭单周流出判断为 stressed 或 critical。
- demand_momentum：综合有效线索、新增付费和客户净变化。contracting 表示需求或客户基础收缩；stable 表示基本持平；growing 表示存在明确正增长；surging 仅用于多个独立需求信号同时显著增强；无法判断当前变化时使用 insufficient_data。本周新增付费和客户净变化是已实现的期间增量，其中任一明确为正且没有更强的收缩证据时，应判断为 growing；获客渠道单一、企业线索为零等信号只影响增长质量和风险，不能把已实现的净增长改判为 stable。
- unit_economics：综合经营收入、服务交付成本和交付利润率。healthy 表示收入能够覆盖交付成本且有合理余量；marginal 表示接近盈亏平衡或余量很薄；loss_making 表示交付收入不足以覆盖对应成本；缺少收入或成本证据时使用 insufficient_data。
- service_pressure：综合容量利用、过载、延迟、错误率、停机和工单。underutilized 表示容量明显闲置且没有客户可感知的服务异常；balanced 表示负载和质量处于正常范围；pressured 表示已经发生停机、明显质量波动、积压或容量压力，但尚未严重失控；overloaded 表示有直接证据表明过载或严重服务故障。可靠性和客户影响优先于单纯的低利用率；产品配置变化或社交帖子不能单独决定该标签。缺少容量与可靠性证据时使用 insufficient_data。
- customer_health：综合客户净变化、流失、工单积压和企业谈判。healthy 表示客户基础与服务关系稳定；watch 表示已有局部早期风险；deteriorating 表示客户或服务关系明确恶化；critical 表示大规模流失或严重积压威胁持续经营；缺少客户证据时使用 insufficient_data。
""".strip()


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
    first_id_by_role = {
        report.role: report.evidence[0].id for report in role_reports.reports
    }
    market_id = first_id_by_role[Role.MARKET]
    finance_id = first_id_by_role[Role.FINANCE]
    product_id = first_id_by_role[Role.PRODUCT]
    customer_id = first_id_by_role[Role.CUSTOMER]
    example = {
        "diagnosis": "用一句话概括最重要的经营状态，不提出行动建议",
        "dimensions": [
            {
                "dimension": "cash_health",
                "label": "watch",
                "confidence": 0.7,
                "evidence_ids": [finance_id],
                "rationale": "解释财务证据为何对应当前标签",
            },
            {
                "dimension": "demand_momentum",
                "label": "stable",
                "confidence": 0.6,
                "evidence_ids": [market_id],
                "rationale": "解释需求证据为何对应当前标签",
            },
            {
                "dimension": "unit_economics",
                "label": "marginal",
                "confidence": 0.6,
                "evidence_ids": [finance_id],
                "rationale": "解释收入和成本证据为何对应当前标签",
            },
            {
                "dimension": "service_pressure",
                "label": "balanced",
                "confidence": 0.7,
                "evidence_ids": [product_id],
                "rationale": "解释服务证据为何对应当前标签",
            },
            {
                "dimension": "customer_health",
                "label": "watch",
                "confidence": 0.6,
                "evidence_ids": [customer_id],
                "rationale": "解释客户证据为何对应当前标签",
            },
        ],
        "key_evidence_ids": [finance_id, market_id, product_id],
        "hypotheses": [{
            "cause": "一种需要后续数据验证的跨角色解释",
            "evidence_for": [market_id, customer_id],
            "evidence_against": [],
            "competing_causes": [],
            "confidence": 0.5,
            "validation_test": "说明下一周观察哪些公开信号来区分解释",
        }],
        "latent_risks": [{
            "risk": "已有早期证据支持、但后果尚未充分暴露的风险",
            "evidence_ids": [product_id],
            "early_indicator": "后续应持续观察的公开指标",
            "horizon_weeks": 2,
            "severity": 3,
        }],
    }
    system_prompt = f"""你是企业经营状态识别系统的状态重构器。

输入包含市场、财务、产品和客户四份角色报告。evidence 中的 fact 由程序根据公开经营数据确定性生成，不可改写。你的任务是跨角色判断当前经营状态，并选择最重要的事实；你只识别状态，不制定策略。

五维标签含义：
{_DIMENSION_RULES}

必须遵守：
1. dimensions 必须且只能包含五个固定维度，每个维度恰好一次，并按上面的标签含义判断。
2. key_evidence_ids 最多 3 个，只能引用输入中的真实 evidence id，按对当前经营决策的重要性排序。最终简报会直接展示对应 fact，因此不要重写数字事实。
3. diagnosis、dimension rationale、hypotheses 和 latent_risks 只能使用输入证据支持。证据明确写着比较数据不足时，不得使用“上升、下降、改善、恶化、持续”等当前趋势结论。例如单周净现金流为负只能写“本周净流出”，不能写“持续净流出”。描述未来风险时可以写“若净流出持续”，但不能把尚未观察到的持续性写成既成事实。
4. hypotheses 最多 2 条，必须区分支持证据、实际存在的反对证据和竞争解释。没有反对证据或竞争解释时使用空数组，不得为了填字段而编造。
5. latent_risks 最多 2 条，只保留已有早期证据、但后果尚未充分暴露的风险。
6. 所有 evidence id 必须原样来自输入；不得补充外部事实、隐藏状态、行动建议或已经确认的因果关系。
7. 只返回一个符合 Schema 的 JSON 对象，不要 Markdown、代码围栏或额外文字。

输出 Schema：
{json.dumps(StateAssessment.model_json_schema(), ensure_ascii=False, indent=2)}

完整格式示例（内容只说明格式，不是本周结论）：
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
    """修复时重新检查所有受影响结论，不做仅能通过 Schema 的局部补丁。"""

    system_prompt, original_user_prompt = build_state_prompts(role_reports)
    user_prompt = f"""{original_user_prompt}

上一份回答无法通过程序校验。请根据错误重新审查整份经营画像：受错误证据影响的维度、诊断、假设和风险都要同步修正，不能只删除字段或替换 id 来绕过校验。

上一份回答：
{invalid_response}

程序校验错误：
{validation_error}

请重新返回一个完整、合法的 JSON 对象。"""
    return system_prompt, user_prompt
