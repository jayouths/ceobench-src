"""Assert every tool and queryable table has proper documentation coverage.

Run: uv run python -m pytest tests/test_docs_coverage.py -v
"""
import importlib
import json
import tempfile
from inspect import Parameter, signature
from pathlib import Path

from saas_bench.novamind_api.analytics import set_targeted_ops_spend
from saas_bench.novamind_api.market import research_group
from saas_bench.simulator.config import (
    BenchmarkConfig,
    DEFAULT_ENTERPRISE_OPS_SCALE,
    DEFAULT_INDIVIDUAL_OPS_SCALE,
    ENTERPRISE_RESPONSE_TIMEOUT_DAYS,
    INITIAL_CUSTOMER_GROUPS,
    MIN_RESEARCH_DURATION_DAYS,
    MIN_RESEARCH_QUALITY_BOOST,
    MODEL_TIERS,
    RESEARCH_TIERS,
    get_development_quality_formula,
    get_info_level_noise_label,
)
from saas_bench.simulator.tools import TOOL_DOCS, get_tool_descriptions
from saas_bench.simulator.database import TABLE_DOCS, init_database
from saas_bench.runtime._public_cli import _build_parser, get_cli_docs
from saas_bench.runtime.api_server import _TOOL_DISPATCH
from saas_bench.runtime.docs_generator import (
    _EXCLUDED_TOOLS,
    _TOOL_TO_MODULE,
    generate_public_docs,
)


_REMOVED_LEGACY_TOOLS = {
    "register_daily_calculation",
    "remove_daily_calculation",
    "list_daily_calculations",
    "list_all_tables",
    "describe_tables",
    "get_tool_documentation",
    "register_script",
    "run_script",
    "list_scripts",
    "delete_script",
}


def test_cli_docs_follow_public_parser():
    """CLI 文档必须覆盖 Agent 真实可用的命令和新版预测参数。"""
    _, commands = _build_parser()
    docs = get_cli_docs()

    for command in commands:
        assert f"./novamind-operation {command}" in docs
    assert "cash_26wk_upper" in docs
    assert "<cash_1wk> <cash_4wk> <cash_12wk>" not in docs


def test_operations_docs_follow_simulator_defaults():
    """Agent 看到的运营系数必须与模拟器默认规则一致。"""
    config = BenchmarkConfig()
    assert config.individual_ops_scale == DEFAULT_INDIVIDUAL_OPS_SCALE
    assert config.enterprise_ops_scale == DEFAULT_ENTERPRISE_OPS_SCALE

    operations_docs = json.dumps(
        {
            "global": TOOL_DOCS["set_daily_spend"],
            "targeted": TOOL_DOCS["set_targeted_ops_spend"],
        }
    )
    assert f"scale_g = {config.individual_ops_scale:g}" in operations_docs
    assert f"scale_g = {config.enterprise_ops_scale:g}" in operations_docs


def test_ads_strength_guidance_matches_simulator_rules():
    """获客投放与产品内广告必须在 Agent 说明中保持分离。"""
    config = BenchmarkConfig()
    instructions = Path(
        "src/saas_bench/agents/simulator_instructions.md"
    ).read_text()
    system_prompt = Path(
        "src/saas_bench/agents/bash_agent/system_prompt.md"
    ).read_text()

    assert config.ads_strength_global == 0.0
    assert "ads_strength = 0.0" in instructions
    assert "it does not change lead generation" in instructions
    assert "multiplier on ad effectiveness" not in instructions
    assert 'by_group={"S2": 1.5}' not in system_prompt


def test_model_tier_docs_follow_simulator_rules(make_initialized_sim, make_agent_tools):
    """档位配置、工具返回和 Agent 可见示例必须使用同一组模型参数。"""
    conn, _, config = make_initialized_sim()
    result = make_agent_tools(conn, config).get_cost_info()
    tool_description = next(
        tool["description"]
        for tool in get_tool_descriptions()
        if tool["name"] == "get_cost_info"
    )

    for tier, spec in MODEL_TIERS.items():
        returned = result.data["model_tiers"][tier]
        documented = TOOL_DOCS["get_cost_info"]["returns"]["success"][
            "model_tiers"
        ][str(tier)]
        configured = TOOL_DOCS["set_model_tiers"]["tier_info"][str(tier)]

        assert returned["cost_per_usage_unit"] == spec.unit_cost
        assert returned["quality_multiplier"] == spec.quality_multiplier
        assert returned["class"] == spec.model_class
        assert documented == returned
        assert configured["cost_per_unit"] == spec.unit_cost
        assert configured["quality_multiplier"] == spec.quality_multiplier
        assert configured["class"] == spec.model_class
        assert f"(q={spec.quality_multiplier:.2f})" in tool_description


def test_enterprise_timeout_docs_follow_simulator_rule(tmp_path):
    """企业回复期限在工具说明和主说明中必须与真实规则一致。"""
    timeout = str(ENTERPRISE_RESPONSE_TIMEOUT_DAYS)
    enterprise_docs = json.dumps(TOOL_DOCS["send_enterprise_deal"])
    assert f"within {timeout} days" in enterprise_docs

    output_dir = tmp_path / "docs"
    generate_public_docs(output_dir)
    instructions = (output_dir / "simulator-instructions.md").read_text()
    assert f"{timeout}-day timeout" in instructions
    assert "{enterprise_response_timeout_days}" not in instructions


def test_enterprise_deal_docs_only_advertise_monthly_contracts():
    """企业报价的正式说明和成功示例不得继续宣传多月合同。"""
    docs = TOOL_DOCS["send_enterprise_deal"]
    example_deals = docs["example_call"]["arguments"]["deals"]
    successful_examples = docs["sample_io"]["success"]

    for _, offerings in example_deals:
        assert all(len(offering) == 2 for offering in offerings)
    for example in successful_examples:
        for _, offerings in example["input"]["deals"]:
            assert all(len(offering) == 2 for offering in offerings)

    assert "month-to-month" in docs["description"]
    assert "duration is not configurable" in docs["impact"]
    assert "longer contracts" not in docs["impact"]


def test_enterprise_deal_schema_constrains_offerings():
    """Provider 必须直接看到报价数量、套餐和价格边界。"""
    deal_schema = TOOL_DOCS["send_enterprise_deal"]["inputSchema"]["properties"][
        "deals"
    ]["items"]
    offerings_schema = deal_schema["prefixItems"][1]
    offering_schema = offerings_schema["items"]

    assert (deal_schema["minItems"], deal_schema["maxItems"]) == (2, 2)
    assert (offerings_schema["minItems"], offerings_schema["maxItems"]) == (1, 3)
    assert offering_schema["prefixItems"][0]["enum"] == ["A", "B", "C"]
    assert offering_schema["prefixItems"][1]["minimum"] == 0


def test_group_research_requires_explicit_target_level():
    """工具 schema 与公开 Python 客户端都必须要求研究目标等级。"""
    docs = TOOL_DOCS["research_group"]
    assert "target_level" in docs["inputSchema"]["required"]
    assert (
        signature(research_group).parameters["target_level"].default
        is Parameter.empty
    )


def test_targeted_ops_docs_match_public_sdk():
    """定向运营的四种范围必须在文档和公开客户端中同时可用。"""
    expected_scopes = {"by_group", "by_plan", "by_group_plan", "by_customer"}
    docs = TOOL_DOCS["set_targeted_ops_spend"]
    sdk_parameters = set(signature(set_targeted_ops_spend).parameters)

    assert set(docs["inputSchema"]["properties"]) == expected_scopes
    assert sdk_parameters == expected_scopes


def test_research_docs_follow_configured_tiers(tmp_path):
    """研发工具和公开说明不得再维护另一套档位参数。"""
    list_docs = TOOL_DOCS["list_research_projects"]
    assert str(len(RESEARCH_TIERS)) in list_docs["description"]
    assert list_docs["returns"]["data"]["tiers"][0]["cost"] == (
        RESEARCH_TIERS[0].cost
    )
    mechanics = TOOL_DOCS["start_research_project"]["mechanics"]
    assert str(MIN_RESEARCH_DURATION_DAYS) in mechanics["duration"]
    assert f"{MIN_RESEARCH_QUALITY_BOOST:g}" in mechanics["quality_boost"]

    output_dir = tmp_path / "docs"
    generate_public_docs(output_dir)
    instructions = (output_dir / "simulator-instructions.md").read_text()
    assert f"{len(RESEARCH_TIERS)} independent tiers" in instructions
    assert "{research_tier_count}" not in instructions
    assert "35–380" not in instructions
    assert "+0.04 to +0.85" not in instructions


def test_market_tool_examples_match_default_information_levels():
    """工具示例可以是假设场景，但等级与误差必须符合默认规则。"""
    config = BenchmarkConfig()
    level_one = get_info_level_noise_label(config, 1)

    overview_docs = TOOL_DOCS["get_market_overview"]
    early_game = overview_docs["sample_io"]["success"][0]["output"]
    for group_id in INITIAL_CUSTOMER_GROUPS:
        group_line = next(
            line for line in early_game.splitlines()
            if line.strip().startswith(f"{group_id}:")
        )
        assert f"Level 1 ({level_one})" in group_line

    insight_examples = TOOL_DOCS["get_group_insights"]["sample_io"]["success"]
    initial_example = next(
        example for example in insight_examples
        if example["label"] == "Initial group before research"
    )
    assert f"Info Level: 1 ({level_one})" in initial_example["output"]
    assert "full accuracy" not in str(insight_examples).lower()

    docs_text = json.dumps(
        {
            "overview": overview_docs,
            "insights": TOOL_DOCS["get_group_insights"],
        },
        ensure_ascii=False,
    )
    noise_labels = {
        get_info_level_noise_label(config, level) for level in range(1, 6)
    }
    for level in range(1, 6):
        expected = get_info_level_noise_label(config, level)
        for wrong_label in noise_labels - {expected}:
            assert f"Level {level} ({wrong_label})" not in docs_text
            assert f"Info Level: {level} ({wrong_label})" not in docs_text


def test_development_quality_docs_follow_simulator_formula(tmp_path):
    """研发工具与 Agent 主说明不得再维护独立增益公式。"""
    global_formula = get_development_quality_formula("global_spend")
    targeted_formula = get_development_quality_formula(
        "targeted_spend",
        targeted=True,
    )
    tool_docs = json.dumps(
        {
            "daily": TOOL_DOCS["set_daily_spend"],
            "targeted": TOOL_DOCS["set_targeted_dev_spend"],
        },
        ensure_ascii=False,
    )
    assert global_formula in tool_docs
    assert targeted_formula in tool_docs
    assert "0.006 × ln(1 + global_spend/5000)" not in tool_docs
    assert "0.005 × log(1 + spend/1000)" not in tool_docs

    output_dir = tmp_path / "docs"
    generate_public_docs(output_dir)
    instructions = (output_dir / "simulator-instructions.md").read_text()
    assert global_formula in instructions
    assert targeted_formula in instructions
    assert "{global_development_quality_formula}" not in instructions
    assert "{targeted_development_quality_formula}" not in instructions


def test_complete_public_docs_generator_writes_every_stable_entry(tmp_path):
    """完整生成入口不得只更新 JSON，而遗漏 Agent 会直接阅读的 Markdown。"""
    output_dir = tmp_path / "docs"

    generate_public_docs(output_dir)

    expected_files = {
        "cli.md",
        "cli-reference.md",
        "simulator-instructions.md",
        "tables-reference.md",
        "tools-reference.md",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert (output_dir / "cli.md").read_text() == (
        output_dir / "cli-reference.md"
    ).read_text()
    assert list((output_dir / "api").glob("*.json"))
    assert list((output_dir / "tables").glob("*.json"))


def test_all_tools_mapped_or_excluded():
    """Every tool in TOOL_DOCS must be in _TOOL_TO_MODULE or _EXCLUDED_TOOLS."""
    all_tools = set(TOOL_DOCS.keys())
    mapped = set(_TOOL_TO_MODULE.keys())
    excluded = _EXCLUDED_TOOLS
    unmapped = all_tools - mapped - excluded

    assert not unmapped, (
        f"Tools in TOOL_DOCS not in _TOOL_TO_MODULE or _EXCLUDED_TOOLS: {sorted(unmapped)}. "
        f"Add them to docs_generator.py _TOOL_TO_MODULE (for docs/api/) or _EXCLUDED_TOOLS."
    )


def test_no_stale_tool_mappings():
    """Every tool in _TOOL_TO_MODULE must exist in TOOL_DOCS."""
    for tool_name in _TOOL_TO_MODULE:
        assert tool_name in TOOL_DOCS, (
            f"Tool '{tool_name}' in _TOOL_TO_MODULE but not in TOOL_DOCS. Remove it or add the tool."
        )


def test_no_stale_excluded_tools():
    """Every tool in _EXCLUDED_TOOLS must exist in TOOL_DOCS."""
    for tool_name in _EXCLUDED_TOOLS:
        assert tool_name in TOOL_DOCS, (
            f"Tool '{tool_name}' in _EXCLUDED_TOOLS but not in TOOL_DOCS. Remove it or add the tool."
        )


def test_public_python_tools_match_dispatch_and_sdk():
    """公开工具必须同时具备文档映射、服务端分发和 SDK 调用入口。"""
    documented_tools = set(_TOOL_TO_MODULE)
    assert documented_tools == set(_TOOL_DISPATCH)

    for tool_name, module_name in _TOOL_TO_MODULE.items():
        module = importlib.import_module(f"saas_bench.novamind_api.{module_name}")
        assert callable(getattr(module, tool_name, None)), (
            f"Documented tool '{tool_name}' has no callable SDK entry in "
            f"novamind_api.{module_name}"
        )


def test_public_tools_reference_excludes_non_sdk_and_removed_tools(tmp_path):
    """Python 工具总表不得再次列出 CLI 命令或已删除的旧接口。"""
    output_dir = tmp_path / "docs"
    generate_public_docs(output_dir)
    reference = (output_dir / "tools-reference.md").read_text()

    for tool_name in _TOOL_TO_MODULE:
        assert f"`{tool_name}`" in reference
    for tool_name in _EXCLUDED_TOOLS | _REMOVED_LEGACY_TOOLS:
        assert f"`{tool_name}`" not in reference


def test_table_docs_match_real_tables():
    """Every TABLE_DOCS entry must correspond to a real table in the DB."""
    tmpdir = tempfile.mkdtemp()
    try:
        conn = init_database(Path(tmpdir) / 'test.db')
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        stale = [t for t in TABLE_DOCS if t not in tables]
        assert not stale, (
            f"TABLE_DOCS entries with no matching DB table: {stale}. Remove them."
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir)
