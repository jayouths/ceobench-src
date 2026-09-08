"""Assert every tool and queryable table has proper documentation coverage.

Run: uv run python -m pytest tests/test_docs_coverage.py -v
"""
import json
import tempfile
from pathlib import Path

from saas_bench.simulator.config import (
    BenchmarkConfig,
    DEFAULT_ENTERPRISE_OPS_SCALE,
    DEFAULT_INDIVIDUAL_OPS_SCALE,
    ENTERPRISE_RESPONSE_TIMEOUT_DAYS,
    MIN_RESEARCH_DURATION_DAYS,
    MIN_RESEARCH_QUALITY_BOOST,
    MODEL_TIERS,
    RESEARCH_TIERS,
)
from saas_bench.simulator.tools import TOOL_DOCS, get_tool_descriptions
from saas_bench.simulator.database import TABLE_DOCS, init_database
from saas_bench.runtime.docs_generator import (
    _EXCLUDED_TOOLS,
    _TOOL_TO_MODULE,
    generate_public_docs,
)
from saas_bench.runtime._public_cli import _build_parser, get_cli_docs


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
