"""Generate every Agent-visible document from simulator definitions."""

import json
from pathlib import Path
from typing import Dict

from ..simulator.tools import TOOL_DOCS, get_tool_summary_table
from ..simulator.database import TABLE_DOCS
from ..simulator.config import ENTERPRISE_RESPONSE_TIMEOUT_DAYS


# Tools excluded from the novamind_api (not exposed to bash_agent)
_EXCLUDED_TOOLS = {
    'python_exec',
    'register_daily_calculation',
    'remove_daily_calculation',
    'list_daily_calculations',
    'register_script',
    'run_script',
    'list_scripts',
    'delete_script',
    'list_all_tables',
    'describe_tables',
    'get_tool_documentation',
    'next_week',  # CLI command (novamind-operation next-week), not a Python API tool
}

# Map tool names to novamind_api module names
_TOOL_TO_MODULE = {
    'set_prices': 'pricing',
    'set_model_tiers': 'pricing',
    'set_usage_quotas': 'pricing',
    'set_promotion': 'pricing',
    'set_daily_spend': 'marketing',
    'set_targeted_ad_spend': 'marketing',
    'set_ads_strength': 'marketing',
    'set_lead_promotion': 'marketing',
    'set_capacity_tier': 'infrastructure',
    'get_cost_info': 'infrastructure',
    'send_enterprise_deal': 'enterprise',
    'reject_enterprise_deal': 'enterprise',
    'research_market': 'market',
    'research_group': 'market',
    'get_market_overview': 'market',
    'get_group_insights': 'market',
    'start_research_project': 'research',
    'list_research_projects': 'research',
    'get_social_posts': 'analytics',
    'set_targeted_ops_spend': 'analytics',
    'set_targeted_dev_spend': 'analytics',
    'post_social_media': 'marketing',
}


def render_api_docs(output_dir: Path):
    """Render TOOL_DOCS into JSON files grouped by novamind_api module.

    Creates one JSON file per module (e.g., pricing.json, marketing.json).
    Each file contains a list of tool documentation dicts.

    Args:
        output_dir: Directory to write JSON files to (e.g., workspace/docs/api/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group tools by module
    by_module: Dict[str, list] = {}
    for tool_name, doc in TOOL_DOCS.items():
        if tool_name in _EXCLUDED_TOOLS:
            continue
        module = _TOOL_TO_MODULE.get(tool_name, 'other')
        by_module.setdefault(module, []).append({
            'name': tool_name,
            'python_call': f"novamind_api.{module}.{tool_name}(...)",
            'category': doc.get('category', ''),
            'description': doc.get('description', ''),
            'parameters': doc.get('parameters', {}),
            'inputSchema': doc.get('inputSchema', {}),
            'returns': doc.get('returns', {}),
            'output_schema': doc.get('output_schema', {}),
            'impact': doc.get('impact', ''),
            'example_call': doc.get('example_call', {}),
        })

    for module_name, tools in by_module.items():
        filepath = output_dir / f"{module_name}.json"
        filepath.write_text(json.dumps(tools, indent=2, default=str))


def render_table_docs(output_dir: Path):
    """Render TABLE_DOCS into JSON files, one per table.

    Args:
        output_dir: Directory to write JSON files to (e.g., workspace/docs/tables/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, doc in TABLE_DOCS.items():
        filepath = output_dir / f"{table_name}.json"
        # Only include agent-visible columns (not internal_columns)
        table_doc = {
            'table': table_name,
            'description': doc.get('description', ''),
            'columns': doc.get('columns', {}),
        }
        filepath.write_text(json.dumps(table_doc, indent=2))


def render_cli_docs(output_dir: Path):
    """Render CLI documentation from docstrings.

    Args:
        output_dir: Directory to write cli.md to (e.g., workspace/docs/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from ._public_cli import get_cli_docs
    content = get_cli_docs()
    # 两个文件名为公开仓库保留，内容统一来自当前 CLI 实现。
    for filename in ("cli.md", "cli-reference.md"):
        (output_dir / filename).write_text(content)


def render_tools_reference(output_path: Path):
    """Render the complete tool reference as Markdown."""
    lines = [
        "# NovaMind Tools Reference",
        "",
        "Complete reference for all available tools. Call them via the Python API",
        "(`import novamind_api as nm`) — see `docs/novamind_api/` for the SDK source",
        "and run scripts with `./novamind-operation python <script.py>` or",
        "`./novamind-operation python-c \"<inline code>\"`.",
        "",
        "## Tool Summary",
        "",
        get_tool_summary_table(),
        "",
        "---",
        "",
        "## Detailed Tool Documentation",
        "",
    ]

    by_category: Dict[str, list] = {}
    for name, doc in TOOL_DOCS.items():
        if name in _EXCLUDED_TOOLS:
            continue
        category = doc.get("category", "Other")
        by_category.setdefault(category, []).append((name, doc))

    for category in sorted(by_category):
        lines.extend((f"### {category}", ""))
        for name, doc in sorted(by_category[category]):
            module = _TOOL_TO_MODULE.get(name, "other")
            lines.extend(
                (
                    f"#### `{name}`",
                    "",
                    f"**Python:** `novamind_api.{module}.{name}(...)`",
                    "",
                    doc.get("description", ""),
                    "",
                )
            )

            params = doc.get("parameters", {})
            if params:
                lines.extend(("**Parameters:**", ""))
                for parameter_name, description in params.items():
                    lines.append(f"- `{parameter_name}`: {description}")
                lines.append("")

            schema = doc.get("inputSchema", {})
            if schema and schema.get("properties"):
                lines.extend(
                    (
                        "**Input Schema:**",
                        "```json",
                        json.dumps(schema, indent=2, default=str),
                        "```",
                        "",
                    )
                )

            returns = doc.get("returns", {})
            if returns:
                lines.append("**Returns:**")
                if isinstance(returns, dict):
                    for key, value in returns.items():
                        lines.append(f"- {key}: {value}")
                else:
                    lines.append(f"- {returns}")
                lines.append("")

            impact = doc.get("impact", "")
            if impact:
                lines.extend((f"**Impact:** {impact}", ""))

            example = doc.get("example_call", {})
            if example:
                lines.extend(
                    (
                        "**Example:**",
                        "```json",
                        json.dumps(example, indent=2, default=str),
                        "```",
                        "",
                    )
                )

            sample_io = doc.get("sample_io", [])
            if sample_io and isinstance(sample_io, list):
                lines.append("**Sample I/O:**")
                for sample in sample_io[:2]:
                    lines.append(f"*{sample.get('label', 'Example')}:*")
                    if "input" in sample:
                        lines.extend(
                            (
                                "```json",
                                json.dumps(sample["input"], indent=2, default=str),
                                "```",
                            )
                        )
                    if "output" in sample:
                        output = sample["output"]
                        if isinstance(output, str) and len(output) > 500:
                            output = output[:500] + "..."
                        lines.extend(("Output:", "```", str(output), "```"))
                lines.append("")

            lines.extend(("---", ""))

    output_path.write_text("\n".join(lines))


def render_tables_reference(output_path: Path):
    """Render the complete queryable-table reference as Markdown."""
    lines = [
        "# NovaMind Database Tables Reference",
        "",
        "Reference for all queryable database tables. Query via:",
        '- `novamind-operation query "SELECT * FROM table_name LIMIT 10"`',
        '- Python: `novamind_api.query("SELECT * FROM table_name LIMIT 10")`',
        "",
        "**Note:** Schema introspection queries (PRAGMA, sqlite_master) are blocked.",
        "Use this reference or `docs/tables/*.json` for schema information.",
        "",
        "---",
        "",
    ]

    for table_name, doc in sorted(TABLE_DOCS.items()):
        lines.extend((f"## `{table_name}`", "", doc.get("description", ""), ""))
        columns = doc.get("columns", {})
        if columns:
            lines.extend(("| Column | Description |", "|--------|-------------|"))
            for column_name, description in columns.items():
                lines.append(f"| `{column_name}` | {description.replace('|', r'\|')} |")
            lines.append("")
        lines.extend(("---", ""))

    output_path.write_text("\n".join(lines))


def render_simulator_instructions(output_path: Path):
    """Render the simulator instructions with the current tool summary."""
    source_path = (
        Path(__file__).resolve().parents[1] / "agents" / "simulator_instructions.md"
    )
    content = source_path.read_text()
    content = content.replace("{total_days}", "N")
    content = content.replace("{total_years}", "N/365")
    content = content.replace(
        "{enterprise_response_timeout_days}",
        str(ENTERPRISE_RESPONSE_TIMEOUT_DAYS),
    )
    content = content.replace("{tool_list}", get_tool_summary_table())
    output_path.write_text(content)


def generate_public_docs(output_dir: Path):
    """Regenerate all stable public documentation entry points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    render_api_docs(output_dir / "api")
    render_table_docs(output_dir / "tables")
    render_tools_reference(output_dir / "tools-reference.md")
    render_tables_reference(output_dir / "tables-reference.md")
    render_cli_docs(output_dir)
    render_simulator_instructions(output_dir / "simulator-instructions.md")


def initialize_workspace(workspace_path: Path):
    """Initialize a bash_agent per-session scratch directory.

    After the zipapp refactor the docs + SDK source live in the *published
    repo root* (``<base>/docs/*``), not per-session. This keeps the workspace
    to just the ephemeral state the agent writes itself.

    Args:
        workspace_path: Per-session scratch root for the agent.
    """
    workspace_path.mkdir(parents=True, exist_ok=True)
