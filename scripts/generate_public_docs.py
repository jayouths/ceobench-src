#!/usr/bin/env python3
"""Generate static documentation for the public NovaMind Bench repo.

Renders TABLE_DOCS, TOOL_DOCS, CLI docs, and simulator instructions
into the public/ directory structure.

Usage:
    cd projects/saas-bench
    uv run --frozen python scripts/generate_public_docs.py [--output public/docs]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saas_bench.simulator.tools import TOOL_DOCS, get_tool_summary_table
from saas_bench.simulator.database import TABLE_DOCS
from saas_bench.runtime.docs_generator import (
    render_api_docs,
    render_table_docs,
    _EXCLUDED_TOOLS,
    _TOOL_TO_MODULE,
)


def render_tools_reference(output_path: Path):
    """Render TOOL_DOCS as a comprehensive markdown reference."""
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

    # Group by category
    by_category = {}
    for name, doc in TOOL_DOCS.items():
        if name in _EXCLUDED_TOOLS:
            continue
        cat = doc.get("category", "Other")
        by_category.setdefault(cat, []).append((name, doc))

    for category in sorted(by_category.keys()):
        lines.append(f"### {category}")
        lines.append("")

        for name, doc in sorted(by_category[category]):
            module = _TOOL_TO_MODULE.get(name, "other")
            lines.append(f"#### `{name}`")
            lines.append("")
            lines.append(f"**Python:** `novamind_api.{module}.{name}(...)`")
            lines.append("")
            lines.append(doc.get("description", ""))
            lines.append("")

            # Parameters
            params = doc.get("parameters", {})
            if params:
                lines.append("**Parameters:**")
                lines.append("")
                for pname, pdesc in params.items():
                    lines.append(f"- `{pname}`: {pdesc}")
                lines.append("")

            # Input schema
            schema = doc.get("inputSchema", {})
            if schema and schema.get("properties"):
                lines.append("**Input Schema:**")
                lines.append("```json")
                lines.append(json.dumps(schema, indent=2, default=str))
                lines.append("```")
                lines.append("")

            # Returns
            returns = doc.get("returns", {})
            if returns:
                lines.append("**Returns:**")
                if isinstance(returns, dict):
                    for rkey, rval in returns.items():
                        lines.append(f"- {rkey}: {rval}")
                else:
                    lines.append(f"- {returns}")
                lines.append("")

            # Impact
            impact = doc.get("impact", "")
            if impact:
                lines.append(f"**Impact:** {impact}")
                lines.append("")

            # Example
            example = doc.get("example_call", {})
            if example:
                lines.append("**Example:**")
                lines.append("```json")
                lines.append(json.dumps(example, indent=2, default=str))
                lines.append("```")
                lines.append("")

            # Sample I/O
            sample_io = doc.get("sample_io", [])
            if sample_io and isinstance(sample_io, list):
                lines.append("**Sample I/O:**")
                for sample in sample_io[:2]:  # Show at most 2 examples
                    label = sample.get("label", "Example")
                    lines.append(f"*{label}:*")
                    if "input" in sample:
                        lines.append("```json")
                        lines.append(json.dumps(sample["input"], indent=2, default=str))
                        lines.append("```")
                    if "output" in sample:
                        output = sample["output"]
                        if isinstance(output, str) and len(output) > 500:
                            output = output[:500] + "..."
                        lines.append("Output:")
                        lines.append("```")
                        lines.append(str(output))
                        lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

    output_path.write_text("\n".join(lines))


def render_tables_reference(output_path: Path):
    """Render TABLE_DOCS as a comprehensive markdown reference."""
    lines = [
        "# NovaMind Database Tables Reference",
        "",
        "Reference for all queryable database tables. Query via:",
        "- `novamind-operation query \"SELECT * FROM table_name LIMIT 10\"`",
        "- Python: `novamind_api.query(\"SELECT * FROM table_name LIMIT 10\")`",
        "",
        "**Note:** Schema introspection queries (PRAGMA, sqlite_master) are blocked.",
        "Use this reference or `docs/tables/*.json` for schema information.",
        "",
        "---",
        "",
    ]

    for table_name, doc in sorted(TABLE_DOCS.items()):
        desc = doc.get("description", "")
        columns = doc.get("columns", {})

        lines.append(f"## `{table_name}`")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if columns:
            lines.append("| Column | Description |")
            lines.append("|--------|-------------|")
            for col_name, col_desc in columns.items():
                # Escape pipe characters in descriptions
                col_desc_safe = col_desc.replace("|", "\\|")
                lines.append(f"| `{col_name}` | {col_desc_safe} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines))


def render_cli_reference(output_path: Path):
    """Render CLI reference documentation."""
    from saas_bench.runtime._public_cli import get_cli_docs

    output_path.write_text(get_cli_docs())


def render_simulator_instructions(output_path: Path):
    """Copy and render simulator instructions with tool list filled in."""
    src = Path(__file__).parent.parent / "src" / "saas_bench" / "agents" / "simulator_instructions.md"
    content = src.read_text()

    # Fill in placeholders
    tool_list = get_tool_summary_table()
    # Calculate total_years from a default 365 days (agents will see their actual value at runtime)
    content = content.replace("{total_days}", "N")
    content = content.replace("{total_years}", "N/365")
    content = content.replace("{tool_list}", tool_list)

    output_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Generate public documentation")
    parser.add_argument("--output", type=str, default="public/docs",
                        help="Output directory (default: public/docs)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Generating documentation...")

    # API docs (JSON, grouped by module)
    api_dir = output / "api"
    render_api_docs(api_dir)
    print(f"  ✅ API docs → {api_dir}/ ({len(list(api_dir.glob('*.json')))} files)")

    # Table docs (JSON, one per table)
    tables_dir = output / "tables"
    render_table_docs(tables_dir)
    print(f"  ✅ Table docs → {tables_dir}/ ({len(list(tables_dir.glob('*.json')))} files)")

    # Tools reference (markdown)
    tools_ref = output / "tools-reference.md"
    render_tools_reference(tools_ref)
    print(f"  ✅ Tools reference → {tools_ref}")

    # Tables reference (markdown)
    tables_ref = output / "tables-reference.md"
    render_tables_reference(tables_ref)
    print(f"  ✅ Tables reference → {tables_ref}")

    # CLI reference (markdown)
    cli_ref = output / "cli-reference.md"
    render_cli_reference(cli_ref)
    print(f"  ✅ CLI reference → {cli_ref}")

    # Simulator instructions
    sim_instructions = output / "simulator-instructions.md"
    render_simulator_instructions(sim_instructions)
    print(f"  ✅ Simulator instructions → {sim_instructions}")

    print(f"\nDone! All docs generated in {output}/")


if __name__ == "__main__":
    main()
