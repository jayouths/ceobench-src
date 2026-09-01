"""实验运行目录的配置身份读取与校验。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from saas_bench.experiment.experiment_config import load_experiment_config


RUN_CONFIG_FIELDS = {
    "run_id", "experiment_name", "agent_type", "model", "provider", "api_type",
    "base_url", "reasoning_effort", "temperature", "top_p", "tool_choice",
    "max_output_tokens", "timeout_seconds", "request_options", "pricing",
    "pricing_model_map", "api_key_env", "api_key_required", "seed", "scenario",
    "total_days", "initial_cash", "max_decision_turns_per_batch",
    "max_invalid_responses_per_turn", "simulator_llm", "analysis_module",
    "analysis_model", "git_commit",
}


def resolve_run_directory(value: str, search_root: Path | None = None) -> Path:
    """将 run id 或目录解析为唯一的实验运行目录。"""
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()

    candidates: list[Path] = []
    root_path = Path.cwd() if search_root is None else Path(search_root)
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", ".venv", "__pycache__", "tmp"}
        ]
        path = Path(root)
        if "config.json" not in files:
            continue
        try:
            saved = json.loads((path / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(saved, dict) and saved.get("run_id") == value:
            candidates.append(path.resolve())
            dirs[:] = []
    if not candidates:
        raise FileNotFoundError(f"No run directory found for resume id {value!r}")
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"Resume id {value!r} is ambiguous; pass one directory: {joined}"
        )
    return candidates[0]


def load_saved_run_config(run_dir: Path) -> dict[str, Any]:
    """读取断点恢复所需的原实验配置，禁止使用当前 TOML 覆盖。"""
    config_path = Path(run_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    try:
        saved = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run config is invalid JSON: {config_path}") from exc
    if not isinstance(saved, dict):
        raise ValueError("Run config root must be an object")

    missing = RUN_CONFIG_FIELDS - saved.keys()
    if missing:
        raise ValueError(f"Run config is missing fields: {sorted(missing)}")
    if saved["agent_type"] != "bash_agent":
        raise ValueError(f"Run config is not for bash_agent: {saved['agent_type']!r}")
    if not isinstance(saved["run_id"], str) or not saved["run_id"]:
        raise ValueError("Run config run_id must be a non-empty string")
    if not isinstance(saved["git_commit"], str) or not saved["git_commit"]:
        raise ValueError("Run config git_commit must be a non-empty string")
    return saved


def create_new_runner(config_path: Path):
    """根据一份完整 TOML 创建新实验 Runner。"""
    # 局部导入避免 runner 在初始化时与配置模块循环依赖。
    from .runner import BashAgentRunner

    file_config = load_experiment_config(config_path)
    experiment = file_config.experiment
    decision = file_config.decision_agent
    return BashAgentRunner(
        model=decision.model,
        experiment_name=experiment.name,
        provider=decision.provider,
        api_type=decision.api_type,
        base_url=decision.base_url,
        api_key_env=decision.api_key_env,
        api_key_required=decision.api_key_required,
        seed=experiment.seed,
        scenario=experiment.scenario,
        total_days=experiment.days,
        initial_cash=experiment.initial_cash,
        max_decision_turns_per_batch=experiment.max_decision_turns_per_batch,
        max_invalid_responses_per_turn=experiment.max_invalid_responses_per_turn,
        workspace_base=Path(experiment.workspace),
        reasoning_effort=decision.reasoning_effort,
        temperature=decision.temperature,
        top_p=decision.top_p,
        tool_choice=decision.tool_choice,
        max_output_tokens=decision.max_output_tokens,
        timeout_seconds=decision.timeout_seconds,
        request_options=decision.request_options,
        pricing=decision.pricing,
        pricing_model_map=decision.pricing_model_map,
        simulator_llm_config=file_config.simulator_overrides(),
        analysis_module_config=asdict(file_config.modules.analysis),
        analysis_model_config=(
            file_config.analysis.as_dict() if file_config.analysis else None
        ),
    )


def create_resumed_runner(value: str):
    """仅根据原运行目录中的不可变配置创建恢复 Runner。"""
    from .runner import BashAgentRunner

    run_dir = resolve_run_directory(value)
    saved = load_saved_run_config(run_dir)
    return BashAgentRunner(
        model=saved["model"],
        experiment_name=saved["experiment_name"],
        provider=saved["provider"],
        api_type=saved["api_type"],
        base_url=saved["base_url"],
        api_key_env=saved["api_key_env"],
        api_key_required=saved["api_key_required"],
        seed=saved["seed"],
        scenario=saved["scenario"],
        total_days=saved["total_days"],
        initial_cash=saved["initial_cash"],
        max_decision_turns_per_batch=saved["max_decision_turns_per_batch"],
        max_invalid_responses_per_turn=saved["max_invalid_responses_per_turn"],
        reasoning_effort=saved["reasoning_effort"],
        temperature=saved["temperature"],
        top_p=saved["top_p"],
        tool_choice=saved["tool_choice"],
        max_output_tokens=saved["max_output_tokens"],
        timeout_seconds=saved["timeout_seconds"],
        request_options=saved["request_options"],
        pricing=saved["pricing"],
        pricing_model_map=saved["pricing_model_map"],
        simulator_llm_config=saved["simulator_llm"],
        analysis_module_config=saved["analysis_module"],
        analysis_model_config=saved["analysis_model"],
        git_commit=saved["git_commit"],
        continue_from=run_dir,
    )
