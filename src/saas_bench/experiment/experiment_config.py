"""Load reproducible experiment and model settings from TOML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Optional
import tomllib

_SUPPORTED_PROVIDERS = {"openai"}
_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
_MODEL_KEYS = {
    "provider", "api_type", "model", "base_url", "api_key_env", "api_key_required", "reasoning_effort",
    "temperature", "top_p", "tool_choice", "max_output_tokens", "timeout_seconds",
    "pricing", "pricing_model_map", "request_options", "tasks",
}
_TASK_KEYS = {
    "reasoning_effort", "temperature", "top_p", "max_output_tokens", "request_options",
}
_SOCIAL_TASKS = {
    "customer_post", "macro_post", "competitor_post", "agent_post_judge",
    "agent_post_reply",
}
# analysis 模块包含两类调用：四个职能角色分别生成报告，随后统一重构经营状态。
_ANALYSIS_TASKS = {"role_report", "state_reconstruction"}
_EXPERIMENT_KEYS = {
    "name", "seed", "days", "scenario", "initial_cash", "workspace",
    "max_decision_turns_per_batch", "max_invalid_responses_per_turn",
}
_ANALYSIS_MODULE_KEYS = {
    "enabled", "max_schema_retries", "max_enterprise_threads",
}


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = ""
    seed: int = 42
    days: int = 3650
    scenario: str = "default"
    initial_cash: float = 1_000_000.0
    workspace: str = "outputs/runs"
    max_decision_turns_per_batch: int = 100
    max_invalid_responses_per_turn: int = 3


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    api_type: str
    model: str
    max_output_tokens: int
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_required: bool = True
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tool_choice: Optional[str] = None
    timeout_seconds: float = 600.0
    pricing: dict[str, dict[str, Any]] = field(default_factory=dict)
    pricing_model_map: dict[str, str] = field(default_factory=dict)
    request_options: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisModuleSettings:
    """隐性经营状态识别模块的实验开关。"""

    enabled: bool
    max_schema_retries: int
    max_enterprise_threads: int


@dataclass(frozen=True)
class ModuleSettings:
    analysis: AnalysisModuleSettings


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSettings
    modules: ModuleSettings
    decision_agent: ModelSettings
    social_llm: ModelSettings
    analysis: Optional[ModelSettings]

    def simulator_overrides(self) -> dict[str, Any]:
        return _model_overrides("social_post_llm", self.social_llm)


def load_experiment_config(path: Optional[Path]) -> ExperimentConfig:
    if path is None:
        raise ValueError("an explicit experiment config path is required")

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    _reject_unknown(raw, {"experiment", "modules", "models"}, "root")
    experiment = _load_experiment(
        _table(raw.get("experiment"), "experiment"), ExperimentSettings()
    )
    models_raw = _required_table(raw.get("models"), "models")
    _reject_unknown(models_raw, {"decision_agent", "social_llm", "analysis"}, "models")
    decision_agent = _load_model(
        _required_table(models_raw.get("decision_agent"), "models.decision_agent"),
        "models.decision_agent",
        valid_tasks=set(),
        require_tool_choice=True,
    )
    social_llm = _load_model(
        _required_table(models_raw.get("social_llm"), "models.social_llm"),
        "models.social_llm",
        valid_tasks=_SOCIAL_TASKS,
    )

    modules_raw = _required_table(raw.get("modules"), "modules")
    _reject_unknown(modules_raw, {"analysis"}, "modules")
    analysis_module = _load_analysis_module(
        _required_table(modules_raw.get("analysis"), "modules.analysis")
    )

    # 关闭模块时不要求配置无用模型；开启后必须完整声明模型身份和价格，
    # 避免实验运行到第一周才因缺少配置失败。
    analysis_model_raw = models_raw.get("analysis")
    if analysis_module.enabled and analysis_model_raw is None:
        raise ValueError(
            "models.analysis must be explicitly configured when modules.analysis.enabled is true"
        )
    analysis_model = None
    if analysis_model_raw is not None:
        analysis_model = _load_model(
            _required_table(analysis_model_raw, "models.analysis"),
            "models.analysis",
            valid_tasks=_ANALYSIS_TASKS,
        )

    return ExperimentConfig(
        experiment=experiment,
        modules=ModuleSettings(analysis=analysis_module),
        decision_agent=decision_agent,
        social_llm=social_llm,
        analysis=analysis_model,
    )


def _load_analysis_module(raw: Mapping[str, Any]) -> AnalysisModuleSettings:
    _reject_unknown(raw, _ANALYSIS_MODULE_KEYS, "modules.analysis")
    missing = sorted(_ANALYSIS_MODULE_KEYS - set(raw))
    if missing:
        raise ValueError(
            f"modules.analysis must explicitly configure: {', '.join(missing)}"
        )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("modules.analysis.enabled must be a boolean")
    max_schema_retries = raw["max_schema_retries"]
    if (
        not isinstance(max_schema_retries, int)
        or isinstance(max_schema_retries, bool)
        or max_schema_retries < 0
    ):
        raise ValueError(
            "modules.analysis.max_schema_retries must be a non-negative integer"
        )
    max_enterprise_threads = raw["max_enterprise_threads"]
    if (
        not isinstance(max_enterprise_threads, int)
        or isinstance(max_enterprise_threads, bool)
        or max_enterprise_threads <= 0
    ):
        raise ValueError(
            "modules.analysis.max_enterprise_threads must be a positive integer"
        )
    return AnalysisModuleSettings(
        enabled=enabled,
        max_schema_retries=max_schema_retries,
        max_enterprise_threads=max_enterprise_threads,
    )


def _load_experiment(
    raw: Mapping[str, Any], defaults: ExperimentSettings
) -> ExperimentSettings:
    _reject_unknown(raw, _EXPERIMENT_KEYS, "experiment")
    required = {
        "name",
        "max_decision_turns_per_batch",
        "max_invalid_responses_per_turn",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"experiment must explicitly configure: {', '.join(missing)}"
        )
    values = asdict(defaults)
    values.update(raw)
    validate_experiment_name(values["name"])
    if not isinstance(values["seed"], int):
        raise ValueError("experiment.seed must be an integer")
    if not isinstance(values["days"], int) or values["days"] < 0:
        raise ValueError("experiment.days must be a non-negative integer")
    if not isinstance(values["initial_cash"], (int, float)) or values["initial_cash"] <= 0:
        raise ValueError("experiment.initial_cash must be positive")
    if not isinstance(values["scenario"], str) or not values["scenario"]:
        raise ValueError("experiment.scenario must be a non-empty string")
    if not isinstance(values["workspace"], str) or not values["workspace"]:
        raise ValueError("experiment.workspace must be a non-empty string")
    max_turns = values["max_decision_turns_per_batch"]
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
        raise ValueError(
            "experiment.max_decision_turns_per_batch must be a positive integer"
        )
    max_invalid = values["max_invalid_responses_per_turn"]
    if not isinstance(max_invalid, int) or isinstance(max_invalid, bool) or max_invalid <= 0:
        raise ValueError(
            "experiment.max_invalid_responses_per_turn must be a positive integer"
        )
    return ExperimentSettings(**values)


def validate_experiment_name(value: Any) -> str:
    """实验名称同时作为目录名，必须安全、稳定且便于命令行处理。"""
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", value
    ):
        raise ValueError(
            "experiment.name must use lowercase letters, digits, and single hyphens"
        )
    return value


def _load_model(
    raw: Mapping[str, Any],
    section: str,
    valid_tasks: set[str],
    require_tool_choice: bool = False,
) -> ModelSettings:
    _reject_unknown(raw, _MODEL_KEYS, section)
    required = {"provider", "api_type", "model", "max_output_tokens"}
    if require_tool_choice:
        required.add("tool_choice")
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"{section} must explicitly configure: {', '.join(missing)}"
        )

    provider = raw["provider"]
    if provider not in _SUPPORTED_PROVIDERS:
        allowed = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ValueError(f"{section}.provider must be one of: {allowed}")
    if not isinstance(raw["model"], str) or not raw["model"]:
        raise ValueError(f"{section}.model must be a non-empty string")
    api_type = raw["api_type"]
    if not isinstance(api_type, str):
        raise ValueError(f"{section}.api_type must be a string")
    from .llm_provider import (
        validate_provider_api_type,
        validate_reasoning_effort,
        validate_tool_choice,
    )
    validate_provider_api_type(provider, api_type, section)

    pricing = _load_pricing(raw.get("pricing"), section)
    pricing_model_map = _load_pricing_model_map(
        raw.get("pricing_model_map"), section, pricing
    )
    configured_pricing_model = pricing_model_map.get(raw["model"], raw["model"])
    if configured_pricing_model not in pricing:
        raise ValueError(
            f"{section} model {raw['model']!r} resolves to pricing model "
            f"{configured_pricing_model!r}, which is missing from {section}.pricing"
        )

    values = {
        "provider": provider,
        "api_type": api_type,
        "model": raw["model"],
        "base_url": raw.get("base_url"),
        "api_key_env": raw.get("api_key_env"),
        "api_key_required": raw.get("api_key_required", True),
        "reasoning_effort": raw.get("reasoning_effort"),
        "temperature": raw.get("temperature"),
        "top_p": raw.get("top_p"),
        "tool_choice": raw.get("tool_choice"),
        "max_output_tokens": raw["max_output_tokens"],
        "timeout_seconds": raw.get("timeout_seconds", 600.0),
        "pricing": pricing,
        "pricing_model_map": pricing_model_map,
        "request_options": _load_request_options(
            raw.get("request_options"), f"{section}.request_options"
        ),
        "tasks": _load_tasks(raw.get("tasks"), section, valid_tasks),
    }

    temperature = values["temperature"]
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
    ):
        raise ValueError(f"{section}.temperature must be between 0 and 2")
    top_p = values["top_p"]
    if top_p is not None and (
        not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1
    ):
        raise ValueError(f"{section}.top_p must be between 0 and 1")
    max_tokens = values["max_output_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"{section}.max_output_tokens must be a positive integer")
    timeout = values["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"{section}.timeout_seconds must be positive")
    for key in ("base_url", "api_key_env", "reasoning_effort"):
        if values[key] is not None and not isinstance(values[key], str):
            raise ValueError(f"{section}.{key} must be a string")
    if not isinstance(values["api_key_required"], bool):
        raise ValueError(f"{section}.api_key_required must be a boolean")
    if values["api_key_required"] and not values["api_key_env"]:
        raise ValueError(
            f"{section}.api_key_env is required when api_key_required is true"
        )
    reasoning = values["reasoning_effort"]
    validate_reasoning_effort(api_type, reasoning, section)
    if require_tool_choice:
        validate_tool_choice(values["tool_choice"], section)
    elif values["tool_choice"] is not None:
        raise ValueError(f"{section}.tool_choice is only valid for models.decision_agent")

    for task_name, task_values in values["tasks"].items():
        validate_reasoning_effort(
            api_type,
            task_values.get("reasoning_effort", reasoning),
            f"{section}.tasks.{task_name}",
        )

    return ModelSettings(**values)


def _model_overrides(prefix: str, settings: ModelSettings) -> dict[str, Any]:
    return {
        f"{prefix}_provider": settings.provider,
        f"{prefix}_api_type": settings.api_type,
        f"{prefix}_model": settings.model,
        f"{prefix}_base_url": settings.base_url,
        f"{prefix}_api_key_env": settings.api_key_env,
        f"{prefix}_api_key_required": settings.api_key_required,
        f"{prefix}_reasoning_effort": settings.reasoning_effort,
        f"{prefix}_temperature": settings.temperature,
        f"{prefix}_top_p": settings.top_p,
        f"{prefix}_max_tokens": settings.max_output_tokens,
        f"{prefix}_timeout_seconds": settings.timeout_seconds,
        f"{prefix}_pricing": settings.pricing,
        f"{prefix}_pricing_model_map": settings.pricing_model_map,
        f"{prefix}_request_options": settings.request_options,
        f"{prefix}_task_parameters": settings.tasks,
    }


def _load_tasks(
    value: Any, section: str, valid_tasks: set[str]
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    tasks = _table(value, f"{section}.tasks")
    result: dict[str, dict[str, Any]] = {}
    for name, raw_task in tasks.items():
        if name not in valid_tasks:
            allowed = ", ".join(sorted(valid_tasks)) or "none"
            raise ValueError(
                f"unknown {section}.tasks entry {name!r}; allowed tasks: {allowed}"
            )
        task_section = f"{section}.tasks.{name}"
        task = _table(raw_task, task_section)
        _reject_unknown(task, _TASK_KEYS, task_section)
        values = dict(task)
        values["request_options"] = _load_request_options(
            values.get("request_options"), f"{task_section}.request_options"
        )
        temperature = values.get("temperature")
        if temperature is not None and (
            not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
        ):
            raise ValueError(f"{task_section}.temperature must be between 0 and 2")
        top_p = values.get("top_p")
        if top_p is not None and (
            not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1
        ):
            raise ValueError(f"{task_section}.top_p must be between 0 and 1")
        max_tokens = values.get("max_output_tokens")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or max_tokens <= 0
        ):
            raise ValueError(f"{task_section}.max_output_tokens must be positive")
        reasoning = values.get("reasoning_effort")
        if reasoning is not None and reasoning not in _REASONING_EFFORTS:
            allowed = ", ".join(sorted(_REASONING_EFFORTS))
            raise ValueError(f"{task_section}.reasoning_effort must be one of: {allowed}")
        result[str(name)] = values
    return result


def _load_request_options(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    options = dict(_table(value, section))
    allowed = {"extra_body", "extra_headers", "extra_query"}
    _reject_unknown(options, allowed, section)
    for key, item in options.items():
        if not isinstance(item, dict):
            raise ValueError(f"{section}.{key} must be a TOML table")
    return options


def _load_pricing(value: Any, section: str) -> dict[str, dict[str, Any]]:
    raw = _required_table(value, f"{section}.pricing")
    result: dict[str, dict[str, Any]] = {}
    for model, raw_price in raw.items():
        price_section = f"{section}.pricing.{model}"
        price = dict(_table(raw_price, price_section))
        keys = {
            "currency",
            "uncached_input_cost_per_million",
            "cached_input_cost_per_million",
            "output_cost_per_million",
        }
        _reject_unknown(price, keys, price_section)
        missing = sorted(keys - set(price))
        if missing:
            raise ValueError(
                f"{price_section} must explicitly configure: {', '.join(missing)}"
            )
        currency = price["currency"]
        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isascii()
            or not currency.isalpha()
            or currency != currency.upper()
        ):
            raise ValueError(
                f"{price_section}.currency must be an uppercase three-letter code"
            )
        numeric_keys = keys - {"currency"}
        for key in numeric_keys:
            if not isinstance(price[key], (int, float)) or price[key] < 0:
                raise ValueError(f"{price_section}.{key} must be non-negative")
        result[str(model)] = {
            "currency": currency,
            **{key: float(price[key]) for key in numeric_keys},
        }
    return result


def _load_pricing_model_map(
    value: Any,
    section: str,
    pricing: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    if value is None:
        return {}
    raw = _table(value, f"{section}.pricing_model_map")
    result: dict[str, str] = {}
    for channel_model, official_model in raw.items():
        if not isinstance(channel_model, str) or not channel_model:
            raise ValueError(
                f"{section}.pricing_model_map keys must be non-empty strings"
            )
        if not isinstance(official_model, str) or not official_model:
            raise ValueError(
                f"{section}.pricing_model_map.{channel_model} must be a non-empty string"
            )
        if official_model not in pricing:
            raise ValueError(
                f"{section}.pricing_model_map.{channel_model} targets unknown "
                f"pricing model {official_model!r}"
            )
        result[channel_model] = official_model
    return result


def _table(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a TOML table")
    return value


def _required_table(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"{section} must be explicitly configured")
    return _table(value, section)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} setting(s): {', '.join(unknown)}")
