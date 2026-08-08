"""Load reproducible experiment and model settings from TOML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import tomllib

_API_KEY_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "modal": "MODAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "ai_sandbox": "AI_SANDBOX_KEY",
}

_BASE_URL_BY_PROVIDER = {
    "xai": "https://api.x.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "together": "https://api.together.xyz/v1",
}

_DECISION_PROVIDERS = {
    "openai", "xai", "google", "anthropic", "bedrock", "modal",
    "together", "ai_sandbox",
}
_SIMULATOR_PROVIDERS = {"openai", "anthropic", "bedrock"}
_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
_MODEL_KEYS = {
    "provider", "model", "base_url", "api_key_env", "api_key_required", "reasoning_effort",
    "temperature", "top_p", "max_output_tokens", "timeout_seconds",
    "input_cost_per_million", "output_cost_per_million",
}
_EXPERIMENT_KEYS = {
    "seed", "days", "scenario", "initial_cash", "workspace", "label",
}


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 42
    days: int = 3650
    scenario: str = "default"
    initial_cash: float = 1_000_000.0
    workspace: str = "bash_agent_runs"
    label: Optional[str] = None


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_required: bool = True
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout_seconds: float = 600.0
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSettings
    decision_agent: ModelSettings
    social_llm: ModelSettings
    enterprise_llm: ModelSettings

    def simulator_overrides(self) -> dict[str, Any]:
        return {
            **_model_overrides("social_post_llm", self.social_llm),
            **_model_overrides("enterprise_llm", self.enterprise_llm),
        }


def load_experiment_config(path: Optional[Path]) -> ExperimentConfig:
    if path is None:
        raise ValueError(
            "an explicit experiment config is required; pass --config <path>"
        )

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    _reject_unknown(raw, {"experiment", "models"}, "root")
    experiment_raw = _table(raw.get("experiment"), "experiment")
    models_raw = _required_table(raw.get("models"), "models")
    _reject_unknown(models_raw, {"decision_agent", "social_llm", "enterprise_llm"}, "models")

    return ExperimentConfig(
        experiment=_load_experiment(experiment_raw, ExperimentSettings()),
        decision_agent=_load_model(
            _required_table(models_raw.get("decision_agent"), "models.decision_agent"),
            _DECISION_PROVIDERS,
            "models.decision_agent",
            default_max_output_tokens=16_384,
        ),
        social_llm=_load_model(
            _required_table(models_raw.get("social_llm"), "models.social_llm"),
            _SIMULATOR_PROVIDERS,
            "models.social_llm",
            default_max_output_tokens=1_000,
        ),
        enterprise_llm=_load_model(
            _required_table(models_raw.get("enterprise_llm"), "models.enterprise_llm"),
            _SIMULATOR_PROVIDERS,
            "models.enterprise_llm",
            default_max_output_tokens=300,
        ),
    )


def _load_experiment(
    raw: Mapping[str, Any], defaults: ExperimentSettings
) -> ExperimentSettings:
    _reject_unknown(raw, _EXPERIMENT_KEYS, "experiment")
    values = asdict(defaults)
    values.update(raw)
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
    if values["label"] is not None and not isinstance(values["label"], str):
        raise ValueError("experiment.label must be a string")
    return ExperimentSettings(**values)


def _load_model(
    raw: Mapping[str, Any],
    valid_providers: set[str],
    section: str,
    default_max_output_tokens: int,
) -> ModelSettings:
    _reject_unknown(raw, _MODEL_KEYS, section)
    missing = sorted({"provider", "model"} - set(raw))
    if missing:
        raise ValueError(
            f"{section} must explicitly configure: {', '.join(missing)}"
        )

    provider = raw["provider"]
    if provider not in valid_providers:
        allowed = ", ".join(sorted(valid_providers))
        raise ValueError(f"{section}.provider must be one of: {allowed}")
    if not isinstance(raw["model"], str) or not raw["model"]:
        raise ValueError(f"{section}.model must be a non-empty string")

    values = {
        "provider": provider,
        "model": raw["model"],
        "base_url": raw.get("base_url", default_base_url(provider)),
        "api_key_env": raw.get("api_key_env", default_api_key_env(provider)),
        "api_key_required": raw.get("api_key_required", True),
        "reasoning_effort": raw.get("reasoning_effort"),
        "temperature": raw.get("temperature"),
        "top_p": raw.get("top_p"),
        "max_output_tokens": raw.get(
            "max_output_tokens", default_max_output_tokens
        ),
        "timeout_seconds": raw.get("timeout_seconds", 600.0),
        "input_cost_per_million": raw.get("input_cost_per_million"),
        "output_cost_per_million": raw.get("output_cost_per_million"),
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
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ValueError(f"{section}.max_output_tokens must be a positive integer")
    timeout = values["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"{section}.timeout_seconds must be positive")
    for key in ("input_cost_per_million", "output_cost_per_million"):
        price = values[key]
        if price is not None and (
            not isinstance(price, (int, float)) or price < 0
        ):
            raise ValueError(f"{section}.{key} must be non-negative")
    configured_prices = (
        values["input_cost_per_million"] is not None,
        values["output_cost_per_million"] is not None,
    )
    if configured_prices[0] != configured_prices[1]:
        raise ValueError(
            f"{section} must configure input and output costs together"
        )
    for key in ("base_url", "api_key_env", "reasoning_effort"):
        if values[key] is not None and not isinstance(values[key], str):
            raise ValueError(f"{section}.{key} must be a string")
    if not isinstance(values["api_key_required"], bool):
        raise ValueError(f"{section}.api_key_required must be a boolean")
    reasoning = values["reasoning_effort"]
    if reasoning is not None and reasoning not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"{section}.reasoning_effort must be one of: {allowed}")

    return ModelSettings(**values)


def _model_overrides(prefix: str, settings: ModelSettings) -> dict[str, Any]:
    return {
        f"{prefix}_provider": settings.provider,
        f"{prefix}_model": settings.model,
        f"{prefix}_base_url": settings.base_url,
        f"{prefix}_api_key_env": settings.api_key_env,
        f"{prefix}_api_key_required": settings.api_key_required,
        f"{prefix}_reasoning_effort": settings.reasoning_effort,
        f"{prefix}_temperature": settings.temperature,
        f"{prefix}_top_p": settings.top_p,
        f"{prefix}_max_tokens": settings.max_output_tokens,
        f"{prefix}_timeout_seconds": settings.timeout_seconds,
        f"{prefix}_input_cost_per_million": settings.input_cost_per_million,
        f"{prefix}_output_cost_per_million": settings.output_cost_per_million,
    }


def default_api_key_env(provider: str) -> Optional[str]:
    return _API_KEY_ENV_BY_PROVIDER.get(provider)


def default_base_url(provider: str) -> Optional[str]:
    return _BASE_URL_BY_PROVIDER.get(provider)


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
