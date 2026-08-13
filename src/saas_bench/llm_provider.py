"""Provider-independent LLM client creation and text-call normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


API_ANTHROPIC_MESSAGES = "anthropic_messages"
API_OPENAI_RESPONSES = "openai_responses"
API_OPENAI_CHAT = "openai_chat_completions"
SUPPORTED_API_TYPES = {
    API_ANTHROPIC_MESSAGES,
    API_OPENAI_RESPONSES,
    API_OPENAI_CHAT,
}
_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class MissingModelPricingError(ValueError):
    """Raised when a response model has no explicit price entry."""


@dataclass(frozen=True)
class TextLLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    raw_response: Any


def validate_provider_api_type(provider: str, api_type: str, section: str) -> None:
    supported_providers = {"openai", "openai_compatible", "anthropic", "bedrock"}
    if provider not in supported_providers:
        allowed = ", ".join(sorted(supported_providers))
        raise ValueError(f"{section}.provider must be one of: {allowed}")
    if api_type not in SUPPORTED_API_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_API_TYPES))
        raise ValueError(f"{section}.api_type must be one of: {allowed}")
    if provider in {"anthropic", "bedrock"}:
        if api_type != API_ANTHROPIC_MESSAGES:
            raise ValueError(
                f"{section}: provider {provider!r} requires api_type "
                f"{API_ANTHROPIC_MESSAGES!r}"
            )
    elif api_type == API_ANTHROPIC_MESSAGES:
        raise ValueError(
            f"{section}: api_type {API_ANTHROPIC_MESSAGES!r} requires provider "
            "'anthropic' or 'bedrock'"
        )


def validate_reasoning_effort(
    api_type: str, reasoning_effort: Optional[str], section: str
) -> None:
    """Validate the protocol-level reasoning setting before an experiment starts."""
    if reasoning_effort is None:
        return
    if reasoning_effort not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"{section}.reasoning_effort must be one of: {allowed}")
    if api_type == API_ANTHROPIC_MESSAGES:
        raise ValueError(
            f"{section}.reasoning_effort is not supported for Anthropic Messages; "
            "configure native thinking parameters in request_options"
        )


def create_llm_client(
    *,
    provider: str,
    api_type: str,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout_seconds: float,
):
    """Create the SDK client required by the explicitly selected API type."""
    validate_provider_api_type(provider, api_type, "model")

    if provider == "bedrock":
        from anthropic import AnthropicBedrock
        import os

        region = os.environ.get("AWS_REGION")
        if not region:
            raise ValueError("AWS_REGION must be explicitly configured for bedrock")
        return AnthropicBedrock(
            aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            aws_region=region,
            timeout=timeout_seconds,
        )

    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_text_model(
    *,
    client,
    api_type: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    reasoning_effort: Optional[str],
    request_options: Optional[dict[str, Any]] = None,
) -> TextLLMResult:
    """Call a text model and normalize content and usage across SDK protocols."""
    validate_reasoning_effort(api_type, reasoning_effort, "model")
    sampling: dict[str, Any] = {}
    if temperature is not None:
        sampling["temperature"] = temperature
    if top_p is not None:
        sampling["top_p"] = top_p
    extras = dict(request_options or {})

    if api_type == API_ANTHROPIC_MESSAGES:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            **sampling,
            **extras,
        }
        response = client.messages.create(**kwargs)
        text = "\n".join(
            str(getattr(block, "text", ""))
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "text", None)
        ).strip()
        usage = getattr(response, "usage", None)
        return TextLLMResult(
            text=text,
            model=str(getattr(response, "model", None) or model),
            input_tokens=_int_attr(usage, "input_tokens"),
            output_tokens=_int_attr(usage, "output_tokens"),
            raw_response=response,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if api_type == API_OPENAI_RESPONSES:
        kwargs = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            **sampling,
            **extras,
        }
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**kwargs)
        usage = getattr(response, "usage", None)
        return TextLLMResult(
            text=str(getattr(response, "output_text", "") or "").strip(),
            model=str(getattr(response, "model", None) or model),
            input_tokens=_int_attr(usage, "input_tokens"),
            output_tokens=_int_attr(usage, "output_tokens"),
            raw_response=response,
        )

    if api_type == API_OPENAI_CHAT:
        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
            **sampling,
            **extras,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        return TextLLMResult(
            text=str(getattr(message, "content", "") or "").strip(),
            model=str(getattr(response, "model", None) or model),
            input_tokens=_int_attr(usage, "prompt_tokens"),
            output_tokens=_int_attr(usage, "completion_tokens"),
            raw_response=response,
        )

    raise ValueError(f"Unsupported api_type: {api_type!r}")


def token_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    return (
        input_tokens * input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000


def model_token_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, dict[str, float]],
) -> float:
    if model not in pricing:
        raise MissingModelPricingError(
            f"No token pricing configured for served model {model!r}; add it to pricing"
        )
    price = pricing[model]
    return token_cost_usd(
        input_tokens,
        output_tokens,
        price["input_cost_per_million"],
        price["output_cost_per_million"],
    )


def _int_attr(value: Any, name: str) -> int:
    return int(getattr(value, name, 0) or 0) if value is not None else 0
