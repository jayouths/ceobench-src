"""OpenAI SDK client creation and response normalization for main experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


API_OPENAI_RESPONSES = "openai_responses"
API_OPENAI_CHAT = "openai_chat_completions"
SUPPORTED_API_TYPES = {
    API_OPENAI_RESPONSES,
    API_OPENAI_CHAT,
}
_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
_TOOL_CHOICES = {"auto", "required"}


class MissingModelPricingError(ValueError):
    """Raised when a response model has no explicit price entry."""


@dataclass(frozen=True)
class TextLLMResult:
    """项目内部统一的用量口径，不等同于所有供应商的原始字段。

    input_tokens 是计费总输入，cached_tokens 是其中命中缓存的部分；
    output_tokens 是按输出单价计费的总输出，推理 Token 通常包含在其中。
    reasoning_tokens 只用于独立观测，不从 output_tokens 中减除，也不重复计费。
    当前主实验只接入 OpenAI SDK 及其兼容协议。接入新协议时，
    必须根据官方计费口径单独适配，不能只根据字段名推断。
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    raw_response: Any


@dataclass(frozen=True)
class TokenCost:
    """One normalized amount calculated from the canonical pricing model."""

    amount: float
    currency: str
    pricing_model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "currency": self.currency,
            "pricing_model": self.pricing_model,
        }


def validate_provider_api_type(provider: str, api_type: str, section: str) -> None:
    if provider != "openai":
        raise ValueError(f"{section}.provider must be 'openai'")
    if api_type not in SUPPORTED_API_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_API_TYPES))
        raise ValueError(f"{section}.api_type must be one of: {allowed}")


def validate_reasoning_effort(
    api_type: str, reasoning_effort: Optional[str], section: str
) -> None:
    """Validate the protocol-level reasoning setting before an experiment starts."""
    if reasoning_effort is None:
        return
    if reasoning_effort not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"{section}.reasoning_effort must be one of: {allowed}")


def validate_tool_choice(tool_choice: Optional[str], section: str) -> None:
    """Validate the provider-independent decision-agent tool policy."""
    if tool_choice not in _TOOL_CHOICES:
        allowed = ", ".join(sorted(_TOOL_CHOICES))
        raise ValueError(f"{section}.tool_choice must be one of: {allowed}")


def api_tool_choice(api_type: str, tool_choice: str) -> str:
    """Validate and return the OpenAI-compatible tool policy."""
    validate_tool_choice(tool_choice, "model")
    if api_type in {API_OPENAI_RESPONSES, API_OPENAI_CHAT}:
        return tool_choice
    raise ValueError(f"Unsupported api_type: {api_type!r}")


def openai_chat_request_parameters(
    *,
    model: str,
    max_output_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    reasoning_effort: Optional[str],
    request_options: Optional[dict[str, Any]] = None,
    tool_choice: Optional[str] = None,
) -> dict[str, Any]:
    """Build one request accepted by OpenAI-compatible Chat endpoints."""
    validate_reasoning_effort(API_OPENAI_CHAT, reasoning_effort, "model")
    # max_tokens 是当前目标端点共同支持的字段；不在主实验层
    # 根据厂商名切换参数。特有参数由 request_options 显式透传。
    params: dict[str, Any] = {"model": model, "max_tokens": max_output_tokens}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    if tool_choice is not None:
        params["tool_choice"] = api_tool_choice(API_OPENAI_CHAT, tool_choice)
    params.update(request_options or {})
    return params


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
            cached_tokens=_nested_int_attr(
                usage, "input_tokens_details", "cached_tokens"
            ),
            reasoning_tokens=_nested_int_attr(
                usage, "output_tokens_details", "reasoning_tokens"
            ),
            raw_response=response,
        )

    if api_type == API_OPENAI_CHAT:
        kwargs = openai_chat_request_parameters(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            request_options=extras,
        )
        kwargs["messages"] = messages
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        return TextLLMResult(
            text=str(getattr(message, "content", "") or "").strip(),
            model=str(getattr(response, "model", None) or model),
            input_tokens=_int_attr(usage, "prompt_tokens"),
            output_tokens=_int_attr(usage, "completion_tokens"),
            cached_tokens=openai_chat_cached_tokens(usage),
            reasoning_tokens=_nested_int_attr(
                usage, "completion_tokens_details", "reasoning_tokens"
            ),
            raw_response=response,
        )

    raise ValueError(f"Unsupported api_type: {api_type!r}")


def token_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    uncached_input_cost_per_million: float,
    cached_input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    """Calculate one call using the provider-reported cache split."""
    for name, value in {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if cached_tokens > input_tokens:
        raise ValueError("cached_tokens cannot exceed input_tokens")

    uncached_tokens = input_tokens - cached_tokens
    return (
        uncached_tokens * uncached_input_cost_per_million
        + cached_tokens * cached_input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000


def model_token_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    pricing: Mapping[str, Mapping[str, Any]],
    pricing_model_map: Optional[Mapping[str, str]] = None,
) -> TokenCost:
    # 渠道请求名和服务端返回名可能不同，统一映射到官方模型后计价。
    pricing_model = (pricing_model_map or {}).get(model, model)
    if pricing_model not in pricing:
        raise MissingModelPricingError(
            f"No token pricing configured for served model {model!r} "
            f"(resolved pricing model {pricing_model!r})"
        )
    price = pricing[pricing_model]
    return TokenCost(
        amount=token_cost(
            input_tokens,
            output_tokens,
            cached_tokens,
            price["uncached_input_cost_per_million"],
            price["cached_input_cost_per_million"],
            price["output_cost_per_million"],
        ),
        currency=str(price["currency"]),
        pricing_model=pricing_model,
    )


def _int_attr(value: Any, name: str) -> int:
    return int(getattr(value, name, 0) or 0) if value is not None else 0


def _nested_int_attr(value: Any, parent: str, name: str) -> int:
    return _int_attr(getattr(value, parent, None), name) if value is not None else 0


def openai_chat_cached_tokens(usage: Any) -> int:
    """Normalize cache hits from OpenAI and DeepSeek Chat Completions.

    OpenAI puts the value in ``prompt_tokens_details.cached_tokens`` while
    DeepSeek returns ``prompt_cache_hit_tokens`` at the top level. If a
    provider returns both fields, disagreement is a protocol error rather than
    something the experiment should silently price.
    """
    nested = _nested_int_attr(usage, "prompt_tokens_details", "cached_tokens")
    deepseek = _int_attr(usage, "prompt_cache_hit_tokens")
    if nested and deepseek and nested != deepseek:
        raise ValueError(
            "Conflicting cached-token counts in Chat Completions usage: "
            f"prompt_tokens_details.cached_tokens={nested}, "
            f"prompt_cache_hit_tokens={deepseek}"
        )
    return deepseek or nested
