"""Provider-independent LLM client creation and text-call normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


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
    """项目内部统一的用量口径，不等同于所有供应商的原始字段。

    input_tokens 是计费总输入，cached_tokens 是其中命中缓存的部分；
    output_tokens 是按输出单价计费的总输出，推理 Token 通常包含在其中。
    OpenAI/DeepSeek 基本直接符合该口径；Anthropic 的缓存读写量原本
    独立于 input_tokens，必须在本兼容层合并。接入新 Provider 时应依据
    其官方计费文档单独适配，不能只根据字段名称推断包含关系。
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    raw_response: Any


@dataclass(frozen=True)
class TokenCost:
    """One provider-billed amount in its original settlement currency."""

    amount: float
    currency: str

    def as_dict(self) -> dict[str, Any]:
        return {"amount": self.amount, "currency": self.currency}


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
        cache_read_tokens = _int_attr(usage, "cache_read_input_tokens")
        cache_creation_tokens = _int_attr(usage, "cache_creation_input_tokens")
        # TODO: Anthropic 缓存写入存在独立价格，且可能按缓存期限区分。价格模型
        # 支持该维度前禁止静默套用普通输入价格，否则账单金额不可用于论文实验。
        if cache_creation_tokens:
            raise NotImplementedError(
                "Anthropic cache creation pricing is not configured; "
                "disable prompt-cache writes or extend the pricing model"
            )
        return TextLLMResult(
            text=text,
            model=str(getattr(response, "model", None) or model),
            # Anthropic 与 OpenAI/DeepSeek 的原始口径不同：缓存读写量不在
            # input_tokens 中。这里合并成项目统一的“总输入 + 缓存命中子集”。
            input_tokens=(
                _int_attr(usage, "input_tokens")
                + cache_read_tokens
            ),
            output_tokens=_int_attr(usage, "output_tokens"),
            cached_tokens=cache_read_tokens,
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
            cached_tokens=_nested_int_attr(
                usage, "input_tokens_details", "cached_tokens"
            ),
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
            cached_tokens=_nested_int_attr(
                usage, "prompt_tokens_details", "cached_tokens"
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
) -> TokenCost:
    if model not in pricing:
        raise MissingModelPricingError(
            f"No token pricing configured for served model {model!r}; add it to pricing"
        )
    price = pricing[model]
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
    )


def _int_attr(value: Any, name: str) -> int:
    return int(getattr(value, name, 0) or 0) if value is not None else 0


def _nested_int_attr(value: Any, parent: str, name: str) -> int:
    return _int_attr(getattr(value, parent, None), name) if value is not None else 0
