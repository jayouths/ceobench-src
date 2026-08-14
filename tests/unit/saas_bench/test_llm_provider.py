from types import SimpleNamespace

import pytest

from saas_bench.llm_provider import (
    API_ANTHROPIC_MESSAGES,
    API_OPENAI_CHAT,
    API_OPENAI_RESPONSES,
    MissingModelPricingError,
    call_text_model,
    model_token_cost,
    openai_chat_cached_tokens,
    token_cost,
)


class Recorder:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_openai_responses_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        output_text="  response text  ",
        model="served-responses",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=7),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    ))
    client = SimpleNamespace(responses=recorder)

    result = call_text_model(
        client=client,
        api_type=API_OPENAI_RESPONSES,
        model="requested",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=100,
        temperature=0.4,
        top_p=None,
        reasoning_effort=None,
        request_options={"extra_body": {"enable_thinking": False}},
    )

    assert result.text == "response text"
    assert (result.model, result.input_tokens, result.output_tokens) == (
        "served-responses", 12, 5,
    )
    assert result.cached_tokens == 7
    assert result.reasoning_tokens == 3
    assert recorder.calls == [{
        "model": "requested",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "max_output_tokens": 100,
        "temperature": 0.4,
        "extra_body": {"enable_thinking": False},
    }]


def test_openai_chat_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        model="served-chat",
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="chat text"))],
    ))
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))

    result = call_text_model(
        client=client,
        api_type=API_OPENAI_CHAT,
        model="requested",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=50,
        temperature=None,
        top_p=0.8,
        reasoning_effort="high",
    )

    assert (result.text, result.model, result.input_tokens, result.output_tokens) == (
        "chat text", "served-chat", 8, 3,
    )
    assert result.cached_tokens == 2
    assert result.reasoning_tokens == 1
    assert recorder.calls[0]["reasoning_effort"] == "high"
    assert recorder.calls[0]["max_completion_tokens"] == 50
    assert recorder.calls[0]["top_p"] == pytest.approx(0.8)


def test_deepseek_chat_cache_hit_tokens_are_normalized():
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        prompt_cache_hit_tokens=9,
        prompt_cache_miss_tokens=3,
    )
    recorder = Recorder(SimpleNamespace(
        model="deepseek-v4-pro",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
    ))

    result = call_text_model(
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
        api_type=API_OPENAI_CHAT,
        model="deepseek-v4-pro",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=50,
        temperature=None,
        top_p=None,
        reasoning_effort="high",
    )

    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.cached_tokens == 9
    assert result.reasoning_tokens == 0


def test_chat_cache_fields_must_not_disagree():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        prompt_cache_hit_tokens=3,
    )

    with pytest.raises(ValueError, match="Conflicting cached-token counts"):
        openai_chat_cached_tokens(usage)


def test_anthropic_messages_request_and_normalized_result():
    recorder = Recorder(SimpleNamespace(
        model="served-anthropic",
        content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        usage=SimpleNamespace(
            input_tokens=20,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=0,
            output_tokens=9,
        ),
    ))
    client = SimpleNamespace(messages=recorder)

    result = call_text_model(
        client=client,
        api_type=API_ANTHROPIC_MESSAGES,
        model="claude-sonnet-test",
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=200,
        temperature=None,
        top_p=None,
        reasoning_effort=None,
        request_options={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
    )

    assert (result.text, result.model, result.input_tokens, result.output_tokens) == (
        "first\nsecond", "served-anthropic", 25, 9,
    )
    assert result.cached_tokens == 5
    assert result.reasoning_tokens == 0
    assert recorder.calls[0]["thinking"] == {"type": "adaptive"}
    assert recorder.calls[0]["output_config"] == {"effort": "medium"}


def test_anthropic_cache_creation_fails_until_its_price_is_supported():
    recorder = Recorder(SimpleNamespace(
        model="served-anthropic",
        content=[SimpleNamespace(text="response")],
        usage=SimpleNamespace(
            input_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=3,
            output_tokens=9,
        ),
    ))

    with pytest.raises(NotImplementedError, match="cache creation pricing"):
        call_text_model(
            client=SimpleNamespace(messages=recorder),
            api_type=API_ANTHROPIC_MESSAGES,
            model="claude-sonnet-test",
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=200,
            temperature=None,
            top_p=None,
            reasoning_effort=None,
        )


def test_cost_uses_served_model_and_rejects_unknown_model():
    pricing = {
        "requested": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.1,
            "output_cost_per_million": 2.0,
        },
        "served": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 3.0,
            "cached_input_cost_per_million": 0.25,
            "output_cost_per_million": 4.0,
        },
    }

    cost = model_token_cost("served", 1_000_000, 1_000_000, 250_000, pricing)
    assert cost.amount == pytest.approx(6.3125)
    assert cost.currency == "CNY"
    assert cost.pricing_model == "served"
    with pytest.raises(MissingModelPricingError, match="unlisted"):
        model_token_cost("unlisted", 1, 1, 0, pricing)


def test_cost_maps_channel_model_to_canonical_pricing_model():
    pricing = {
        "deepseek-v4-pro": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.32,
            "cached_input_cost_per_million": 0.044,
            "output_cost_per_million": 3.96,
        }
    }

    cost = model_token_cost(
        "DeepSeek-V4-Pro-0813",
        1_000_000,
        1_000_000,
        250_000,
        pricing,
        {"DeepSeek-V4-Pro-0813": "deepseek-v4-pro"},
    )

    assert cost.amount == pytest.approx(4.961)
    assert cost.currency == "USD"
    assert cost.pricing_model == "deepseek-v4-pro"


@pytest.mark.parametrize(
    ("cached_tokens", "expected"),
    [(0, 3.0), (250_000, 2.525), (1_000_000, 1.1)],
)
def test_cost_splits_cached_and_uncached_input(cached_tokens, expected):
    assert token_cost(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_tokens=cached_tokens,
        uncached_input_cost_per_million=2.0,
        cached_input_cost_per_million=0.1,
        output_cost_per_million=10.0,
    ) == pytest.approx(expected)


def test_cost_rejects_cached_tokens_above_total_input():
    with pytest.raises(ValueError, match="cannot exceed"):
        token_cost(10, 1, 11, 1.0, 0.1, 2.0)
